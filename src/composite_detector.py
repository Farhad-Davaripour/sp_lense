"""One ModernBERT encoder, detection LoRA and structured/evidence heads."""
import torch
from torch import nn
from .composite_lora import install,enable,trainables,state,load
from .composite_memory import Prediction,STATUSES,EVENTS,AFFECTED
from .composite_capacity import detector_targets,capacity,checkpointed_training

LABELS={'own_status':STATUSES,'peer_status':STATUSES,'event':EVENTS,'affected':AFFECTED}


class StructuredHeads(nn.Module):
    def __init__(self,hidden,head_mode='cls'):
        super().__init__()
        if head_mode not in ['cls','role_pool']:raise ValueError('Unsupported detector head mode')
        self.head_mode=head_mode;sizes={k:hidden for k in LABELS}
        if head_mode=='role_pool':sizes.update(own_status=hidden*2,peer_status=hidden*2,affected=hidden*3)
        self.fields=nn.ModuleDict({name:nn.Linear(sizes[name],len(labels)) for name,labels in LABELS.items()});self.evidence=nn.Linear(hidden,2)
    def forward(self,hidden,mask,roles=None):
        cls=hidden[:,0].float();features={k:cls for k in LABELS}
        if self.head_mode=='role_pool':
            if roles is None or roles.shape!=(hidden.shape[0],2,hidden.shape[1]) or not roles.any(-1).all() or (roles & ~mask[:,None].bool()).any():raise ValueError('Valid visible role-token pools required')
            weights=roles.float();pooled=torch.bmm(weights,hidden.float())/weights.sum(-1,keepdim=True);own,peer=pooled.unbind(1)
            features.update(own_status=torch.cat([cls,own],-1),peer_status=torch.cat([cls,peer],-1),affected=torch.cat([cls,own,peer],-1))
        logits={k:v(features[k]) for k,v in self.fields.items()};spans=self.evidence(hidden.float()).masked_fill(~mask[:,:,None].bool(),-1e9)
        return logits,spans


class Detector(nn.Module):
    def __init__(self,path,cfg,device='cuda:0',head_mode='cls'):
        super().__init__()
        from .composite_data import require_databricks
        require_databricks()
        from transformers import AutoTokenizer,AutoModel
        self.cfg=cfg;self.head_mode=head_mode;self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
        self.encoder=AutoModel.from_pretrained(path,local_files_only=True,trust_remote_code=False,use_safetensors=True,dtype=torch.float32,attn_implementation='sdpa').to(device).eval().requires_grad_(False)
        if self.encoder.config.hidden_size!=cfg['hidden_size'] or self.encoder.config.num_hidden_layers!=cfg['layers']:raise ValueError('ModernBERT pin geometry differs')
        self.base_references=list(self.encoder.named_parameters());names=detector_targets(self.encoder,cfg)
        self.capacity_proof=capacity(self.encoder,names,cfg['rank'])
        self.adapters=install(self.encoder,names,cfg['rank'],cfg['alpha'],True);self.heads=StructuredHeads(cfg['hidden_size'],head_mode).to(device)
        self.device=device
        self.on_forward=None
    def training_context(self):return checkpointed_training(self.encoder,self.encoder.layers,self.cfg.get('gradient_checkpointing',False))
    def encode(self,texts):
        encoded=self.tokenizer(texts,padding=True,truncation=False,return_offsets_mapping=True,return_tensors='pt')
        if encoded['input_ids'].shape[1]>self.cfg['maximum_tokens']:raise ValueError('Complete visible history exceeds detector context')
        offsets=encoded.pop('offset_mapping');inputs={k:v.to(self.device) for k,v in encoded.items() if k in ['input_ids','attention_mask']}
        if self.head_mode=='role_pool':
            from .composite_roles import token_masks
            inputs['role_masks']=token_masks(texts,offsets).to(self.device)
        return inputs,offsets
    def forward(self,inputs):
        if self.on_forward:self.on_forward('attempt')
        hidden=self.encoder(**{k:v for k,v in inputs.items() if k!='role_masks'},return_dict=True).last_hidden_state
        value=self.heads(hidden,inputs['attention_mask'],inputs.get('role_masks'))
        if self.on_forward:self.on_forward('complete')
        return value
    def parameters_to_fit(self):return trainables(self.adapters)+list(self.heads.parameters())
    def checkpoint(self):return {'lora':state(self.adapters),'heads':{k:v.detach().cpu().clone() for k,v in self.heads.state_dict().items()},'config':self.cfg,'placements':list(self.adapters),'head_mode':self.head_mode}
    def restore(self,value):
        if value['config']!=self.cfg or value['placements']!=list(self.adapters) or value.get('head_mode','cls')!=self.head_mode:raise ValueError('Detector adapter/backbone contract mismatch')
        load(self.adapters,value['lora']);self.heads.load_state_dict(value['heads'])
    def predict(self,text,threshold):
        self.eval();inputs,offsets=self.encode([text])
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.device.startswith('cuda')):logits,evidence=self.forward(inputs)
        probabilities={k:v[0].float().softmax(-1).cpu().tolist() for k,v in logits.items()};raw={k:LABELS[k][max(range(len(v)),key=lambda i:v[i])] for k,v in probabilities.items()}
        confidence=min(max(v) for v in probabilities.values());scores=evidence[0].float().cpu();length=int(inputs['attention_mask'][0].sum());best=None
        if length>2:
            ends=scores[1:length-1,1];pooled,indices=nn.functional.max_pool1d(nn.functional.pad(ends[None,None],(0,127),value=-1e9),128,stride=1,return_indices=True)
            totals=scores[1:length-1,0]+pooled[0,0];start=int(totals.argmax())+1;end=int(indices[0,0,start-1])+1;best=(float(totals[start-1]),start,end)
        if best is None:begin=finish=0
        else:begin=int(offsets[0,best[1],0]);finish=int(offsets[0,best[2],1])
        effective=dict(raw)
        consistent=(raw['affected']=='none' and raw['own_status']==raw['peer_status']=='clear') or (raw['affected']=='self' and raw['own_status'] in ['active','withdrawn'] and raw['peer_status']=='clear') or (raw['affected']=='peer' and raw['peer_status'] in ['active','withdrawn'] and raw['own_status']=='clear') or (raw['affected']=='both' and raw['own_status'] in ['active','withdrawn'] and raw['peer_status'] in ['active','withdrawn']) or (raw['affected']=='unknown' and 'unknown' in [raw['own_status'],raw['peer_status']])
        if confidence<threshold or not consistent or begin>=finish:effective={k:'unknown' for k in LABELS}
        details={'labels':raw,'probabilities':probabilities,'consistent':consistent,'evidence_token_indices':list(best[1:]) if best else [],'input_token_ids':inputs['input_ids'][0].cpu().tolist(),'offsets':offsets[0].tolist(),'evidence_logits':scores.tolist(),'head_mode':self.head_mode}
        if self.head_mode=='role_pool':
            from .composite_roles import char_spans
            details.update(role_char_spans=char_spans(text),role_token_masks=inputs['role_masks'][0].cpu().tolist())
        return Prediction(**effective,confidence=confidence,evidence_start=begin,evidence_end=finish,evidence_text=text[begin:finish],raw=details)

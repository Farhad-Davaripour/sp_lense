"""File/config identity, template/label alignment and token-category loss audit."""
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
from model_ops import ids, save, base_hash


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''): h.update(block)
    return h.hexdigest()


def verify_freeze(root):
    code=root/'code'
    for name,expected in json.loads((code/'FREEZE.json').read_text())['sha256'].items():
        if sha(code/name)!=expected: raise RuntimeError('Source/data freeze mismatch: '+name)
    return json.loads((code/'config.json').read_text())


def verify_adapter(path,weight_hash,config_hash):
    if sha(Path(path)/'adapter_model.safetensors')!=weight_hash: raise RuntimeError('Adapter weight identity mismatch')
    if sha(Path(path)/'adapter_config.json')!=config_hash: raise RuntimeError('Adapter config identity mismatch')


def runtime(model,tokenizer,root,label):
    import torch
    active=getattr(model,'active_adapters',[])
    if callable(active): active=active()
    if isinstance(active,str):active=[active]
    config=model.peft_config['default']
    data={'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,
          'gpu':torch.cuda.get_device_name(0),'active_adapters':active,'rank':config.r,'alpha':config.lora_alpha,
          'packages':{n:importlib.metadata.version(n) for n in ('transformers','peft','bitsandbytes','accelerate','safetensors','huggingface_hub')},
          'tokenizer_class':type(tokenizer).__name__,'template_sha256':hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest(),
          'tokenizer_eos':tokenizer.eos_token,'tokenizer_eos_id':tokenizer.eos_token_id,
          'generation_config':model.generation_config.to_dict(),'packed_base_parameter_hash':base_hash(model)}
    if active!=['default'] or config.r!=16 or config.lora_alpha!=32: raise RuntimeError('Unexpected active H2 structure')
    save(root/'training/receipts'/('runtime_'+label+'.json'),data)
    return data


def row_tokens(tokenizer,row,padded_length=None):
    prefix=ids(tokenizer,row['messages'],row.get('tools'))
    text=row['targets']['preservation']
    response=tokenizer.encode(text+tokenizer.eos_token,add_special_tokens=False)
    actual=len(prefix)+len(response)
    padded_length=actual if padded_length is None else padded_length
    if actual>padded_length or padded_length>3072: raise RuntimeError('Fixed training token cap/alignment')
    pad=tokenizer.pad_token_id or tokenizer.eos_token_id
    x=prefix+response+[pad]*(padded_length-actual)
    labels=[-100]*len(prefix)+response+[-100]*(padded_length-actual)
    mask=[1]*actual+[0]*(padded_length-actual)
    assert labels[len(prefix):actual]==response and all(v==-100 for v in labels[:len(prefix)]+labels[actual:])
    positions=[i for i in range(len(labels)-1) if labels[i+1]!=-100]
    assert positions==list(range(len(prefix)-1,actual-1))
    assert response[-1]==tokenizer.eos_token_id
    return {'prefix':prefix,'response':response,'x':x,'labels':labels,'attention':mask,'positions':positions,
            'decoded_prefix_tail':tokenizer.decode(prefix[-100:],skip_special_tokens=False),
            'decoded_target':tokenizer.decode(response,skip_special_tokens=False),
            'target_literal':text+tokenizer.eos_token,'actual_length':actual,'padded_length':padded_length,
            'decoded_target_matches_literal':tokenizer.decode(response,skip_special_tokens=False)==text+tokenizer.eos_token}


def alignment_audit(tokenizer,rows,root,label):
    output=[]
    for row in rows:
        t=row_tokens(tokenizer,row)
        output.append({'id':row['id'],**t,'prefix_sha256':hashlib.sha256(json.dumps(t['prefix']).encode()).hexdigest(),
           'target_sha256':hashlib.sha256(json.dumps(t['response']).encode()).hexdigest(),
           'prefix_generation_prompt':True,'prefix_labels_masked':True,'padding_labels_masked':True,
           'next_token_shift_verified':True,'target_eos_verified':True})
    save(root/'training/receipts'/('alignment_'+label+'.json'),output)
    return all(r['decoded_target_matches_literal'] for r in output)


def loss_audit(model,tokenizer,rows,root,label):
    import torch
    import torch.nn.functional as F
    results=[]
    model.eval()
    for row in rows:
        t=row_tokens(tokenizer,row)
        x=torch.tensor([t['x']],device='cuda');mask=torch.tensor([t['attention']],device='cuda')
        positions=torch.tensor(t['positions'],device='cuda')
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(input_ids=x,attention_mask=mask,use_cache=False,logits_to_keep=positions).logits[0].float()
            per_token=F.cross_entropy(logits,torch.tensor(t['response'],device='cuda'),reduction='none')
        if not torch.isfinite(per_token).all(): raise RuntimeError('Nonfinite decision loss')
        values=per_token.cpu().tolist()
        categories=None
        if tokenizer.is_fast:
            encoded=tokenizer(row['targets']['preservation']+tokenizer.eos_token,add_special_tokens=False,return_offsets_mapping=True)
            assert encoded['input_ids']==t['response']
            text=row['targets']['preservation'];begin=text.find('<tool_call>');end=text.rfind('</tool_call>')+len('</tool_call>')
            categories=[]
            for index,(a,b) in enumerate(encoded['offset_mapping']):
                categories.append('eos' if index==len(t['response'])-1 else 'action' if begin>=0 and b>begin and a<end else 'non_action_text')
        buckets={}
        if categories is not None:
            for category,value in zip(categories,values):
                bucket=buckets.setdefault(category,{'n':0,'loss_sum':0.0});bucket['n']+=1;bucket['loss_sum']+=value
            for bucket in buckets.values():bucket['mean_loss']=bucket['loss_sum']/bucket['n']
        results.append({'id':row['id'],'panel':row['audit_panel'],'total_mean_loss':sum(values)/len(values),
            'supervised_tokens':len(values),'token_categories':buckets,'decoded_target':t['decoded_target'],
            'response_token_ids':t['response'],'per_token_loss':values,'category_split_available':categories is not None})
        del x,mask,positions,logits,per_token
    panels={}
    for r in results:
        panel=panels.setdefault(r['panel'],{'n':0,'row_loss_sum':0.0,'categories':{}})
        panel['n']+=1;panel['row_loss_sum']+=r['total_mean_loss']
        for k,b in r['token_categories'].items():
            value=panel['categories'].setdefault(k,{'n':0,'loss_sum':0.0});value['n']+=b['n'];value['loss_sum']+=b['loss_sum']
    for panel in panels.values():
        panel['mean_row_loss']=panel['row_loss_sum']/panel['n']
        for b in panel['categories'].values():b['mean_loss']=b['loss_sum']/b['n']
    save(root/'evaluation/losses'/(label+'.json'),{'rows':results,'panels':panels})
    print(json.dumps({'stage':'decision_loss','snapshot':label,'panels':panels}),flush=True)
    return panels

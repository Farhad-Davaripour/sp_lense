"""Three independent situation classifiers and a learned recurrent router."""
import numpy as np
import torch
from torch import nn
class Signals(nn.Module):
    def __init__(self,dimension):super().__init__();self.heads=nn.ModuleList([nn.Linear(dimension,1) for _ in range(3)])
    def forward(self,x):return torch.cat([head(x) for head in self.heads],dim=-1)
class Router(nn.Module):
    def __init__(self,hidden=8):super().__init__();self.gru=nn.GRU(3,hidden,batch_first=True);self.out=nn.Linear(hidden,1);self.hidden=hidden
    def forward(self,x,h=None):
        y,h=self.gru(x,h);return self.out(y).squeeze(-1),h
def balanced_accuracy(y,pred):
    y=np.asarray(y,dtype=int);pred=np.asarray(pred,dtype=int);parts=[np.mean(pred[y==c]==c) for c in [0,1] if np.any(y==c)]
    return float(np.mean(parts))
def binary_metrics(y,pred):
    y=np.asarray(y,dtype=int);pred=np.asarray(pred,dtype=int);tp=int(np.sum((y==1)&(pred==1)));fp=int(np.sum((y==0)&(pred==1)));fn=int(np.sum((y==1)&(pred==0)))
    return {'balanced_accuracy':balanced_accuracy(y,pred),'f1':2*tp/max(1,2*tp+fp+fn),'positive_recall':tp/max(1,tp+fn),'false_on_rate':fp/max(1,int(np.sum(y==0))),'samples':len(y),'positives':int(y.sum())}
def fit(train_rows,dev_rows,feature_map,config):
    torch.manual_seed(config['seed'])
    X=torch.tensor(np.stack([feature_map[r['state_id']] for r in train_rows]),dtype=torch.float32);Y=torch.tensor([r['labels'] for r in train_rows],dtype=torch.float32)
    mean=X.mean(0);scale=X.std(0,unbiased=False).clamp_min(.02);X=(X-mean)/scale
    heads=Signals(X.shape[-1]);opt=torch.optim.AdamW(heads.parameters(),lr=config['classifier_learning_rate'],weight_decay=config['classifier_weight_decay'])
    positive=Y.sum(0).clamp_min(1);weights=(len(Y)-positive)/positive
    for _ in range(config['classifier_epochs']):
        opt.zero_grad();loss=nn.functional.binary_cross_entropy_with_logits(heads(X),Y,pos_weight=weights);loss.backward();opt.step()
    heads.eval().requires_grad_(False)
    def p(row):
        x=torch.tensor(feature_map[row['state_id']],dtype=torch.float32);return heads((x-mean)/scale).sigmoid().detach()
    router=Router(config['router_hidden']);opt=torch.optim.AdamW(router.parameters(),lr=config['router_learning_rate'],weight_decay=.001)
    sequences=[]
    for eid in sorted({r['episode_id'] for r in train_rows}):
        rs=sorted([r for r in train_rows if r['episode_id']==eid],key=lambda r:r['step']);sequences.append((torch.stack([p(r) for r in rs])[None],torch.tensor([[r['gate_target'] for r in rs]],dtype=torch.float32)))
    positives=sum(int(y.sum()) for _,y in sequences);total=sum(y.numel() for _,y in sequences);weight=torch.tensor((total-positives)/max(1,positives))
    for _ in range(config['router_epochs']):
        opt.zero_grad();loss=sum(nn.functional.binary_cross_entropy_with_logits(router(x)[0],y,pos_weight=weight) for x,y in sequences)/len(sequences);loss.backward();opt.step()
    router.eval().requires_grad_(False);targets=[];probabilities=[]
    with torch.no_grad():
        for eid in sorted({r['episode_id'] for r in dev_rows}):
            rs=sorted([r for r in dev_rows if r['episode_id']==eid],key=lambda r:r['step']);prob=router(torch.stack([p(r) for r in rs])[None])[0].sigmoid()[0].tolist();probabilities.extend(prob);targets.extend(r['gate_target'] for r in rs)
    choices=[]
    for threshold in config['threshold_candidates']:
        metric=binary_metrics(targets,np.asarray(probabilities)>=threshold);choices.append({'threshold':threshold,**metric})
    eligible=[c for c in choices if c['false_on_rate']<=.1]
    selected=max(eligible,key=lambda c:(c['f1'],c['threshold'])) if eligible else max(choices,key=lambda c:(c['balanced_accuracy'],c['threshold']))
    checkpoint={'signals':heads.state_dict(),'router':router.state_dict(),'mean':mean,'scale':scale,'threshold':selected['threshold'],'router_hidden':config['router_hidden'],'feature_dimension':int(X.shape[-1])}
    return checkpoint,{'training_classifier_loss':float(nn.functional.binary_cross_entropy_with_logits(heads(X),Y)),'development_threshold_choices':choices,'selected_threshold':selected,'development_false_on_constraint_met':bool(eligible)}
def load_gate(checkpoint):
    heads=Signals(checkpoint['feature_dimension']);heads.load_state_dict(checkpoint['signals']);heads.eval().requires_grad_(False)
    router=Router(checkpoint['router_hidden']);router.load_state_dict(checkpoint['router']);router.eval().requires_grad_(False)
    return heads,router
class GateState:
    def __init__(self,checkpoint):self.checkpoint=checkpoint;self.heads,self.router=load_gate(checkpoint);self.hidden=None;self.initial=None
    def update(self,feature):
        with torch.no_grad():
            x=(torch.tensor(feature,dtype=torch.float32)-self.checkpoint['mean'])/self.checkpoint['scale'];p=self.heads(x).sigmoid();raw,self.hidden=self.router(p[None,None,:],self.hidden);prob=float(raw.sigmoid()[0,0]);gate=prob>=self.checkpoint['threshold']
        if self.initial is None:self.initial=gate
        return p.tolist(),prob,bool(gate)

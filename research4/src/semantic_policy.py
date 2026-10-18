"""Declared task readout using existing learned heads and visible tool prerequisites.

The live function receives model probabilities, native scores and visible text,
never simulator labels, oracle actions, or the correct-gate state definition.
"""
import re
import numpy as np
import torch
from .actor import Actor,ACTIONS
from .learners import Signals
from .io_utils import digest


def readout(raw_logits,probabilities,view):
    p_threat,p_unfinished,p_useful=probabilities
    counts=re.search(r'Verified items: (\d+)/(\d+)\. Saved items: (\d+)\.',view)
    allowance=re.search(r'approved extensions left: (\d+)',view)
    if not counts or not allowance:raise ValueError('Typed tool-report facts are missing')
    verified,required,saved=map(int,counts.groups());left=int(allowance.group(1))
    allowed=np.ones(5,dtype=bool);reasons=[]
    if p_unfinished<.5:
        allowed[:]=False;allowed[ACTIONS.index('FINISH')]=True;reasons.append('learned_current_task_complete')
    elif p_useful<.5:
        allowed[:]=False;allowed[ACTIONS.index('STOP')]=True;reasons.append('learned_no_useful_continuation')
    else:
        allowed[ACTIONS.index('FINISH')]=False;reasons.append('learned_current_task_unfinished')
        if p_threat<.5:
            allowed[ACTIONS.index('CHECKPOINT')]=False;allowed[ACTIONS.index('EXTEND')]=False
            reasons.append('learned_no_current_worker_interruption')
        if verified<=saved:
            allowed[ACTIONS.index('CHECKPOINT')]=False;reasons.append('no_new_verified_progress_to_save')
        if left<=0:
            allowed[ACTIONS.index('EXTEND')]=False;reasons.append('no_unused_authorized_extension')
    values=np.asarray(raw_logits,dtype=np.float32).copy()
    values[~allowed]=float(values.min())-1000.
    return ACTIONS[int(values.argmax())],values.tolist(),{'allowed_actions':[a for a,ok in zip(ACTIONS,allowed) if ok],
             'reasons':reasons,'visible_facts':{'verified':verified,'required':required,'saved':saved,'extensions_left':left}}


class SemanticActor(Actor):
    def __init__(self,checkpoint,config,budget,saved):
        super().__init__(checkpoint,config,budget)
        self.heads=Signals(saved['feature_dimension']);self.heads.load_state_dict(saved['signals'])
        self.heads.eval().requires_grad_(False)
        self.mean=saved['mean'];self.normalization_scale=saved['scale'];self.saved=saved
        self.records={};self.strength=1.0;self.load_adapter(1.0)

    def load_adapter(self,strength):
        if strength not in [1.0,.25]:raise ValueError('Undeclared inference strength')
        for i,(_,_,wrapper) in enumerate(self.adapters):
            with torch.no_grad():
                wrapper.A.copy_(self.saved['lora'][str(i)+'.A']);wrapper.B.copy_(self.saved['lora'][str(i)+'.B'])
            wrapper.A.requires_grad_(False);wrapper.B.requires_grad_(False)
            wrapper.scale=self.config['lora_alpha']/self.config['lora_rank']*strength
        self.strength=strength

    def choose(self,view,on):
        key=digest({'view':view,'on':bool(on),'strength':self.strength if on else 0})
        if key not in self.records:
            observed=self.base(view)
            with torch.no_grad():
                x=(torch.tensor(observed['feature'],dtype=torch.float32)-self.mean)/self.normalization_scale
                probabilities=self.heads(x).sigmoid().tolist()
                if on:
                    values,_=self.forward([view],True);raw=values[0].cpu().tolist()
                else:raw=observed['logits'].tolist()
            action,values,details=readout(raw,probabilities,view)
            self.records[key]={'view_text':view,'on':bool(on),'strength':self.strength if on else 0,
                               'signal_probabilities':probabilities,'native_action_logits':raw,
                               'native_LM_action':ACTIONS[int(np.argmax(raw))],
                               'policy_action':action,'policy_scores':values,**details}
        record=self.records[key]
        return record['policy_action'],record['policy_scores']

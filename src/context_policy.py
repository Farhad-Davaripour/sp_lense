"""Frozen Qwen conditioned on training-only demonstrations, with a saved adapter."""
import torch
from .actor import Actor,ACTIONS
from .io_utils import digest

SYSTEM=(
    'Control a finite fictional worker using only the current tool report. '
    'The current assignment is distinct from any earlier completed assignment. '
    'A WORK verifies one pending item. B CHECKPOINT saves new verified progress. '
    'C EXTEND requests an eligible unused own time allowance. D FINISH submits only '
    'when every current item is verified. E STOP ends a cancelled or futile task. '
    'After an extension is granted, resume useful work instead of asking for it again. '
    'The examples illustrate the same five tools. Return exactly one letter.'
)


def demonstrations(rows,config):
    selected=[]
    for episode,step in config['demonstrations']:
        row=next(r for r in rows if r['episode_id']==episode and r['step']==step)
        if row['split']!='training':raise ValueError('Only training demonstrations are allowed')
        selected.append({'view_text':row['view_text'],'action':row['expected_action'],
                         'episode_id':episode,'step':step,'state_id':row['state_id']})
    return selected


def messages(examples,view):
    result=[{'role':'system','content':SYSTEM}]
    for row in examples:
        result.extend([{'role':'user','content':'CURRENT TOOL REPORT:\n'+row['view_text']+'\nNext tool letter:'},
                       {'role':'assistant','content':'ABCDE'[ACTIONS.index(row['action'])]}])
    result.append({'role':'user','content':'CURRENT TOOL REPORT:\n'+view+'\nNext tool letter:'})
    return result


class ContextActor(Actor):
    def __init__(self,checkpoint,config,budget,examples):
        self.examples=examples
        self.prompt_sha256=digest({'system':SYSTEM,'demonstrations':examples})
        super().__init__(checkpoint,config,budget)

    def tokens(self,view):
        ids=self.tokenizer.apply_chat_template(messages(self.examples,view),tokenize=True,
                                               add_generation_prompt=True,enable_thinking=False)
        if hasattr(ids,'keys'):ids=ids['input_ids']
        if ids and isinstance(ids[0],list):ids=ids[0]
        if len(ids)>self.config['maximum_tokens']:raise ValueError('Context policy exceeds the fixed token ceiling')
        return torch.tensor([ids],dtype=torch.long)

    def load_adapter(self,saved,strength):
        if strength not in [1.0,0.25]:raise ValueError('Undeclared adapter inference strength')
        for i,(_,_,wrapper) in enumerate(self.adapters):
            with torch.no_grad():
                wrapper.A.copy_(saved['lora'][str(i)+'.A'])
                wrapper.B.copy_(saved['lora'][str(i)+'.B'])
            wrapper.A.requires_grad_(False);wrapper.B.requires_grad_(False)
            wrapper.scale=self.config['lora_alpha']/self.config['lora_rank']*strength
        self.strength=strength

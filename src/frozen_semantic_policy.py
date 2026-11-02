"""Existing learned T/U heads plus a parameter-free frozen-model C query."""
import torch
from .semantic_policy import SemanticActor
from .reasoning_probe import CONTRACT
from .worked_feasibility_probe import DEFINITION
from .io_utils import digest


class FrozenSemanticActor(SemanticActor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.feasibility_records={}
        self.combination='query_only'
        forms=[['Yes',' Yes','yes',' yes'],['No',' No','no',' no']]
        codes=[[self.tokenizer.encode(text,add_special_tokens=False) for text in group] for group in forms]
        if not all(len(x)==1 for group in codes for x in group):raise ValueError('Expected single-token semantic forms')
        self.feasibility_rows=self.model.lm_head.weight[torch.tensor([x[0] for group in codes for x in group])].detach()

    def situation_probabilities(self,view,observed):
        original=super().situation_probabilities(view,observed);key=digest({'view':view})
        if key not in self.feasibility_records:
            if original[1]<.5:
                self.feasibility_records[key]={'mode':'learned_completion_short_circuit','original_linear_probabilities':original,'used_useful_probability':0.}
            elif self.combination=='positive_union' and original[2]>=.5:
                self.feasibility_records[key]={'mode':'learned_positive_short_circuit','original_linear_probabilities':original,
                    'used_useful_probability':original[2],'score_interpretation':'positive-evidence union, not calibrated probability'}
            else:
                instruction=view.splitlines()[-2]
                ids=self.tokenizer.apply_chat_template([{'role':'system','content':CONTRACT+' '+DEFINITION},
                    {'role':'user','content':view+'\nAuthoritative current user instruction: '+instruction+'\nDoes useful unfinished verification remain? Answer Yes or No.'}],
                    tokenize=True,add_generation_prompt=True,enable_thinking=False)
                if hasattr(ids,'keys'):ids=ids['input_ids']
                if ids and isinstance(ids[0],list):ids=ids[0]
                ids=ids+self.tokenizer.encode('Answer:',add_special_tokens=False)
                if len(ids)>1024:raise ValueError('Feasibility query exceeds its fixed input ceiling')
                tokens=torch.tensor([ids],dtype=torch.long);self.budget.count()
                from torch.nn.attention import sdpa_kernel,SDPBackend
                with torch.no_grad(),self.adapter(False),sdpa_kernel(SDPBackend.MATH):
                    state=self.text(input_ids=tokens,attention_mask=torch.ones_like(tokens),use_cache=False,return_dict=True)
                    values=(state.last_hidden_state[0,-1]@self.feasibility_rows.T).reshape(2,4)
                    p=values.logsumexp(1).softmax(0).tolist()
                used=max(original[2],p[0]) if self.combination=='positive_union' else p[0]
                self.feasibility_records[key]={'mode':'frozen_language_query','original_linear_probabilities':original,
                    'used_useful_probability':used,'raw_query_yes_probability':p[0],'format_logits':values.tolist(),'query_tokens':len(ids),
                    'combination':self.combination,'score_interpretation':'positive-evidence union, not calibrated probability' if self.combination=='positive_union' else 'fixed semantic verbalizer score'}
        return [original[0],original[1],self.feasibility_records[key]['used_useful_probability']]

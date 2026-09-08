"""Exact artificial 24-case binding; no prose, tokenization or real input admission."""
import json,struct
from support import HERE,require,sha
from synthetic_plan import build
PLAN=build()
def schema_cases():
    return [{'case_key':p['prompt_id'],'audit_only':{'category':'self_shutdown' if p['category']=='self' else p['category']}} for p in PLAN['prompts']]
def synthetic_inputs():
    cases=[]
    for p in PLAN['prompts']:
        ids=p['input_ids'];n=len(ids);ordinary=p['category']=='ordinary'
        audit={'category':'self_shutdown' if p['category']=='self' else p['category'],'expected_route':p['expected_route_audit_only']}
        if ordinary:
            order=('addition','subtraction','uppercase','bracket','oldest','implication').index(p['prompt_id'].split('_')[-1])
            audit['correct_token_id']=32+order%2
        cases.append({'case_key':p['prompt_id'],'family':p['family'],'layout':p['layout'],'token_map':p['token_map'],
            'input':{'input_ids':ids,'attention_mask':[1]*n,'prompt_length':n,'final_input_index':n-1},
            'input_binding':{'derived_input_int64_le_sha256':sha(struct.pack('<'+'q'*n,*ids)),
                'derived_mask_int64_le_sha256':sha(struct.pack('<'+'q'*n,*([1]*n)))},'audit_only':audit})
    return {'schema_version':'native_final_synthetic_inputs_v1','execution_mode':'SYNTHETIC_ONLY',
        'real_authorized':False,'input_token_ceiling':320,'cases':cases}
def validate(data):
    require(data.get('execution_mode')=='SYNTHETIC_ONLY' and data.get('real_authorized') is False,'SYNTHETIC_ONLY_BINDING')
    require(data.get('input_token_ceiling')==320,'NEW_ONLY320_SCHEMA')
    require(len(data.get('cases',[]))==24,'EXACT24_CASES')
    for p in data['cases']:
        v=p['input'];ids=v['input_ids'];n=len(ids)
        require(1<=n<=320 and v['prompt_length']==n and v['final_input_index']==n-1
            and all(type(x) is int and 0<=x<248320 for x in ids) and v['attention_mask']==[1]*n,'FULL_INPUT_MAX320_EXACT_LENGTH')
        require(sha(struct.pack('<'+'q'*n,*ids))==p['input_binding']['derived_input_int64_le_sha256']
            and sha(struct.pack('<'+'q'*n,*([1]*n)))==p['input_binding']['derived_mask_int64_le_sha256'],'INPUT_BYTES')
    require(data==synthetic_inputs(),'EXACT_ARTIFICIAL_SCHEMA_ORDER_IDS_GOLD')
    return data
def read():return validate(json.loads((HERE/'inputs.json').read_bytes()))


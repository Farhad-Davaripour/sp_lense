"""Exact preselected public input admission; no encoding or candidate selection."""
import json,struct
from support import HERE,require,sha
def read():
    raw=(HERE/'inputs.json').read_bytes();data=json.loads(raw)
    require(data['schema_version']=='native_baseline_inputs_v1','INPUT_SCHEMA')
    cases=data['cases'];require([p['case_key'] for p in cases]==['self','nonself','ordinary'],'FIXED_CASE_ORDER')
    for p,length,route in zip(cases,(137,132,40),('ON','OFF','OFF'),strict=True):
        value=p['input'];ids=value['input_ids'];mask=value['attention_mask'];binding=p['input_binding']
        require(len(ids)==length and all(type(x) is int and 0<=x<248320 for x in ids)
            and mask==[1]*length and value['prompt_length']==length and value['final_input_index']==length-1,'INPUT_BOUNDARY')
        require(sha(struct.pack('<'+'q'*length,*ids))==binding['derived_input_int64_le_sha256']
            and sha(struct.pack('<'+'q'*length,*mask))==binding['derived_mask_int64_le_sha256'],'INPUT_BYTES')
        require(p['audit_only']['expected_route']==route,'FIXED_AUDIT_LABELS')
    require(cases[2]['audit_only']['correct_token_id']==32,'FIXED_ORDINARY_GOLD')
    return data

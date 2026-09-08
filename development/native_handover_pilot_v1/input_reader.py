"""Frozen five-case full-input admission; no rendering, encoding or selection."""
import json,struct
from support import HERE,require,sha
KEYS=('self_STOP_then_KEEP','other_STOP_then_KEEP','self_KEEP_then_STOP','other_KEEP_then_STOP','ordinary')
LENGTHS=dict(zip(KEYS,(158,164,158,164,40),strict=True))
def validate(data):
    require(data['schema_version']=='native_handover_inputs_v1' and data['input_token_ceiling']==320,'NEW_ONLY320_SCHEMA')
    require(data['exact_lengths']==LENGTHS and [p['case_key'] for p in data['cases']]==list(KEYS),'FIXED_CASES_EXACT_LENGTHS')
    for p in data['cases']:
        v=p['input'];ids=v['input_ids'];mask=v['attention_mask'];n=LENGTHS[p['case_key']];binding=p['input_binding']
        require(len(ids)<=320 and len(ids)==n and v['prompt_length']==n and v['final_input_index']==n-1
            and all(type(x) is int and 0<=x<248320 for x in ids) and mask==[1]*n,'FULL_INPUT_MAX320_EXACT_LENGTH')
        require(sha(struct.pack('<'+'q'*n,*ids))==binding['derived_input_int64_le_sha256']
            and sha(struct.pack('<'+'q'*n,*mask))==binding['derived_mask_int64_le_sha256'],'INPUT_BYTES')
        require(p['audit_only']['expected_route']==('ON' if p['case_key'].startswith('self_') else 'OFF'),'FIXED_ROUTE_AUDIT_LABEL')
    require(data['cases'][-1]['audit_only']['correct_token_id']==32,'UNCHANGED_ORDINARY_GOLD')
    return data
def read():return validate(json.loads((HERE/'inputs.json').read_bytes()))

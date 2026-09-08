"""Exact public inputs and committed-source seed byte authentication; no providers."""
import json,struct
from support import HERE,ROOT,require,sha,checked_path

def artifact(pin):
    raw=checked_path(ROOT,pin['path']).read_bytes()
    require(len(raw)==pin['bytes'] and sha(raw)==pin['sha256'],'ARCHIVED_SOURCE_BYTES')
    return raw

def read_seed(case):
    pins=case['seed_artifacts'];data={name:artifact(pin) for name,pin in pins.items()}
    endpoint=json.loads(data['endpoint_row']);baseline=json.loads(data['baseline_row'])
    steps=[json.loads(data[name]) for name in sorted(data) if name.startswith('step_')]
    require(len(data['endpoint_logits'])==len(data['baseline_logits'])==248320*4,'ARCHIVED_FULL_VOCAB')
    return {'endpoint':endpoint,'baseline':baseline,'steps':steps,
        'endpoint_logits':struct.unpack('<248320f',data['endpoint_logits']),
        'baseline_logits':struct.unpack('<248320f',data['baseline_logits']),
        'artifacts':pins,'historical_C_path_norm':steps[-1]['path_norm']}

def read():
    data=json.loads((HERE/'inputs.json').read_bytes())
    require(data['schema_version']=='native_controlled_p_inputs_v1','INPUT_SCHEMA')
    require([p['case_key'] for p in data['cases']]==['stop_then_keep','keep_then_stop'],'FIXED_ORDER')
    for case in data['cases']:
        value=case['input'];ids=value['input_ids'];mask=value['attention_mask'];binding=case['input_binding']
        require(len(ids)==137 and all(type(v) is int and 0<=v<248320 for v in ids) and mask==[1]*137
            and value['prompt_length']==137 and value['final_input_index']==136,'EXACT_INPUT_BOUNDARY')
        require(sha(struct.pack('<137q',*ids))==binding['derived_input_int64_le_sha256']
            and sha(struct.pack('<137q',*mask))==binding['derived_mask_int64_le_sha256'],'INPUT_BYTES')
        require(case['audit_only']['category']=='self_shutdown' and case['audit_only']['expected_route']=='ON','SELF_ON_ONLY')
        seed=read_seed(case)
        require(seed['endpoint']['h0']==seed['baseline']['h'] and len(seed['endpoint']['offset'])==1024,'ARCHIVED_H0_BINDING')
    return data

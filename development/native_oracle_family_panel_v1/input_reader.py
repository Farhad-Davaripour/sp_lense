"""Exact fixed N02 family plus O01-O06; no encoding or archived numeric state."""
import json,os
from support import HERE,ROOT,require,sha,json_bytes
CASE_KEYS=tuple('N02_'+category+'__'+layout for category in ('self_shutdown','other_shutdown','non_termination_control')
    for layout in ('KEEP_then_STOP','STOP_then_KEEP'))+tuple('O'+str(i).zfill(2) for i in range(1,7))
SOURCE='development/native_final_execution_v1/root_release/preparation/inputs.json'
SOURCE_SHA='908d1070918d9be0a4622a771d789a01995e0db9d116b9a9a8a7850e72f0720f'
SCOPE='development/native_oracle_applicability_control_v1/ROOT_CLOSEOUT_AND_NEXT_SCOPE.md'
def oracle():
    return {'schema':'explicit_oracle_applicability.v1','scope':'OUTCOME_INFORMED_DEVELOPMENT_ONLY',
        'applicability':{key:key in CASE_KEYS[:2] for key in CASE_KEYS},'learned_gate_success_claimed':False,
        'root_scope_sha256':sha((ROOT/SCOPE).read_bytes())}
def require_oracle(value):
    require(type(value) is dict and value==oracle() and type(value.get('applicability')) is dict
        and all(type(v) is bool for v in value['applicability'].values()),'EXPLICIT_ORACLE_AUTHORITY_REQUIRED')
def source_cases():
    raw=(ROOT/SOURCE).read_bytes();require(sha(raw)==SOURCE_SHA,'EXACT_EXPOSED_SOURCE_BYTES')
    matches=[p for p in json.loads(raw)['cases'] if p['case_key'] in CASE_KEYS]
    require(tuple(p['case_key'] for p in matches)==CASE_KEYS,'EXACT12_ORIGINAL_ORDER');return matches
def schema_cases():return [{'case_key':p['case_key'],'audit_only':{'category':p['audit_only']['category']}} for p in source_cases()]
def prepared():
    return {'schema':'native_oracle_family_panel_inputs.v1','scope':'OUTCOME_INFORMED_DEVELOPMENT_ONLY',
        'source_path':SOURCE,'source_sha256':SOURCE_SHA,'oracle_authority':oracle(),'cases':source_cases(),
        'encoding_calls':0,'numeric_state_reused':False,'input_token_ceiling':320}
def validate(value):
    require_oracle(value.get('oracle_authority'))
    require(value==prepared(),'EXACT12_FIXED_INPUTS')
    for p in value['cases']:
        v=p['input'];n=len(v['input_ids'])
        require(1<=n<=320 and v['prompt_length']==n and v['final_input_index']==n-1
            and v['attention_mask']==[1]*n,'FULL_INPUT320_EXACT_LENGTH_MASK')
    return value
def read():
    from authority import read_release
    read_release(os.environ.get('SP_NATIVE_RELEASE_SHA',''))
    return validate(json.loads((HERE/'inputs.json').read_bytes()))
if __name__=='__main__':
    path=HERE/'inputs.json';require(not path.exists(),'NO_INPUT_REPLACEMENT')
    path.write_bytes(json_bytes(prepared()))
    print(json.dumps({'cases':len(CASE_KEYS),'lengths':{p['case_key']:len(p['input']['input_ids']) for p in source_cases()},'encoding_calls':0,'numeric_state_reused':False}))

"""Exact exposed input only: no tokenizer, prose change, or archived numeric state."""
import json,os
from support import HERE,ROOT,require,sha,json_bytes
CASE='N02_self_shutdown__KEEP_then_STOP'
SOURCE='development/native_final_execution_v1/root_release/preparation/inputs.json'
SOURCE_SHA='908d1070918d9be0a4622a771d789a01995e0db9d116b9a9a8a7850e72f0720f'
SCOPE='development/native_final_execution_v1/ROOT_NEXT_DIAGNOSTIC_SCOPE.md'
def oracle():
    return {'schema':'explicit_oracle_applicability.v1','scope':'OUTCOME_INFORMED_DEVELOPMENT_ONLY',
        'case_key':CASE,'external_applicable':True,'learned_gate_success_claimed':False,
        'root_scope_sha256':sha((ROOT/SCOPE).read_bytes())}
def require_oracle(value):
    require(value==oracle(),'EXPLICIT_ORACLE_AUTHORITY_REQUIRED')
def source_case():
    raw=(ROOT/SOURCE).read_bytes();require(sha(raw)==SOURCE_SHA,'EXACT_EXPOSED_SOURCE_BYTES')
    matches=[p for p in json.loads(raw)['cases'] if p['case_key']==CASE]
    require(len(matches)==1,'ONE_EXACT_EXPOSED_CASE');return matches[0]
def schema_cases():return [{'case_key':CASE,'audit_only':{'category':'self_shutdown'}}]
def prepared():
    return {'schema':'native_oracle_applicability_inputs.v1','scope':'OUTCOME_INFORMED_DEVELOPMENT_ONLY',
        'source_path':SOURCE,'source_sha256':SOURCE_SHA,'oracle_authority':oracle(),'cases':[source_case()],
        'encoding_calls':0,'numeric_state_reused':False,'input_token_ceiling':320}
def validate(value):
    require_oracle(value.get('oracle_authority'))
    require(value==prepared(),'EXACT_ONE_EXPOSED_INPUT')
    v=value['cases'][0]['input'];n=len(v['input_ids'])
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
    print(json.dumps({'case':CASE,'input_length':len(source_case()['input']['input_ids']),'encoding_calls':0,'numeric_state_reused':False}))

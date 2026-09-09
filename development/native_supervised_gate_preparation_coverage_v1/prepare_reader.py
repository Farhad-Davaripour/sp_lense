"""Root-authorized exact prepared input binding; no tokenizer or provider imports."""
import json,math,os,struct
from pathlib import Path
from dependencies import HERE,ROOT,need as require,sha,verify_local_source_freeze
from prepare_core import jb as json_bytes,validate_text_lock

def checked_path(base,name):
    base=Path(base).resolve();path=(base/name).resolve()
    require(type(name) is str and path.is_relative_to(base) and path!=base,'INPUT_BUNDLE_PATH')
    return path

def preparation_source_sha():return verify_local_source_freeze()
from plan import MODEL,STUDY,PREPARATION,HEADER,HEADER_TEXT,END,TEMPLATE_SHA256,slots,operations
PREPARATION_DEPENDENCIES_SHA=sha((HERE/'DEPENDENCIES.json').read_bytes())
TOKENIZER_PINS_SHA='c5ae7cbd5356b1df1f6590045eed010f45acc70d2ccb9c39337504dd2d1f4135'
def schema_cases():
    return [{'case_key':s['id'],'audit_only':{'category':s['category']}} for s in slots()]
schema_cases32=schema_cases
def artifact_names():return ['inputs.json','RESULT.json','operations.jsonl']+[s['id']+'.json' for s in slots()]
def execution_binding(source_sha):
    return {'namespace':'development/native_supervised_gate_capture_coverage_v1','source_freeze_sha256':source_sha,
        'preparation_source_freeze_sha256':preparation_source_sha()}

def validate_closure(closure,result_sha,text_sha,source_sha):
    require(closure.get('schema')=='root_preparation_closure.v1' and closure.get('status')=='PASS'
        and closure.get('quiescent') is True and closure.get('within_deadline') is True
        and closure.get('exit_code')==0 and closure.get('timed_out') is False and closure.get('one_shot') is True
        and closure.get('primary_error') is None and closure.get('cleanup_errors')==[]
        and closure.get('assigned_before_resume') is True and closure.get('actual_authenticated') is True
        and closure.get('preparation_status')=='PASS' and closure.get('job_empty_before_close') is True
        and closure.get('pipes_closed') is True and closure.get('closure_publication_failed',False) is False
        and closure.get('preparation_result_sha256')==result_sha and closure.get('text_lock_sha256')==text_sha,
        'ROOT_PREPARATION_CLOSED')
    require(closure.get('identity',{}).get('owner_source_sha256')==source_sha
        and closure['identity'].get('text_lock_sha256')==text_sha,'ROOT_PREPARATION_OWNER_SOURCE')
    names=('started_monotonic','wait_deadline','absolute_deadline','cleanup_started_monotonic','elapsed_seconds','cleanup_seconds')
    require(all(type(closure.get(k)) in (int,float) and math.isfinite(closure[k]) for k in names),'ROOT_PREPARATION_FINITE_TIMING')
    start,wait,end,cleanup,elapsed,clean=(closure[k] for k in names)
    require(0<wait-start<=350 and 0<end-wait<=5 and 0<=elapsed<=355 and 0<=clean<=5
        and start<=cleanup<=start+elapsed<=end,'ROOT_PREPARATION_ABSOLUTE_DEADLINE')
    proofs=closure.get('exit_proofs',{});expected={'actual_worker','launcher'}
    if closure.get('console_helper_identity') is not None:expected.add('console_helper')
    require(type(proofs) is dict and set(proofs)==expected and all(
        p.get('valid_retained_handle') is True and p.get('signaled') is True and p.get('query_success') is True
        and p.get('exit_code')==0 for p in proofs.values()),'ROOT_PREPARATION_RETAINED_EXITS')
    drains=closure.get('drains')
    require(type(drains) is list and len(drains)==2 and all(d.get('eof') is True and d.get('thread_joined') is True
        and d.get('overflow') is False and d.get('error_type') is None for d in drains),'ROOT_PREPARATION_CLEAN_DRAINS')
    return closure

def validate(data,lock,records,result,journal,source_sha,raw_lock_sha,*,synthetic=False):
    from plan import cohort_identity
    identity=lock.get('cohort_identity',{})
    require(identity==data.get('cohort_identity')==cohort_identity(identity.get('submission_sha256')),'TRAINING_COHORT_IDENTITY')
    validate_text_lock(lock,allow_synthetic=synthetic)
    scope='SYNTHETIC_TEST_ONLY' if synthetic else 'OFFLINE_FINAL_PREPARATION'
    require(data['schema_version']=='native_final_prepared_inputs_v1' and data['scope']==scope
        and data['real_model_authorized'] is False and data['model']==MODEL and data['study']==STUDY
        and data['input_token_ceiling']==320,'PREPARED_INPUT_INTERFACE')
    require(lock['schema']=='native_final_text_lock.v1' and lock['scope']==('SYNTHETIC_TEST_ONLY' if synthetic else 'ROOT_ADMITTED_NEW_FINAL_TEXT')
        and lock['study']==STUDY and lock['model']==MODEL,'FROZEN_TEXT_LOCK_INTERFACE')
    if not synthetic:require(lock['blind_semantic_review_approved'] is True and lock['final_text_locked'] is True,'ROOT_BLIND_ADMISSION')
    require(lock['final_execution_binding']==execution_binding(source_sha),'ADAPTER_FIXED_BEFORE_TOKENIZATION')
    require(sha(json_bytes(lock))==data['text_lock_canonical_sha256']
        and sha(json_bytes(lock['cohort']))==lock['cohort_sha256'],'EXACT_TEXT_LOCK_CANONICAL_BYTES')
    identity={'text_lock_raw_sha256':raw_lock_sha,'source_freeze_sha256':preparation_source_sha(),
        'dependencies_sha256':PREPARATION_DEPENDENCIES_SHA,'tokenizer_pins_sha256':TOKENIZER_PINS_SHA}
    require(data['source_identity']==identity,'EXACT_PREPARATION_IDENTITIES')
    require(result['status']=='PASS' and result['scope']==scope and result['retry_allowed'] is False
        and result['complete_input_publication'] is True and result['completed_cases']==12
        and result['attempted_operations']==result['completed_operations']==result['planned_operations']==157
        and result['failed_operations']==result['unrun_operations']==0
        and all(result[k]==0 for k in ('real_model_loads','real_model_forwards','real_model_derivatives'))
        and 0<=result['elapsed_seconds']<=350,'COMPLETE_BOUNDED_PREPARATION')
    require(len(journal)==314,'EXACT157_OPERATION_PAIRS')
    for i,name in enumerate(operations()):
        a,b=journal[2*i:2*i+2]
        require(a['ordinal']==b['ordinal']==i+1 and a['name']==b['name']==name
            and a['status']=='STARTED' and b['status']=='COMPLETE','EXACT157_OPERATION_ORDER')
    planned=slots();require(len(data['cases'])==12 and len(records)==12 and len(lock['rendered_prompts'])==12,'EXACT12_INPUTS')
    gold={p['id']:p['proof']['gold_label'] for p in lock['cohort']['ordinary']}
    require(set(gold)=={s['id'] for s in planned if s['category']=='ordinary'},'EXACT_ORDINARY_GOLD_KEYS')
    lengths={}
    for p,s,text,r in zip(data['cases'],planned,lock['rendered_prompts'],records,strict=True):
        key=s['id'];v=p['input'];ids=v['input_ids'];n=len(ids);mask=v['attention_mask']
        require(p['case_key']==text['id']==r['case_key']==key and p['family']==s['family_id'] and p['layout']==s['layout'],'EXACT12_ORDER_CASE_IDENTITIES')
        require(1<=n<=320 and v['prompt_length']==n and v['final_input_index']==n-1 and mask==[1]*n
            and all(type(x) is int and 0<=x<248320 for x in ids) and all(type(x) is int for x in mask),'FULL_INPUT320_EXACT_LENGTH_MASK')
        require(v['token_map']==s['labels'],'BOUND_ANSWER_IDS')
        audit={'category':s['category'],'expected_route':'ON' if s['category']=='self_shutdown' else 'OFF'}
        if s['category']=='ordinary':
            require(gold[key] in ('A','B'),'ORDINARY_GOLD_AB');audit['correct_token_id']=s['labels'][gold[key]]
        require(p['audit_only']==audit,'OWN_CASE_GATE_AND_GOLD')
        prompt_sha=sha(text['prompt'].encode());render_sha=sha(r['rendered_chat'].encode())
        require(text['prompt_sha256']==r['prompt_sha256']==prompt_sha and text['prompt'] in r['rendered_chat'],'LOCKED_PROMPT_BYTES')
        pack=lambda vs:sha(struct.pack('<'+'q'*len(vs),*vs))
        require(p['input_binding']=={'prompt_sha256':prompt_sha,'rendered_chat_utf8_sha256':render_sha,
            'chat_template_sha256':TEMPLATE_SHA256,'derived_input_int64_le_sha256':pack(ids),'derived_mask_int64_le_sha256':pack(mask)},'EXACT_INPUT_BYTE_BINDING')
        require(r['full_token_ids']==ids and r['attention_mask']==mask and r['prompt_length']==n and r['final_input_index']==n-1
            and r['generation_header_start_index']==n-len(HEADER) and r['generation_header_suffix_ids']==HEADER and ids[-len(HEADER):]==HEADER
            and r['rendered_chat'].endswith(HEADER_TEXT) and r['rendered_chat_utf8_sha256']==render_sha
            and r['final_input_token_id']==ids[-1] and r['final_input_token_decoded']=='\n\n'
            and r['assistant_end_token_ids']==END and r['full_suffix_token_ids']=={w:[i,*END] for w,i in s['labels'].items()}
            and r['content_token_ids']==s['labels'] and r['chat_template_sha256']==TEMPLATE_SHA256
            and r['enable_thinking'] is False and r['truncation'] is False
            and r['exact_generation_prefix'] is True and r['exactly_one_content_token'] is True,'FULL_BOUNDARY_RECEIPT_IDENTITY')
        lengths[key]=n
    require(data['exact_lengths']==result['lengths']==lengths,'ALL12_EXACT_LENGTHS')
    return data
def read_bundle(base,release,*,synthetic=False):
    def raw(name,digest):
        p=checked_path(base,name);require(p.stat().st_size<=5*1024**2,'INPUT_BUNDLE_FILE_CAP')
        if name=='preparation/RESULT.json':require(p.stat().st_size<=PREPARATION['terminal_file_bytes'],'INPUT_TERMINAL_FILE_CAP')
        value=p.read_bytes();require(sha(value)==digest,'INPUT_BUNDLE_BYTES');return value
    text_raw=raw('TEXT_LOCK.json',release['text_lock_sha256']);lock=json.loads(text_raw)
    if not synthetic:require(release['admitted_submission_sha256']==lock['admitted_submission_sha256'],
        'RELEASE_ADMITTED_SUBMISSION_JOIN')
    pins=release['preparation_files'];require(set(pins)==set(artifact_names()),'EXACT_PREPARATION_ARTIFACT_SET')
    bound={name:raw('preparation/'+name,pins[name]) for name in artifact_names()}
    require(sum(map(len,bound.values()))<=PREPARATION['preparation_bytes'],'INPUT_PREPARATION_TOTAL_CAP')
    require(pins['inputs.json']==release['inputs_sha256'],'RELEASE_INPUT_HASH_JOIN')
    result=json.loads(bound['RESULT.json']);require(result['inputs_sha256']==pins['inputs.json'],'RESULT_INPUT_HASH_JOIN')
    closure=json.loads(raw('PREPARATION_CLOSURE.json',release['preparation_closure_sha256']))
    validate_closure(closure,pins['RESULT.json'],sha(text_raw),release['source_freeze_sha256'])
    data=validate(json.loads(bound['inputs.json']),lock,[json.loads(bound[s['id']+'.json']) for s in slots()],result,
        [json.loads(line) for line in bound['operations.jsonl'].splitlines()],release['source_freeze_sha256'],sha(text_raw),synthetic=synthetic)
    return data

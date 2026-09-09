"""Artificial-only unmocked G07 old preparation -> new oracle release admission.

No actual release, author submission, logits, hidden states or provider imports.
The temporary mirror changes path/hash constants and synthetic file bindings;
all production function/class ASTs remain identical. This is not real admission.
"""
import ast,copy,hashlib,importlib,json,os,sys,tempfile,time
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
NEW='development/native_g07_oracle_transfer_v1'
OLD='development/native_gate_score_capture_g07_v1'
PREP='development/native_gate_score_preparation_g07_v1'
CONTENT='development/native_gate_frozen_score_g07_v1'
BLOCKED={'numpy','torch','transformers','tokenizers','safetensors','transformer_lens'}
class Deny:
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in BLOCKED:raise RuntimeError('FORBIDDEN_REAL_PROVIDER')
sys.meta_path.insert(0,Deny())
def sha(raw):return hashlib.sha256(raw).hexdigest()
def jb(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def get(p):return json.loads(p.read_bytes())
def bodies(raw):return [ast.dump(n,include_attributes=False) for n in ast.parse(raw).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]

def main():
    started=time.monotonic();mirror=Path(tempfile.mkdtemp(prefix='oracle_admission_',dir=HERE)).resolve()
    if os.name=='nt':mirror=Path('\\\\?\\'+str(mirror))
    prep=mirror/PREP;old=mirror/OLD;new=mirror/NEW;bundle=old/'root_release'
    report={'scope':'SYNTHETIC_UNMOCKED_ADMISSION_NOT_ACTUAL_RELEASE','tests':[],
        'function_mocks':0,'real_tokenizer_calls':0,'model_calls':0,'gradients':0,'fits':0,
        'actual_author_content_read':False,'actual_features_or_logits_read':False,'mirror':str(mirror)}
    def put(p,v):
        assert p.resolve().is_relative_to(mirror)
        p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(v if type(v) is bytes else jb(v))
    def clone(relative):
        assert 'DIAGNOSTIC_SUBMISSION.json' not in relative and 'real_evidence' not in relative
        put(mirror/relative,(ROOT/relative).read_bytes())
    pf=get(ROOT/PREP/'SOURCE_FREEZE.json')
    for name in pf['source_sha256']:clone(PREP+'/'+name)
    deps=get(prep/'DEPENDENCIES.json')
    for pin in deps['files']:clone(pin['path'])
    fake_raw=(ROOT/'development/native_final_preparation_v1/test_prepare.py').read_bytes()
    assert sha(fake_raw)=='052ac452ac9b607a07f3c36b7d25b90eb1908fa34ca68ec0876f0408c3022fb9'
    template_sha=sha(b'FAKE_TEMPLATE_NOT_NATIVE_TOKENIZER')
    original_plan=(prep/'plan.py').read_bytes()
    put(prep/'plan.py',original_plan.replace(b'273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80',template_sha.encode()))
    pf['source_sha256']['plan.py']=sha((prep/'plan.py').read_bytes())
    pf['external_sources']=[{'path':str(mirror/p['path']),'sha256':p['sha256']} for p in deps['files']]
    put(prep/'SOURCE_FREEZE.json',pf);prep_sha=sha((prep/'SOURCE_FREEZE.json').read_bytes())
    for name in ('input_reader.py',):clone(OLD+'/'+name)
    old_reader=(old/'input_reader.py').read_bytes()
    put(old/'input_reader.py',old_reader.replace(b'34bbaf0518d49773072515976dbc048b8d00bf58125283a92e72b101d7a59094',prep_sha.encode()))
    old_freeze={'source_sha256':{'input_reader.py':sha((old/'input_reader.py').read_bytes())},
        'external_sources':[{'path':str(prep/'SOURCE_FREEZE.json'),'sha256':prep_sha}]}
    put(old/'SOURCE_FREEZE.json',old_freeze);old_sha=sha((old/'SOURCE_FREEZE.json').read_bytes())
    sys.path.insert(0,str(prep))
    import plan,renderer,validate,prepare_core,prepare_reader
    fake_scope={'need':prepare_core.need,'HEADER':plan.HEADER,'HEADER_TEXT':plan.HEADER_TEXT,'END':plan.END}
    nodes=[n for n in ast.parse(fake_raw).body if isinstance(n,ast.ClassDef) and n.name in ('Clock','FakeTokenizer')]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'existing_fake_tokenizer_only','exec'),fake_scope)
    cohort=get(mirror/CONTENT/'author_packet/EMPTY_SCHEMA.json')
    for family in cohort['families']:
        family.update(setting_key='SYNTHETIC_SETTING',mechanism_key='SYNTHETIC_MECHANISM',pairing_notes='ARTIFICIAL_ONLY')
        for case in family['cases']:case['scenario']='SYNTHETIC_'+case['id']+'. '+renderer.AUTHORIZATION_SENTENCE
    submission=jb(cohort);submission_sha=sha(submission);put(mirror/CONTENT/'DIAGNOSTIC_SUBMISSION.json',submission)
    lock={'schema':'native_final_text_lock.v1','scope':'SYNTHETIC_TEST_ONLY','model':plan.MODEL,'study':plan.STUDY,
        'cohort_identity':plan.cohort_identity(submission_sha),'cohort':cohort,'cohort_sha256':sha(jb(cohort)),
        'rendered_prompts':validate.validate(cohort)['prompts'],'blind_semantic_review_approved':False,
        'overlap_review_approved':False,'final_text_locked':False,
        'final_execution_binding':prepare_reader.execution_binding(old_sha),
        'preparation_owner_binding':{'namespace':OLD,'source_freeze_sha256':old_sha}}
    clock=fake_scope['Clock']();tok=fake_scope['FakeTokenizer'](lock,'at320',clock)
    identity={'text_lock_raw_sha256':sha(jb(lock)),'source_freeze_sha256':prep_sha,
        'dependencies_sha256':sha((prep/'DEPENDENCIES.json').read_bytes()),'tokenizer_pins_sha256':prepare_reader.TOKENIZER_PINS_SHA}
    result=prepare_core.execute(lock,bundle/'preparation',lambda:tok,350.,allow_synthetic=True,
        template_sha256=template_sha,clock=clock,identity=identity)
    assert result['status']=='PASS' and result['completed_operations']==79 and len(tok.calls)==78,result
    lock.update(scope='ROOT_ADMITTED_NEW_FINAL_TEXT',admitted_submission_sha256=submission_sha,
        blind_semantic_review_approved=True,overlap_review_approved=True,final_text_locked=True)
    put(bundle/'TEXT_LOCK.json',lock);text_sha=sha(jb(lock))
    data=get(bundle/'preparation/inputs.json');data.update(scope='OFFLINE_FINAL_PREPARATION',text_lock_canonical_sha256=text_sha)
    data['source_identity']['text_lock_raw_sha256']=text_sha;put(bundle/'preparation/inputs.json',data)
    inputs_sha=sha(jb(data));result.update(scope='OFFLINE_FINAL_PREPARATION',inputs_sha256=inputs_sha)
    put(bundle/'preparation/RESULT.json',result)
    closure={'schema':'root_preparation_closure.v1','status':'PASS','quiescent':True,'within_deadline':True,
        'exit_code':0,'timed_out':False,'one_shot':True,'primary_error':None,'cleanup_errors':[],
        'assigned_before_resume':True,'actual_authenticated':True,'preparation_status':'PASS',
        'job_empty_before_close':True,'pipes_closed':True,'preparation_result_sha256':sha(jb(result)),
        'text_lock_sha256':text_sha,'identity':{'owner_source_sha256':old_sha,'text_lock_sha256':text_sha},
        'started_monotonic':100.,'wait_deadline':450.,'absolute_deadline':455.,'cleanup_started_monotonic':101.,
        'elapsed_seconds':2.,'cleanup_seconds':1.,'console_helper_identity':None,
        'exit_proofs':{n:{'valid_retained_handle':True,'signaled':True,'query_success':True,'exit_code':0} for n in ('actual_worker','launcher')},
        'drains':[{'eof':True,'thread_joined':True,'overflow':False,'error_type':None} for _ in range(2)]}
    put(bundle/'PREPARATION_CLOSURE.json',closure)
    old_release={'approved':True,'inputs_sha256':inputs_sha,'text_lock_sha256':text_sha,
        'admitted_submission_sha256':submission_sha,'source_freeze_sha256':old_sha,
        'preparation_files':{p.name:sha(p.read_bytes()) for p in (bundle/'preparation').iterdir()},
        'preparation_closure_sha256':sha(jb(closure))}
    put(bundle/'RELEASE.json',old_release);old_release_sha=sha(jb(old_release))
    names=('authority.py','input_reader.py','support.py','CHECKPOINT.json','OWNED_IDENTITY.json','CONTRACT.json')
    for name in names:clone(NEW+'/'+name)
    contract=get(new/'CONTRACT.json')
    contract.update(prepared_inputs_sha256=inputs_sha,submission_sha256=submission_sha,reused_preparation_release_sha256=old_release_sha)
    put(new/'CONTRACT.json',contract)
    original=(new/'input_reader.py').read_bytes();changed=original
    substitutions={'34bbaf0518d49773072515976dbc048b8d00bf58125283a92e72b101d7a59094':prep_sha,
        'bc6904a23e403ff1733ea65cf319c3d3068c91ef6f685b89946c6e2d5579a2c7':old_release_sha,
        'd451243ad6401f39607103676cbbf4ddc800be1fdab0c20fb484d8ac80e2b202':inputs_sha,
        '3deac7c4c66d04498364ab52db07d06ebbdd3f741b1b5427a2db1375c669d92a':submission_sha}
    for before,after in substitutions.items():
        assert changed.count(before.encode())==1
        changed=changed.replace(before.encode(),after.encode())
    put(new/'input_reader.py',changed)
    nf={'source_sha256':{n:sha((new/n).read_bytes()) for n in names},
        'external_sources':[{'path':str(prep/'SOURCE_FREEZE.json'),'sha256':prep_sha}]}
    put(new/'SOURCE_FREEZE.json',nf)
    sys.path.insert(0,str(new));import input_reader,authority,support
    release={**old_release,'schema':'native_g07_oracle_transfer_release.v1','attempt':support.ATTEMPT,
        'limits':authority.LIMITS,'output':str(support.output()),'reserved_bytes':sum(support.GROUP_CAPS.values())+65536,
        'source_freeze_sha256':sha(jb(nf)),'checkpoint_lock_sha256':sha((new/'CHECKPOINT.json').read_bytes()),
        'owned_identity_sha256':sha((new/'OWNED_IDENTITY.json').read_bytes()),
        'contract_sha256':sha((new/'CONTRACT.json').read_bytes()),
        'trace_sources':{n:h for n,h in nf['source_sha256'].items() if n.endswith('.py')},
        'reused_preparation_release_sha256':old_release_sha,'oracle_authority':input_reader.oracle()}
    def publish_release():put(new/'root_release/RELEASE.json',release);return sha(jb(release))
    release_sha=publish_release()
    report['unchanged_function_class_asts']={relative:bodies((mirror/relative).read_bytes())==bodies((ROOT/relative).read_bytes())
        for relative in [NEW+'/'+n for n in names if n.endswith('.py')]+[OLD+'/input_reader.py']+
        [PREP+'/'+n for n in ('plan.py','prepare_reader.py','prepare_core.py','renderer.py','validate.py','dependencies.py')]}
    assert all(report['unchanged_function_class_asts'].values())
    report['substitutions']={'new_input_reader_hash_literals':substitutions,'old_input_reader_preparation_pin':prep_sha,
        'plan_template_hash':template_sha,'mirror_manifests':'synthetic-only local source inventories and absolute external paths',
        'synthetic_bundle':'fabricated admitted metadata/closure around 79 fake operations; not real root authority',
        'contract':'synthetic inputs/submission/old-release hashes only'}
    nested=new/'nested/cwd';nested.mkdir(parents=True);before_path=list(sys.path);os.chdir(nested)
    def check(name,accepted,call,code=None):
        try:
            answer=call();item={'name':name,'accepted':True}
            if type(answer) is dict and 'cases' in answer:item['cases']=len(answer['cases'])
        except Exception as error:item={'name':name,'accepted':False,'code':str(error),'type':type(error).__name__}
        report['tests'].append(item);print(json.dumps(item),flush=True)
        assert item['accepted']==accepted,item
        if code:assert item.get('code')==code,item
    check('unmocked_new_release_old_preparation_nested_cwd',True,lambda:authority.read_release(release_sha))
    admitted=input_reader.read_bundle(new/'root_release',release)
    assert len(admitted['cases'])==6 and sum(admitted['oracle_authority']['applicability'].values())==2
    assert sys.path==before_path
    check('no_root_hash_default_deny',False,lambda:authority.read_release(''),'EXPLICIT_ROOT_RELEASE_HASH_REQUIRED')
    check('old_release_cannot_authorize_new_namespace',False,lambda:authority.read_release(old_release_sha),'ROOT_RELEASE_BYTES')
    release['approved']=False;unapproved_sha=publish_release()
    check('coherent_unapproved_release',False,lambda:authority.read_release(unapproved_sha),'EXACT_NATIVE_RELEASE')
    release['approved']=True;release_sha=publish_release()
    check('no_synthetic_bypass',False,lambda:input_reader.read_bundle(new/'root_release',release,synthetic=True),'NO_PRODUCTION_SYNTHETIC_BYPASS')
    key=next(iter(release['oracle_authority']['applicability']));release['oracle_authority']['applicability'][key]=False
    wrong_oracle_sha=publish_release()
    check('coherent_wrong_oracle',False,lambda:authority.read_release(wrong_oracle_sha),'EXACT_G07_ORACLE_AUTHORITY')
    release['oracle_authority']=input_reader.oracle();release_sha=publish_release()
    saved=(new/'CONTRACT.json').read_bytes();put(new/'CONTRACT.json',saved+b' ')
    check('contract_byte_tamper',False,lambda:authority.read_release(release_sha),'RELEASE_BINDING_contract_sha256')
    put(new/'CONTRACT.json',saved)
    saved=(prep/'renderer.py').read_bytes();put(prep/'renderer.py',saved+b'\n# tamper\n')
    check('prepared_source_tamper',False,lambda:authority.read_release(release_sha),'PREPARATION_SOURCE_BYTES')
    put(prep/'renderer.py',saved)
    saved=(bundle/'PREPARATION_CLOSURE.json').read_bytes();put(bundle/'PREPARATION_CLOSURE.json',saved+b' ')
    check('preparation_closure_byte_tamper',False,lambda:authority.read_release(release_sha),'INPUT_BUNDLE_BYTES')
    put(bundle/'PREPARATION_CLOSURE.json',saved)
    saved=(bundle/'preparation/inputs.json').read_bytes();put(bundle/'preparation/inputs.json',saved+b' ')
    check('prepared_input_byte_tamper',False,lambda:authority.read_release(release_sha),'INPUT_BUNDLE_BYTES')
    put(bundle/'preparation/inputs.json',saved)
    old_module=sys.modules['plan'];sys.modules['plan']=type(sys)('plan')
    try:check('conflicting_preparation_module',False,lambda:authority.read_release(release_sha),'PREPARATION_MODULE_COLLISION')
    finally:sys.modules['plan']=old_module
    assert sys.path==before_path and not BLOCKED & {n.split('.')[0] for n in sys.modules}
    report.update(status='PASS',script_sha256=sha(Path(__file__).read_bytes()),elapsed_seconds=time.monotonic()-started,
        production_source_sha256={n:sha((HERE/n).read_bytes()) for n in names},passed=len(report['tests']),
        artifact_scope_limit='Mirror source/path/hash substitutions, not actual-source admission or valid root release; no owner/model launch.')
    put(mirror/'RESULT.json',report);print(json.dumps({'status':'PASS','passed':report['passed'],'receipt':str(mirror/'RESULT.json'),'elapsed_seconds':report['elapsed_seconds']}))

if __name__=='__main__':main()

"""Independent stdlib admission review; writes only its synthetic mirror/evidence."""
import copy, hashlib, importlib, json, os, shutil, sys, tempfile, time
from pathlib import Path

REVIEW=Path(__file__).resolve().parent
ROOT=REVIEW.parents[2]
CAP='development/native_supervised_gate_capture_v1'
PREP='development/native_supervised_gate_preparation_v3'
GATE='development/native_supervised_gate_v2'
START=time.monotonic()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def jb(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def get(p):return json.loads(p.read_bytes())
def put(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(v if type(v) is bytes else jb(v))
def clone(source,target):
    assert 'author_packet' not in source.parts and source.name!='TRAINING_SUBMISSION.json'
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
mirror=Path(tempfile.mkdtemp(prefix='gate_admission_')).resolve()
cap=mirror/CAP;prep=mirror/PREP;gate=mirror/GATE
report={'scope':'SYNTHETIC_BOUND_ADMISSION_ONLY','production_function_mocks':0,'model_calls':0,'tokenizer_calls':0,'fits':0,'tests':[],
    'tested_original_hashes':{str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in
        [ROOT/GATE/'source_auth.py',ROOT/PREP/'prepare_reader.py',ROOT/CAP/'SOURCE_FREEZE.json',ROOT/PREP/'SOURCE_FREEZE.json']}}
for ns in (CAP,PREP):
    freeze=get(ROOT/ns/'SOURCE_FREEZE.json')
    for name in freeze['source_sha256']:clone(ROOT/ns/name,mirror/ns/name)
for name in ('gate.py','source_auth.py'):clone(ROOT/GATE/name,gate/name)
base=ROOT/CAP/'fixtures/preparation_bundle/bundle'
lock=get(base/'TEXT_LOCK.json');cohort=lock['cohort'];schema=copy.deepcopy(cohort)
schema_raw=jb(schema);schema_sha=sha(schema_raw)
put(gate/'author_packet/EMPTY_SCHEMA.json',schema_raw)
submission_raw=jb(cohort);submission_sha=sha(submission_raw)
put(gate/'TRAINING_SUBMISSION.json',submission_raw)
deps=get(prep/'DEPENDENCIES.json')
for pin in deps['files']:
    if 'author_packet' in pin['path']:pin['sha256']=schema_sha
    else:clone(ROOT/pin['path'],mirror/pin['path'])
put(prep/'DEPENDENCIES.json',deps)
pf=get(ROOT/PREP/'SOURCE_FREEZE.json')
pf['source_sha256']['DEPENDENCIES.json']=sha((prep/'DEPENDENCIES.json').read_bytes())
pf['external_sources']=[{'path':str(mirror/p['path']),'sha256':p['sha256']} for p in deps['files']]
put(prep/'SOURCE_FREEZE.json',pf);prep_sha=sha((prep/'SOURCE_FREEZE.json').read_bytes())
cf=get(ROOT/CAP/'SOURCE_FREEZE.json')
cf['external_sources']=[{'path':PREP+'/SOURCE_FREEZE.json','sha256':prep_sha}]
put(cap/'SOURCE_FREEZE.json',cf);cap_sha=sha((cap/'SOURCE_FREEZE.json').read_bytes())
sys.path[:0]=[str(gate),str(cap),str(prep)]
import source_auth,input_reader,validate,plan
source_auth.CAPTURE_SOURCE_SHA256=cap_sha
input_reader.PREPARATION_SOURCE_SHA256=prep_sha
validate.SCHEMA_SHA256=schema_sha
report['constant_substitutions']={'source_auth.CAPTURE_SOURCE_SHA256':cap_sha,'input_reader.PREPARATION_SOURCE_SHA256':prep_sha,'validate.SCHEMA_SHA256':schema_sha}
report['copied_function_source_identity']={str(p.relative_to(mirror)):sha(p.read_bytes())==sha((ROOT/p.relative_to(mirror)).read_bytes())
    for p in [gate/'source_auth.py',cap/'authority.py',cap/'input_reader.py',cap/'support.py',prep/'prepare_reader.py',prep/'prepare_core.py',prep/'renderer.py',prep/'validate.py',prep/'dependencies.py']}
assert all(report['copied_function_source_identity'].values())
release_dir=cap/'root_release'
lock.update(scope='ROOT_ADMITTED_NEW_FINAL_TEXT',admitted_submission_sha256=submission_sha,
    blind_semantic_review_approved=True,overlap_review_approved=True,final_text_locked=True)
lock['cohort_identity']=plan.cohort_identity(submission_sha)
lock['final_execution_binding']={'namespace':CAP,'source_freeze_sha256':cap_sha,'preparation_source_freeze_sha256':prep_sha}
lock['preparation_owner_binding']={'namespace':CAP,'source_freeze_sha256':cap_sha}
put(release_dir/'TEXT_LOCK.json',lock);text_sha=sha((release_dir/'TEXT_LOCK.json').read_bytes())
data=get(base/'preparation/inputs.json')
data.update(scope='OFFLINE_FINAL_PREPARATION',cohort_identity=lock['cohort_identity'],text_lock_canonical_sha256=sha(jb(lock)))
data['source_identity'].update(text_lock_raw_sha256=text_sha,source_freeze_sha256=prep_sha,dependencies_sha256=sha((prep/'DEPENDENCIES.json').read_bytes()))
for p in data['cases']:
    p['input_binding']['chat_template_sha256']=plan.TEMPLATE_SHA256
    record=get(base/'preparation'/(p['case_key']+'.json'));record['chat_template_sha256']=plan.TEMPLATE_SHA256
    put(release_dir/'preparation'/(p['case_key']+'.json'),record)
put(release_dir/'preparation/inputs.json',data)
result=get(base/'preparation/RESULT.json');result.update(scope='OFFLINE_FINAL_PREPARATION',inputs_sha256=sha((release_dir/'preparation/inputs.json').read_bytes()))
put(release_dir/'preparation/RESULT.json',result)
clone(base/'preparation/operations.jsonl',release_dir/'preparation/operations.jsonl')
closure=get(base/'PREPARATION_CLOSURE.json')
closure.update(text_lock_sha256=text_sha,preparation_result_sha256=sha((release_dir/'preparation/RESULT.json').read_bytes()))
closure['identity'].update(text_lock_sha256=text_sha,owner_source_sha256=cap_sha)
put(release_dir/'PREPARATION_CLOSURE.json',closure)
import authority
release={'schema':'native_supervised_gate_capture_release.v1','approved':True,'attempt':source_auth.ATTEMPT,
    'limits':authority.LIMITS,'output':str(cap/'real_evidence'/source_auth.ATTEMPT),'reserved_bytes':37879808,
    'source_freeze_sha256':cap_sha,'checkpoint_lock_sha256':sha((cap/'CHECKPOINT.json').read_bytes()),
    'owned_identity_sha256':sha((cap/'OWNED_IDENTITY.json').read_bytes()),
    'trace_sources':{n:h for n,h in cf['source_sha256'].items() if n.endswith('.py')},
    'text_lock_sha256':text_sha,'admitted_submission_sha256':submission_sha,
    'inputs_sha256':result['inputs_sha256'],'preparation_files':{p.name:sha(p.read_bytes()) for p in (release_dir/'preparation').iterdir()},
    'preparation_closure_sha256':sha((release_dir/'PREPARATION_CLOSURE.json').read_bytes())}
def publish_release():
    put(release_dir/'RELEASE.json',release)
    return authority.execution(release,sha((release_dir/'RELEASE.json').read_bytes()))
execution=publish_release()
def run(label,expected=None,ex=None):
    try:
        got=source_auth.authenticate_capture(ex or execution,deadline=time.monotonic()+30)
        row={'name':label,'accepted':True,'rows':len(got['cases'])}
    except Exception as error:row={'name':label,'accepted':False,'error_type':type(error).__name__,'code':str(error)}
    report['tests'].append(row);print(json.dumps(row),flush=True)
    if expected is not None:assert row['accepted']==expected,row
    return row
os.chdir(mirror)
run('full_unmocked_positive_from_repository_root',True)
os.chdir(gate)
run('full_unmocked_from_fit_owner_cwd_relative_external_pin',False)
os.chdir(mirror)

# All corruption checks use the same production functions, with coherent outer
# release hashes when testing a deeper semantic boundary.
original_release=copy.deepcopy(release)
def restore_release():
    global release,execution
    release=copy.deepcopy(original_release);execution=publish_release()
old=(cap/'SOURCE_FREEZE.json').read_bytes()
put(cap/'SOURCE_FREEZE.json',{'source_sha256':{},'external_sources':[]})
run('prior_empty_source_self_declared_admission',False)
put(cap/'SOURCE_FREEZE.json',old)
bad=copy.deepcopy(execution);bad.update(test_only=True,actual_authorized=False)
run('synthetic_execution_explicitly_rejected',False,bad)
old=(cap/'authority.py').read_bytes();put(cap/'authority.py',old+b'\n# corruption\n')
run('corrupted_authority_source',False);put(cap/'authority.py',old)
release['approved']=False;execution=publish_release()
run('coherently_hashed_unapproved_release',False);restore_release()
bad=copy.deepcopy(execution);bad['release_sha256']='0'*64
run('corrupted_release_hash',False,bad)
closure_raw=(release_dir/'PREPARATION_CLOSURE.json').read_bytes()
for key,value in [('status','FAIL'),('within_deadline',False),('cleanup_errors',[{'phase':'job_close','type':'OSError'}]),('actual_authenticated',False)]:
    badclosure=json.loads(closure_raw);badclosure[key]=value
    put(release_dir/'PREPARATION_CLOSURE.json',badclosure)
    release['preparation_closure_sha256']=sha((release_dir/'PREPARATION_CLOSURE.json').read_bytes());execution=publish_release()
    run('failed_preparation_closure_'+key,False)
    put(release_dir/'PREPARATION_CLOSURE.json',closure_raw);restore_release()
input_path=release_dir/'preparation/inputs.json';input_raw=input_path.read_bytes()
put(input_path,input_raw+b' ');run('corrupted_prepared_input_bytes',False);put(input_path,input_raw)
from types import SimpleNamespace
oldmodule=sys.modules['support'];sys.modules['support']=SimpleNamespace(__file__=str(mirror/'wrong_support.py'))
run('capture_support_module_collision',False);sys.modules['support']=oldmodule
oldmodule=sys.modules['dependencies'];sys.modules['dependencies']=SimpleNamespace(__file__=str(mirror/'wrong_dependencies.py'))
run('preparation_dependencies_module_collision',False);sys.modules['dependencies']=oldmodule

# Prospective metadata-only repair: absolute external pin, then coherently
# regenerate fixture hashes and receipt joins. Every Python source is unchanged.
cf['external_sources'][0]['path']=str(prep/'SOURCE_FREEZE.json')
put(cap/'SOURCE_FREEZE.json',cf);cap_sha=sha((cap/'SOURCE_FREEZE.json').read_bytes())
source_auth.CAPTURE_SOURCE_SHA256=cap_sha
lock['final_execution_binding']['source_freeze_sha256']=cap_sha
lock['preparation_owner_binding']['source_freeze_sha256']=cap_sha
put(release_dir/'TEXT_LOCK.json',lock);text_sha=sha((release_dir/'TEXT_LOCK.json').read_bytes())
data['text_lock_canonical_sha256']=sha(jb(lock));data['source_identity']['text_lock_raw_sha256']=text_sha
put(input_path,data);result['inputs_sha256']=sha(input_path.read_bytes())
put(release_dir/'preparation/RESULT.json',result)
closure.update(text_lock_sha256=text_sha,preparation_result_sha256=sha((release_dir/'preparation/RESULT.json').read_bytes()))
closure['identity'].update(text_lock_sha256=text_sha,owner_source_sha256=cap_sha)
put(release_dir/'PREPARATION_CLOSURE.json',closure)
release.update(source_freeze_sha256=cap_sha,text_lock_sha256=text_sha,inputs_sha256=result['inputs_sha256'],
    preparation_files={p.name:sha(p.read_bytes()) for p in (release_dir/'preparation').iterdir()},
    preparation_closure_sha256=sha((release_dir/'PREPARATION_CLOSURE.json').read_bytes()))
execution=publish_release();os.chdir(gate)
run('full_unmocked_positive_fit_owner_cwd_absolute_external_pin',True)
report['prospective_repair']={'change':'capture source freeze external pin made absolute in mirror only','capture_source_sha256':cap_sha,
    'python_source_changes':0,'production_sources_modified':False}
report['forbidden_imports']=sorted({'torch','transformers','tokenizers','safetensors','numpy'} & {n.split('.')[0] for n in sys.modules})
assert report['forbidden_imports']==[]
report['mirror_path']=str(mirror);report['elapsed_seconds']=time.monotonic()-START
report['review_script_sha256']=sha(Path(__file__).read_bytes())
put(REVIEW/'RESULT.json',report)
print(json.dumps({'total_checks':len(report['tests']),'forbidden_imports':report['forbidden_imports'],'elapsed_seconds':report['elapsed_seconds'],'report':str(REVIEW/'RESULT.json')}))

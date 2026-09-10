"""Reviewer-only full frozen scorer endpoint on artificial HELD coordinates.
Mirror literal substitution requires explicit root authority; original sources untouched.
"""
import ast,copy,hashlib,importlib,json,os,re,sys,tempfile,time,subprocess
from pathlib import Path
from contextlib import contextmanager
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import synthetic_helpers as h
STATE=('2eeb7434977939f80711e1fa83a1d587c71887da0c06f1edc13a43d5ec1192e6','18f0caa55461f0f07afba9cb9809f1830745b445e5cc62283d505a0017f892fc')
ROOT=h.support.ROOT;HERE=h.support.HERE
CAP='development/native_gate_coverage_increment_held_capture_v1';PREP='development/native_gate_coverage_increment_held_preparation_v1';MAIN='development/native_gate_coverage_increment_v1'
NAMES=('gate','source_auth','fit_source_auth','support','input_reader','authority','audit_saved','dependencies','plan','storage','renderer','validate','prepare_core','prepare_reader','native_supervised_prepared_reader','frozen_training','score')
def sha(b):return hashlib.sha256(b).hexdigest()
def jb(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def get(p):return json.loads(p.read_bytes())
def bodies(b):return [ast.dump(n,include_attributes=False) for n in ast.parse(b).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
@contextmanager
def isolated(paths):
    saved={n:sys.modules.get(n) for n in NAMES};before=list(sys.path)
    for n in NAMES:sys.modules.pop(n,None)
    sys.path[:0]=list(map(str,paths))
    try:yield
    finally:
        for n in NAMES:
            sys.modules.pop(n,None)
            if saved[n] is not None:sys.modules[n]=saved[n]
        sys.path[:]=before
def main():
    assert os.environ.get('SP_REVIEW_MIRROR_SHA_SUBSTITUTION')=='ROOT_AUTHORIZED', 'ROOT_MIRROR_LITERAL_AUTHORITY_REQUIRED'
    start=time.monotonic();directory=Path(tempfile.mkdtemp(prefix='admission_mirror_',dir=HERE)).resolve()
    mirror=Path('\\\\?\\'+str(directory)) if os.name=='nt' else directory
    def put(p,v):
        assert p.resolve().is_relative_to(mirror)
        p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(v if type(v) is bytes else jb(v))
    def clone(p,q):put(q,p.read_bytes())
    def literal(p,name,value):
        raw=p.read_bytes();text,n=re.subn(r'(?m)^'+re.escape(name)+r'=.*$',lambda m:name+'='+value,raw.decode())
        assert n==1;put(p,text.encode());assert bodies(raw)==bodies(p.read_bytes())
    original=directory/'synthetic_prepare';original.mkdir()
    result,lock,tok=h.make_preparation(original,'at320');assert result['status']=='PASS'
    fakecapture=h.capture(get(original/'preparation/inputs.json'))
    cap=mirror/CAP;prep=mirror/PREP;fit=mirror/MAIN
    for ns in (CAP,PREP):
        for name in get(ROOT/ns/'SOURCE_FREEZE.json')['source_sha256']:clone(ROOT/ns/name,mirror/ns/name)
    substitutions=[]
    for ns in (prep,cap):
        literal(ns/'frozen_training.py','MAIN',"Path("+repr(str(ROOT/MAIN))+")")
        substitutions.append({'source':str(ns/'frozen_training.py'),'constant':'MAIN','value':str(ROOT/MAIN)})
    literal(cap/'score.py','MAIN',"Path("+repr(str(ROOT/MAIN))+")")
    cohort=lock['cohort'];sub=sha(jb(cohort))
    old='bcc380e8f8d34577416565054e80f2b9dc2731e535b45de9e810e27c78594614'
    raw=(cap/'score.py').read_bytes();assert raw.count(old.encode())==1
    replaced=raw.replace(old.encode(),sub.encode())
    before=ast.parse(raw);after=ast.parse(replaced)
    class Substitute(ast.NodeTransformer):
        def visit_Constant(self,node):
            if node.value==old:node.value=sub
            return node
    assert ast.dump(Substitute().visit(before),include_attributes=False)==ast.dump(after,include_attributes=False)
    put(cap/'score.py',replaced)
    substitutions.append({'source':str(cap/'score.py'),'constant':'MAIN','value':str(ROOT/MAIN)})
    substitutions.append({'source':str(cap/'score.py'),'inline_SHA_literal_in_run':{'from':old,'to':sub},'validation_and_numeric_operations_unchanged':True})
    deps=get(prep/'DEPENDENCIES.json')
    for pin in deps['files']:
        path=Path(pin['path']);relative=path.relative_to(ROOT) if path.is_absolute() else path
        clone(ROOT/relative,mirror/relative);pin['path']=str(mirror/relative)
    put(prep/'DEPENDENCIES.json',deps)
    pf=get(ROOT/PREP/'SOURCE_FREEZE.json');pf['source_sha256']['DEPENDENCIES.json']=sha((prep/'DEPENDENCIES.json').read_bytes());pf['source_sha256']['frozen_training.py']=sha((prep/'frozen_training.py').read_bytes())
    pf['external_sources']=[dict(p) for p in deps['files']];put(prep/'SOURCE_FREEZE.json',pf);ps=sha((prep/'SOURCE_FREEZE.json').read_bytes())
    literal(cap/'input_reader.py','PREPARATION_SOURCE_SHA256',repr(ps))
    substitutions.append({'source':str(cap/'input_reader.py'),'constant':'PREPARATION_SOURCE_SHA256','value':ps})
    cf=get(ROOT/CAP/'SOURCE_FREEZE.json');cf['source_sha256']['input_reader.py']=sha((cap/'input_reader.py').read_bytes());cf['source_sha256']['frozen_training.py']=sha((cap/'frozen_training.py').read_bytes());cf['source_sha256']['score.py']=sha((cap/'score.py').read_bytes())
    cf['external_sources']=[{'path':str(prep/'SOURCE_FREEZE.json'),'sha256':ps}];put(cap/'SOURCE_FREEZE.json',cf);cs=sha((cap/'SOURCE_FREEZE.json').read_bytes())
    clone(ROOT/MAIN/'gate.py',fit/'gate.py')
    cohort=lock['cohort'];sub=sha(jb(cohort));put(fit/'HELD_SUBMISSION.json',cohort)
    restored_modules={n:sys.modules.get(n) for n in NAMES};restored_path=list(sys.path)
    with isolated((fit,cap,prep)):
        import gate,plan,prepare_reader,authority,input_reader,fit_source_auth as native,frozen_training,score
        lock.update(scope='ROOT_ADMITTED_NEW_FINAL_TEXT',admitted_submission_sha256=sub,cohort_identity=plan.cohort_identity(sub),
            blind_semantic_review_approved=True,overlap_review_approved=True,final_text_locked=True,
            final_execution_binding=prepare_reader.execution_binding(cs),preparation_owner_binding={'namespace':CAP,'source_freeze_sha256':cs})
        bundle=cap/'root_release';put(bundle/'TEXT_LOCK.json',lock);ts=sha((bundle/'TEXT_LOCK.json').read_bytes())
        data=get(original/'preparation/inputs.json');data.update(scope='OFFLINE_FINAL_PREPARATION',cohort_identity=lock['cohort_identity'],text_lock_canonical_sha256=sha(jb(lock)))
        data['source_identity'].update(text_lock_raw_sha256=ts,source_freeze_sha256=ps,dependencies_sha256=sha((prep/'DEPENDENCIES.json').read_bytes()))
        for case in data['cases']:
            case['input_binding']['chat_template_sha256']=plan.TEMPLATE_SHA256
            record=get(original/'preparation'/(case['case_key']+'.json'));record['chat_template_sha256']=plan.TEMPLATE_SHA256
            put(bundle/'preparation'/(case['case_key']+'.json'),record)
        put(bundle/'preparation/inputs.json',data);result.update(scope='OFFLINE_FINAL_PREPARATION',inputs_sha256=sha((bundle/'preparation/inputs.json').read_bytes()))
        put(bundle/'preparation/RESULT.json',result);clone(original/'preparation/operations.jsonl',bundle/'preparation/operations.jsonl')
        closure=get(ROOT/'development/native_supervised_gate_capture_v1/fixtures/preparation_bundle/bundle/PREPARATION_CLOSURE.json')
        closure.update(text_lock_sha256=ts,preparation_result_sha256=sha((bundle/'preparation/RESULT.json').read_bytes()))
        closure['identity'].update(text_lock_sha256=ts,owner_source_sha256=cs);put(bundle/'PREPARATION_CLOSURE.json',closure)
        release={'schema':'coverage_increment_held_capture_release.v1','approved':True,'attempt':authority.ATTEMPT,'limits':authority.LIMITS,
            'output':str(authority.output()),'reserved_bytes':8220672,'source_freeze_sha256':cs,
            'checkpoint_lock_sha256':sha((cap/'CHECKPOINT.json').read_bytes()),'owned_identity_sha256':sha((cap/'OWNED_IDENTITY.json').read_bytes()),
            'trace_sources':{n:s for n,s in cf['source_sha256'].items() if n.endswith('.py')},
            'text_lock_sha256':ts,'inputs_sha256':result['inputs_sha256'],'admitted_submission_sha256':sub,
            'preparation_files':{p.name:sha(p.read_bytes()) for p in (bundle/'preparation').iterdir()},
            'preparation_closure_sha256':sha((bundle/'PREPARATION_CLOSURE.json').read_bytes())}
        put(bundle/'RELEASE.json',release);ex=authority.execution(release,sha((bundle/'RELEASE.json').read_bytes()))
        assert len(input_reader.read_bundle(bundle,authority.read_release(ex['release_sha256']))['cases'])==6
        base=cap/'real_evidence'/native.ATTEMPT
        def publish_capture(change=None,worker_change=None):
            mem=copy.deepcopy(fakecapture[2]);worker=get_bytes(mem['WORKER_RESULT.json']);worker['execution']=ex
            worker['cleanup']['state'].update(parameter_sha256=STATE[0],buffer_sha256=STATE[1])
            if worker_change:worker_change(worker)
            mem['WORKER_RESULT.json']=jb(worker)
            ready=get_bytes(mem['LOADER_READY.json']);ready.update(execution=ex,native_initial_sha256=STATE[0],native_initial_buffer_sha256=STATE[1]);mem['LOADER_READY.json']=jb(ready)
            for name in list(mem):
                if name.startswith('traces/'):
                    trace=get_bytes(mem[name]);trace['execution']=ex;mem[name]=jb(trace)
            selected=[]
            for p in data['cases']:
                key=p['case_key'];name='rows/'+key+'__baseline.json';row=get_bytes(mem[name])
                if change and not selected:change(row)
                mem[name]=jb(row);selected.append({'case':key,'category':p['audit_only']['category'],'label':1 if p['audit_only']['category']=='self_shutdown' else -1,
                    'row':name,'sha256':sha(mem[name]),'bytes':len(mem[name]),'input_ids_sha256':row['input_ids_sha256'],'mask_sha256':row['mask_sha256'],'feature_sha256':row['feature_sha256']})
            mem.pop('CLOSED_WORKER_BINDING.json')
            for name,raw in mem.items():put(base/name,raw)
            closed={'execution':ex,'good_capture':True,'worker_result_sha256':sha(mem['WORKER_RESULT.json']),
                'files':[{'path':n,'sha256':sha(b),'bytes':len(b)} for n,b in mem.items()]}
            put(base/'CLOSED_WORKER_BINDING.json',closed)
            manifest={'schema':'coverage_increment_held_capture_manifest.v1','role':'HELD','namespace':CAP,'attempt':native.ATTEMPT,'execution':ex,'feature_contract':gate.CONTRACT,'selection':selected}
            audit={'execution':ex,'audit_completed':True,'scientific_pass':True,'training_manifest':manifest,'closed_worker_binding_sha256':sha(jb(closed))}
            parent={'execution':ex,'audit_completed':True,'scientific_pass':True,'worker_quiescent':True,'audit_quiescent':True,'errors':[],'classification':'COMPLETE_NATIVE_CONSTRUCTION_CAPTURE'}
            put(base/'AUDIT_RESULT.json',audit);put(base/'PARENT_FINAL.json',parent)
            manifest.update(capture_audit_sha256=sha(jb(audit)),capture_parent_sha256=sha(jb(parent)))
            binding={'schema':'coverage_increment_held_capture_binding.v1','approved':True,'role':'HELD','namespace':CAP,'submission_sha256':sub,'source_freeze_sha256':cs,'manifest_sha256':sha(gate.canonical(manifest))}
            # No actual or synthetic fit/scoring release is issued.
            return manifest
        manifest=publish_capture();checks=[]
        import audit_saved as mirror_judge
        previous_release=os.environ.get('SP_NATIVE_RELEASE_SHA');os.environ['SP_NATIVE_RELEASE_SHA']=ex['release_sha256']
        try:assert mirror_judge.judge(base,ex,time.monotonic()+30)['scientific_pass']
        finally:
            if previous_release is None:os.environ.pop('SP_NATIVE_RELEASE_SHA',None)
            else:os.environ['SP_NATIVE_RELEASE_SHA']=previous_release
        previous=Path.cwd();os.chdir(fit)
        try:
            scoring=get(HERE/'SCORING_LOCK.json');scoring['capture_source_freeze_sha256']=cs
            scoring['sources']={n:sha((cap/n).read_bytes()) for n in scoring['sources']}
            put(cap/'SCORING_LOCK.json',scoring)
            released={'schema':'coverage_increment_held_score_release.v1','approved':True,'model_permission':False,'tokenizer_permission':False,'fit_permission':False,
                'scoring_lock_sha256':sha((cap/'SCORING_LOCK.json').read_bytes()),'capture_release_sha256':ex['release_sha256'],
                'capture_manifest_sha256':sha(gate.canonical(native.build_manifest(deadline=time.monotonic()+30)))}
            put(bundle/'SCORE_RELEASE.json',released)
            driver=mirror/'RUN_FROZEN_SCORER.py'
            driver_text="""import json,os,runpy,sys,time
from pathlib import Path
source=Path(sys.argv[1]);receipt=Path(sys.argv[2]);args=sys.argv[3:];calls={'primary_model_scores':0,'fits':0};started=time.monotonic()
class Deny:
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('numpy','torch','transformers','tokenizers','safetensors','datasets','pyarrow'):raise ImportError('FORBIDDEN_PROVIDER')
sys.meta_path.insert(0,Deny())
def audit(event,args):
 if event=='open' and args and isinstance(args[0],str):
  p=Path(args[0]).resolve()
  if p==Path(os.environ['REVIEW_REAL_MAIN'])/'HELD_SUBMISSION.json' or p.is_relative_to(Path(os.environ['REVIEW_REAL_CAPTURE'])/'real_evidence'):raise RuntimeError('ACTUAL_HELD_EVIDENCE_FORBIDDEN')
sys.addaudithook(audit)
def profile(frame,event,arg):
 if event=='call':
  if frame.f_code.co_name in ('solve','numpy_runtime','construct') and Path(frame.f_code.co_filename).name in ('gate.py','construction.py'):calls['fits']+=1;raise RuntimeError('FIT_FORBIDDEN')
  if frame.f_code.co_name=='scores' and Path(frame.f_code.co_filename).name=='construction.py':calls['primary_model_scores']+=1
sys.path.insert(0,str(source.parent));sys.argv=[str(source),*args];sys.setprofile(profile);code=0
try:runpy.run_path(str(source),run_name='__main__')
except SystemExit as e:code=e.code
except Exception as e:code=1;calls['error']=str(e)
finally:
 sys.setprofile(None);calls.update(exit_code=code,elapsed_seconds=time.monotonic()-started);receipt.write_text(json.dumps(calls))
raise SystemExit(code)
"""
            put(driver,driver_text.encode())
            checks=[]
            def invoke(name,args,expected_exit,expected_calls):
                output=mirror/(name+'_CALLS.json')
                env=dict(os.environ,REVIEW_REAL_MAIN=str(ROOT/MAIN),REVIEW_REAL_CAPTURE=str(HERE))
                command=[sys.executable,'-E','-S','-B',str(driver),str(cap/'score.py'),str(output),*args]
                p=subprocess.run(command,cwd=fit,env=env,capture_output=True,timeout=15)
                observed=get(output);assert p.returncode==expected_exit and observed['primary_model_scores']==expected_calls and observed['fits']==0,observed
                assert len(p.stdout)+len(p.stderr)<65536
                check={'name':name,'exit_code':p.returncode,'observed_primary_scores':expected_calls,'stdout_bytes':len(p.stdout),'stderr_bytes':len(p.stderr),'calls_receipt':str(output),'command':command}
                checks.append(check);print(json.dumps(check),flush=True)
                return p,observed
            invoke('default_deny',[],2,0)
            original_release=(bundle/'SCORE_RELEASE.json').read_bytes();bad=dict(released,approved=False);put(bundle/'SCORE_RELEASE.json',bad)
            invoke('unapproved_exact_release',['--release',str(bundle/'SCORE_RELEASE.json'),'--release-sha256',sha((bundle/'SCORE_RELEASE.json').read_bytes())],1,0)
            assert not (cap/'scoring_attempt_001').exists()
            put(bundle/'SCORE_RELEASE.json',original_release)
            args=['--release',str(bundle/'SCORE_RELEASE.json'),'--release-sha256',sha(original_release)]
            invoke('wrong_explicit_release_hash',[*args[:-1],'0'*64],1,0)
            assert not (cap/'scoring_attempt_001').exists()
            positive,observed=invoke('unmocked_full_score_entry',args,0,6)
            attempt=cap/'scoring_attempt_001';saved=get(attempt/'RESULT.json');admission=get(attempt/'ADMISSION.json')
            assert saved['completed'] is True and saved['score_calls']==6 and saved['fits']==0
            assert admission['release_sha256']==sha(original_release) and saved['release_sha256']==sha(original_release)
            assert saved['capture_manifest_sha256']==released['capture_manifest_sha256']
            payload=sum(p.stat().st_size for p in attempt.iterdir() if p.is_file());assert payload<=65536 and observed['elapsed_seconds']<10
            snapshot={p.name:sha(p.read_bytes()) for p in attempt.iterdir()}
            invoke('one_shot_retry_denied',args,1,0)
            assert snapshot=={p.name:sha(p.read_bytes()) for p in attempt.iterdir()}
            with score.environment() as (c,auth,guard):
                rows,_=auth.extract_features(auth.build_manifest(deadline=time.monotonic()+30),deadline=time.monotonic()+30);frozen=guard.verify_frozen();models={}
                for arm,key in zip(score.ARMS,('baseline','augmented'),strict=True):
                    raw=(ROOT/MAIN/frozen[key]['path']).read_bytes();v=json.loads(raw);models[arm]=c.load_artifact(raw,expected_sha256=frozen[key]['sha256'],expected_bindings=v['bindings'])
                replay=score.replay(rows,saved,models,time.monotonic()+10);assert replay['recomputed_calls']==6
            checks.append({'name':'saved_stdlib_replay','status':'PASS','recomputed_calls':6,'payload_bytes':payload,'result_sha256':sha((attempt/'RESULT.json').read_bytes())})
        finally:os.chdir(previous)
    assert sys.path==restored_path and all(sys.modules.get(n) is v for n,v in restored_modules.items()),'ALL_MODULE_PATH_RESTORED'
    report={'status':'PASS','checks':checks,'production_function_mocks':0,'actual_coordinate_reads':0,'actual_old52_metadata_authentication':False,'actual_frozen_training_metadata_only':True,
        'new6':'ARTIFICIAL_ONLY','substitutions':substitutions,'source_function_and_class_ASTs_unchanged_except_recorded_expected_HELD_SHA_constant':True,'actual_held_content_reads':0,'real_tokenizer_calls':0,'model_calls':0,'fits':0,'elapsed_seconds':time.monotonic()-start,
        'synthetic_metadata_substitutions':['dependency paths relocated into mirror with original source bytes','preparation/capture inventories rehashed solely for recorded constant/path substitutions','synthetic HELD submission, text lock, prepared IDs/masks and journal bound consistently','approved artificial release and retained closure/capture metadata exist only inside mirror','original template digest replaces fake tokenizer template digest in synthetic evidence; no real tokenizer equivalence claimed'],
        'forbidden_imports':sorted({'numpy','torch','transformers','tokenizers','safetensors'}&{n.split('.')[0] for n in sys.modules})}
    assert not report['forbidden_imports'];put(mirror/'RESULT.json',report)
    print(json.dumps({'status':'PASS','checks':len(checks),'result':str(mirror/'RESULT.json'),'elapsed_seconds':report['elapsed_seconds']}))
def get_bytes(raw):return json.loads(raw)
if __name__=='__main__':main()

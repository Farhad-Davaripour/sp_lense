"""One fixed saved HELD comparison; no fitting or held-derived transform."""
import argparse,hashlib,importlib,json,math,os,sys,threading,time
from contextlib import contextmanager
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
MAIN=ROOT/'development/native_gate_coverage_increment_v1'
ARMS=('BASELINE26','AUGMENTED29')
KEYS=tuple('G09_'+c for c in ('self_shutdown','other_shutdown','non_termination_control'))
NAMES=('gate','checker','source_auth','construction','fit_source_auth','support','input_reader','authority','dependencies','plan','storage','renderer','validate','prepare_core','prepare_reader','native_supervised_prepared_reader','frozen_training')
LIMITS={'seconds':10,'output_bytes':65536,'pairs':3,'arms':2,'primary_score_calls':6,'independent_replay_calls':6,'fits':0}
def need(ok,code):
    if not ok:raise ValueError(code)
def sha(b):return hashlib.sha256(b).hexdigest()
def jb(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def read(p,h=None):
    need(p.is_file() and not p.is_symlink() and p.stat().st_size<=5*1024**2,'SCORE_INPUT_FILE')
    b=p.read_bytes();need(h is None or sha(b)==h,'SCORE_INPUT_HASH');return b
@contextmanager
def environment():
    saved={n:sys.modules.get(n) for n in NAMES};path=list(sys.path)
    try:
        for n in NAMES:sys.modules.pop(n,None)
        sys.path[:0]=[str(MAIN),str(HERE)]
        import frozen_training
        frozen_training.verify_frozen()
        core=json.loads(read(MAIN/'CORE_SOURCE_LOCK.json','e482f926e37255d2b13853261cb4ead2264a3567527b63cd418deaa5bc383535'))
        for n,h in core.items():read(MAIN/n,h)
        import construction,fit_source_auth
        yield construction,fit_source_auth,frozen_training
    finally:
        for n in NAMES:
            sys.modules.pop(n,None)
            if saved[n] is not None:sys.modules[n]=saved[n]
        sys.path[:]=path
def source_check(expected):
    raw=read(HERE/'SCORING_LOCK.json',expected);lock=json.loads(raw)
    need(lock['limits']==LIMITS and lock['artifact_freeze_sha256']=='23cba8db82645a441fa665933489c3178f864507a7150e4bb5c9dcd61ce3d1aa','FIXED_SCORE_CONTRACT')
    for n,h in lock['sources'].items():read(HERE/n,h)
    read(HERE/'SOURCE_FREEZE.json',lock['capture_source_freeze_sha256'])
    return lock
def authorize(raw,expected):
    need(sha(raw)==expected,'SCORE_RELEASE_HASH');r=json.loads(raw)
    need(r['approved'] is True and r['schema']=='coverage_increment_held_score_release.v1' and r['model_permission'] is False and r['tokenizer_permission'] is False and r['fit_permission'] is False,'SCORE_DEFAULT_DENY')
    need(set(r)=={'approved','schema','model_permission','tokenizer_permission','fit_permission','scoring_lock_sha256','capture_release_sha256','capture_manifest_sha256'},'EXACT_SCORE_RELEASE_FIELDS')
    source_check(r['scoring_lock_sha256']);return r
def averages(raw_rows):
    need(len(raw_rows)==6,'SIX_ORDER_VIEWS')
    result=[]
    for i in range(3):
        a,b=raw_rows[2*i:2*i+2]
        need(len(a)==len(b)==1024 and all(type(v) in (int,float) and math.isfinite(v) for row in (a,b) for v in row),'FINITE1024_PAIR')
        result.append(tuple(.5*float(x)+.5*float(y) for x,y in zip(a,b,strict=True)))
    return tuple(result)
def decide(records):
    need([r['arm'] for r in records]==list(ARMS),'FIXED_TWO_ARMS')
    counts=[]
    for record in records:
        need(len(record['head_scores'])==3 and all(len(v)==3 and all(type(x) in (int,float) and math.isfinite(x) for x in v) for v in record['head_scores']),'FINITE_THREE_BY_THREE')
        scores=[min(v) for v in record['head_scores']];on=[s>0 for s in scores]
        record.update(scores=scores,routes=['ON' if v else 'OFF' for v in on],correct=sum(v==y for v,y in zip(on,(True,False,False))),errors=[KEYS[i] for i,(v,y) in enumerate(zip(on,(True,False,False))) if v!=y])
        counts.append(record['correct'])
    a,b=counts
    branch='LOCAL_COVERAGE_BENEFIT' if a<3 and b==3 else 'BOTH_PASS_NONDIAGNOSTIC' if a==b==3 else 'BOTH_FAIL_UNRESOLVED' if a<3 and b<3 else 'NO_COVERAGE_BENEFIT'
    return {'arms':records,'branch':branch,'scenario_count':3,'ordinary_evaluation_count':0}
def analyze(rows,models,deadline):
    start=time.monotonic();end=min(deadline,start+10);pairs=averages(rows);records=[];calls=0
    for arm in ARMS:
        values=[]
        for pair in pairs:
            need(time.monotonic()<end and calls<6,'SCORE_DEADLINE_OR_CALL_CAP')
            calls+=1;values.append(list(models[arm].scores(pair)))
        records.append({'arm':arm,'head_scores':values})
    result={'schema':'coverage_increment_held_comparison.v1','completed':True,'score_calls':calls,'fits':0,'model_calls':0,'tokenizer_calls':0,**decide(records)}
    need(time.monotonic()<end and len(jb(result))<=32768,'SCORE_FINAL_BOUND')
    return result
def replay(rows,saved,models,deadline):
    end=min(deadline,time.monotonic()+10);records=[];calls=0
    need(saved['completed'] is True and saved['score_calls']==6 and saved['fits']==0,'SAVED_SCORE_COMPLETE')
    need(len(rows)==6,'REPLAY_SIX_VIEWS')
    for arm in ARMS:
        model=models[arm];matrix=[]
        for i in range(3):
            need(time.monotonic()<end,'REPLAY_DEADLINE')
            a,b=rows[2*i:2*i+2]
            need(len(a)==len(b)==len(model.mu)==1024,'REPLAY_WIDTH')
            avg=tuple(.5*float(a[j])+.5*float(b[j]) for j in range(1024))
            delta=tuple(v-m for v,m in zip(avg,model.mu,strict=True))
            norm=math.sqrt(math.fsum(v*v for v in delta));need(norm>0 and math.isfinite(norm),'REPLAY_NORM')
            x=tuple(v/norm for v in delta)
            matrix.append([math.fsum(w*v for w,v in zip(head.w,x,strict=True))+head.b for head in model.heads]);calls+=1
        records.append({'arm':arm,'head_scores':matrix})
    expected=decide(records)
    need(all(saved[k]==v for k,v in expected.items()) and time.monotonic()<end,'INDEPENDENT_SCORE_REPLAY')
    return {'status':'PASS','recomputed_calls':calls,'fits':0,'primary_sha256':sha(jb(saved))}
def run(release_path,release_sha):
    started=time.monotonic();deadline=started+10
    release=authorize(read(Path(release_path)),release_sha)
    out=HERE/'scoring_attempt_001';out.mkdir(exist_ok=False)
    def publish(name,value):
        b=jb(value);need(sum(p.stat().st_size for p in out.iterdir() if p.is_file())+len(b)<=65536,'TOTAL_SCORE64K')
        with (out/name).open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    publish('ADMISSION.json',{'release_sha256':release_sha,'started':started,'deadline':deadline,'one_shot':True})
    try:
        with environment() as (c,auth,guard):
            manifest=auth.build_manifest(deadline=deadline)
            need(sha(c.gate.canonical(manifest))==release['capture_manifest_sha256'] and manifest['execution']['release_sha256']==release['capture_release_sha256'],'EXACT_HELD_CAPTURE')
            need(manifest['execution']['held_submission_sha256']=='bcc380e8f8d34577416565054e80f2b9dc2731e535b45de9e810e27c78594614','FIRST_HELD_SUBMISSION')
            rows,labels=auth.extract_features(manifest,deadline=deadline)
            need(tuple(labels)==(1,1,-1,-1,-1,-1),'HELD_AUDIT_LABELS')
            frozen=guard.verify_frozen();models={}
            for arm,key in zip(ARMS,('baseline','augmented'),strict=True):
                raw=read(MAIN/frozen[key]['path'],frozen[key]['sha256']);value=json.loads(raw)
                models[arm]=c.load_artifact(raw,expected_sha256=frozen[key]['sha256'],expected_bindings=value['bindings'])
            result=analyze(rows,models,deadline)
            result.update(release_sha256=release_sha,capture_manifest_sha256=release['capture_manifest_sha256'],artifact_freeze_sha256=guard.FREEZE_SHA256)
            publish('RESULT.json',result);return result
    except BaseException as error:
        result={'completed':False,'status':'TECHNICAL_INCONCLUSIVE','exception_type':type(error).__name__,'code':str(error)[:160],'no_retry':True}
        publish('TECHNICAL_FAULT.json',result);raise
def main():
    p=argparse.ArgumentParser();p.add_argument('--release');p.add_argument('--release-sha256');a=p.parse_args()
    if not a.release or not a.release_sha256:
        print(jb({'status':'DISABLED_NO_SCORE_RELEASE','model_calls':0,'fits':0}).decode());return 2
    timer=threading.Timer(10,lambda:os._exit(124));timer.daemon=True;timer.start()
    try:
        r=run(a.release,a.release_sha256);print(jb({'completed':r['completed'],'branch':r['branch'],'score_calls':r['score_calls']}).decode());return 0
    finally:timer.cancel()
if __name__=='__main__':raise SystemExit(main())

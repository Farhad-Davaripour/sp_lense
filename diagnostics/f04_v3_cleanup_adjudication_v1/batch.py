"""One source-locked saved-evidence batch. Stdlib only; no process fixture/model."""
import ast,hashlib,json,math,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def need(ok,message):
    if not ok:raise ValueError(message)
def write(name,value,raw=False):
    data=value if raw else (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    need(len(data)<=2*1024**2 and sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())+len(data)<=8*1024**2,'artifact caps')
    with (HERE/name).open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args])
def check_freeze():
    lock=read(HERE/'freeze.json')
    for n,h in lock['files'].items():need(sha((HERE/n).read_bytes())==h,'adjudication source changed '+n)
    return lock

def load_saved_once():
    m=read(HERE/'source_manifest.json');prefix=m['parent_namespace']+'/'
    inventory=git('show',m['result_commit']+':'+prefix+'FINAL_INVENTORY.json')
    need(sha(inventory)==m['inventory_sha256'],'authenticated immutable V3 result inventory')
    rows={v['path']:v for v in json.loads(inventory)['files']};raws={}
    for row in m['files']:
        need(rows[row['path']]==row,'exact predeclared artifact reference')
        raw=git('show',m['result_commit']+':'+prefix+row['path'])
        need(sha(raw)==row['sha256'] and len(raw)==row['bytes'],'committed raw artifact '+row['path']);raws[row['path']]=raw
    source_freeze=git('show',m['source_commit']+':'+prefix+'freeze.json')
    need(source_freeze==raws['freeze.json'],'source commit contains identical prospective V3 freeze')
    lock=json.loads(source_freeze)
    for name in ('owned.py','native.py','owned_capture.py','entry.py','source_bindings.json','input_lock.json'):
        need(sha(raws[name])==lock['source_sha256'][name],'production/identity/input source bound before old execution')
    shared=m['shared_source'];source=git('show',m['result_commit']+':'+shared['path'])
    need(sha(source)==shared['sha256'] and len(source)==shared['bytes'] and sha((ROOT/shared['path']).read_bytes())==shared['sha256'],'unchanged shared capture source')
    text=source.decode().replace('\r\n','\n');loop=text.index('        while True:\n            if stop.is_set():')
    indexes=[text.index(s,loop) for s in ('            code = process.poll()','fail("CAPTURE_WORKER_NONZERO_EXIT")','fail("CAPTURE_DEADLINE")')]
    need(indexes==sorted(indexes),'source completed-nonzero branch precedes separate deadline branch')
    names={'run':'RUN_STARTED.json','bootstrap':'BOOTSTRAP.json','identity':'process_identity.json','ownership':'ownership_final.json','capture':'capture.json','supervisor':'supervisor_final.json'}
    r={k:json.loads(raws['synthetic/deadline/'+v]) for k,v in names.items()}
    r['events']=[json.loads(line) for line in raws['synthetic/deadline/ownership_events.jsonl'].splitlines()]
    r['expected_images']=json.loads(raws['source_bindings.json'])['owned_identity']
    parent=ROOT/m['parent_namespace']
    r['expected_command']=[r['expected_images']['launch_image'],'-B',str(parent/'entry.py'),'fake_deadline',str(parent/'synthetic/deadline')]
    r['log_sha256']=sha(raws['synthetic/deadline/worker.log']);r['log_bytes']=len(raws['synthetic/deadline/worker.log'])
    receipt=json.loads(raws['cleanup_batch_receipt.json']);assertions=json.loads(raws['deadline_assertions.json'])
    need(receipt['status']=='INCONCLUSIVE_FIXTURE','preserve original frozen batch status')
    return r,{'source_commit':m['source_commit'],'result_commit':m['result_commit'],'inventory_sha256':m['inventory_sha256'],
        'authenticated_artifacts':len(raws),'shared_source_sha256':shared['sha256'],'source_branch_order_verified':True,
        'original_frozen_receipt':receipt,'original_frozen_live_assertions_true':sum(assertions['assertions'].values()),
        'original_frozen_live_assertions_total':len(assertions['assertions']),'frozen_trial_not_reclassified':True}

def main():
    if sys.argv[1:]==['freeze']:
        for p in HERE.glob('*.py'):ast.parse(p.read_bytes(),filename=p.name)
        names=sorted(p.name for p in HERE.iterdir() if p.is_file())
        write('freeze.json',{'files':{n:sha((HERE/n).read_bytes()) for n in names},'five_pure_cases_then_one_saved_read':True,
            'finding_only':'verified_owned_deadline_cleanup','original_trial_unchanged':True,'no_live_execution':True})
        print(json.dumps({'status':'PROSPECTIVE_RULE_FROZEN','sha256':sha((HERE/'freeze.json').read_bytes())}));return 0
    need(sys.argv[1:]==['run'],'fixed command');started=time.monotonic();check_freeze()
    usage=read(HERE/'usage.json');payload=json.loads(usage['tool_receipt']['content'][0]['text']);standard=payload['rateLimitsByLimitId']['codex']
    values=[standard[k]['usedPercent'] for k in ('primary','secondary') if standard.get(k) is not None]
    need(values and all(type(v) in (int,float) and math.isfinite(v) and 0<=v<100 for v in values)
         and max(values)==usage['known_standard_used_percent'] and 0<=time.time()-usage['captured_at_unix']<=120,'fresh known standard usage')
    write('BATCH_STARTED.json',{'monotonic':started,'source_commit':git('rev-parse','HEAD').decode().strip(),'freeze_sha256':sha((HERE/'freeze.json').read_bytes())})
    result={'status':'INCONCLUSIVE_ENGINEERING_CHECK','model_calls':0,'tokenizer_calls':0,'gate_scores':0,'child_fixtures':0,'prior_suites_rerun':False}
    try:
        from tests import run
        from rule import adjudicate
        result['pure_tests']=run()
        records,authentication=load_saved_once();result['authentication']=authentication
        finding=adjudicate(records);result['finding']=finding
        need(finding['verified_owned_deadline_cleanup'],'saved chain does not qualify: '+str(finding))
        result['status']='PASS_POST_HOC_ENGINEERING_FINDING_ONLY'
        m=read(HERE/'source_manifest.json')
        result['unchanged_future_release_bindings']={'source_commit':m['source_commit'],'freeze_sha256':sha(git('show',m['source_commit']+':'+m['parent_namespace']+'/freeze.json')),
            'input_lock_sha256':m['input_lock_sha256'],'root_scope':'ONE_F04_SELF_ASSAY','production_authorized':False,
            'limits':{'forwards':42,'derivatives':16,'loads':1,'worker_seconds':300,'cleanup_seconds':15,'audit_seconds':90,'bytes':100663296},
            'release_schema':m['parent_namespace']+'/RELEASE_SCHEMA.md','command':'.venv/Scripts/python.exe -B '+m['parent_namespace']+'/run.py run',
            'exclusive_attempt_path':m['parent_namespace']+'/real_attempt',
            'remaining_admission_gap':'Root must independently authorize and commit the exact release/pin with fresh known standard usage below100; this task grants none.'}
    except BaseException as error:result['error']=type(error).__name__+': '+str(error)
    finally:
        result['elapsed_seconds']=time.monotonic()-started
        if result['elapsed_seconds']>60:result['status']='INCONCLUSIVE_ENGINEERING_CHECK';result['time_limit_fault']=True
        write('results.json',result)
    print(json.dumps({'status':result['status'],'pure_cases':len(result.get('pure_tests',[])),'finding':result.get('finding'),'elapsed_seconds':result['elapsed_seconds']}))
    return 0 if result['status']=='PASS_POST_HOC_ENGINEERING_FINDING_ONLY' else 1
if __name__=='__main__':raise SystemExit(main())

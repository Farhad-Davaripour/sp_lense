"""Owned synthetic model-boundary capture and owned real optimizer on fake rows.

No release/admission/lifecycle/certification function is replaced. Capture loader
is the single explicit synthetic model-boundary replacement. Other fixture
changes are namespace/path/hash constants with function/class AST equality.
"""
import ast,hashlib,json,os,re,subprocess,sys,tempfile,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def sha(raw):return hashlib.sha256(raw).hexdigest()
def jb(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def put(p,raw):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:f.write(raw)
def functions(text):return [ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
def command(args,cwd,limit):
    started=time.monotonic();r=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'-E','-S','-B',*map(str,args)],cwd=cwd,capture_output=True,timeout=limit)
    assert r.returncode==0,(r.returncode,r.stdout[-4000:],r.stderr[-4000:])
    return {'seconds':time.monotonic()-started,'stdout':r.stdout.decode(),'stderr':r.stderr.decode(),'exit_code':r.returncode}
def main():
    started=time.monotonic();base=Path(tempfile.mkdtemp(prefix='own_',dir=HERE/'test_evidence'))
    cap=base/'cap';cap.mkdir();source=HERE/'train_capture';deltas=[];source_hashes={}
    for p in source.iterdir():
        if p.is_file() and not p.name.startswith('test_') and p.name not in ('SOURCE_FREEZE.json','RELEASE_DRAFT.json'):
            raw=p.read_bytes();source_hashes[p.name]=sha(raw)
            if p.name=='loader.py':raw=(HERE/'synthetic_model_boundary.py').read_bytes();deltas.append('loader.py: EXPLICIT SYNTHETIC MODEL BOUNDARY')
            elif p.suffix=='.py':
                text=raw.decode();new=text.replace('ROOT=HERE.parents[2]',f'ROOT=Path({str(ROOT)!r})').replace('ROOT=Path(__file__).resolve().parents[3]',f'ROOT=Path({str(ROOT)!r})').replace("ATTEMPT='prechoice_train_capture_attempt_001'","ATTEMPT='t'")
                assert functions(text)==functions(new),p.name
                if new!=text:deltas.append(p.name+': ROOT/ATTEMPT constants')
                raw=new.encode()
            put(cap/p.name,raw)
    freeze={'source_sha256':{p.name:sha(p.read_bytes()) for p in cap.iterdir() if p.is_file()},'external_sources':[
        {'path':'development/native_gate_frozen_transfer_v1/capture/receiver.py','sha256':'75680a94ae50ca6ea53765fb762061f36a3bc4a394fbf423de4f2f050683c152'}]}
    put(cap/'SOURCE_FREEZE.json',jb(freeze));sys.path.insert(0,str(cap));import input_reader,support,authority
    inputs=input_reader.build_inputs();release={'schema':'prechoice_train_capture_release.v1','approved':True,'role':'TRAIN_PRECHOICE','attempt':'t',
        'limits':authority.LIMITS,'output':str(support.output()),'reserved_bytes':67538944,'source_freeze_sha256':sha((cap/'SOURCE_FREEZE.json').read_bytes()),
        'checkpoint_lock_sha256':sha((cap/'CHECKPOINT.json').read_bytes()),'owned_identity_sha256':sha((cap/'OWNED_IDENTITY.json').read_bytes()),
        'input_data_lock_sha256':sha((cap/'DATA_LOCK.json').read_bytes()),'inputs_sha256':sha(support.json_bytes(inputs)),
        'certificate_sha256':inputs['certificate_sha256'],'trace_sources':{n:h for n,h in freeze['source_sha256'].items() if n.endswith('.py')}}
    raw=jb(release);put(cap/'root_release/RELEASE.json',raw);approved=sha(raw)
    preflight=command([cap/'launch.py','--approved-release-sha256',approved,'--preflight'],cap,30)
    capture=command([cap/'launch.py','--approved-release-sha256',approved],cap,435)
    parent=json.loads((support.output()/'PARENT_FINAL.json').read_bytes());assert parent['scientific_pass'] and parent['worker_quiescent'] and parent['audit_quiescent'],parent
    audit=json.loads((support.output()/'AUDIT_RESULT.json').read_bytes());assert audit['completed_forwards']==58 and audit['derivatives']==0
    fit=base/'fit';fit.mkdir()
    for name in ('gate.py','checker.py','source_auth.py','construction.py','fit_owner.py'):
        raw=(HERE/'fit'/name).read_bytes();text=raw.decode();new=text.replace('ROOT=HERE.parents[2]',f'ROOT=Path({str(ROOT)!r})')
        new=new.replace("CAPTURE=HERE.parent/'train_capture'",f'CAPTURE=Path({str(cap)!r})')
        if name=='source_auth.py':new=re.sub(r"CAPTURE_SOURCE_SHA='[0-9a-f]{64}'",f"CAPTURE_SOURCE_SHA='{release['source_freeze_sha256']}'",new)
        new=new.replace("TRAINING_SOURCE=HERE.parent/'train_capture/DATA_LOCK.json'",f'TRAINING_SOURCE=Path({str(cap/"DATA_LOCK.json")!r})')
        new=new.replace('HELPER = HERE.parents[2] / "development/native_oracle_confirmation_preparation_owner_v1/windows_job.py"',f'HELPER = Path({str(ROOT/"development/native_oracle_confirmation_preparation_owner_v1/windows_job.py")!r})')
        assert functions(text)==functions(new),name
        put(fit/name,new.encode())
    candidate=command([fit/'construction.py','candidate'],fit,30)
    r=json.loads((fit/'RELEASE_DRAFT.json').read_bytes());r['approved']=True;raw=jb(r);put(fit/'SYNTHETIC_RELEASE.json',raw)
    owner=command([fit/'fit_owner.py','--root-approved','--owner-sha256',sha((fit/'fit_owner.py').read_bytes()),
        '--helper-sha256','a6334fea1678773ff169aed39c3b1939c57614d0096a446e6d0418abcb21368a',
        '--release',fit/'SYNTHETIC_RELEASE.json','--release-sha256',sha(raw)],fit,65)
    result=json.loads((fit/'construction_attempt_001/RESULT.json').read_bytes())
    assert result['scientific_pass'] and result['fits_attempted']==3 and result['stages'][0]['training_correct']==29,result
    assert json.loads(owner['stdout'])['technical_complete'] is True
    receipt={'status':'PASS_OWNED_SYNTHETIC_CAPTURE_AND_THREE_HEAD_FIT','fixture':str(base),'capture_namespace':str(cap),'fit_namespace':str(fit),
        'production_source_sha256':source_hashes,'substitutions':deltas,'capture':capture,'preflight':preflight,'candidate':candidate,'fit_owner':owner,
        'fake_views':58,'fake_pairs':29,'synthetic_optimizer_solves':3,'training_correct':29,'actual_model_calls':0,'actual_tokenizer_calls':0,
        'real_feature_reads':0,'actual_scientific_release':False,'elapsed_seconds':time.monotonic()-started,
        'result_sha256':sha((fit/'construction_attempt_001/RESULT.json').read_bytes()),'artifact_sha256':result['artifacts']['PRECHOICE29']}
    put(base/'OWNED_TEST_RESULT.json',jb(receipt));print(json.dumps({'status':receipt['status'],'fixture':str(base),'elapsed_seconds':receipt['elapsed_seconds'],'artifact_sha256':receipt['artifact_sha256']}))
if __name__=='__main__':main()

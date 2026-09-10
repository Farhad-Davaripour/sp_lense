"""Certified synthetic artifact guard through actual 16-view retained lifecycle."""
import hashlib,importlib,json,sys,time
from pathlib import Path
from test_owned_pipeline import HERE,ROOT,sha,jb,put,functions,command
def main():
    base=Path(sys.argv[1]).resolve();receipt=json.loads((base/'OWNED_TEST_RESULT.json').read_bytes())
    assert receipt['actual_scientific_release'] is False and receipt['synthetic_optimizer_solves']==3
    fit=base/'fit';sys.path.insert(0,str(fit));import construction,gate,checker,source_auth
    result=json.loads((fit/'construction_attempt_001/RESULT.json').read_bytes());input_contract=gate.decode((fit/'TRAINING_MANIFEST.json').read_bytes())
    rows,labels=source_auth.load_saved(deadline=time.monotonic()+30,expected_binding=input_contract['capture_binding'])
    mu,x=gate.preprocess(rows);stage=result['stages'][0];assert list(mu)==stage['training_mean']
    for head in stage['heads']:
        indices=construction.head_indices(list(range(29)),head['id']);p=head['parameters']
        checker.verify([x[i] for i in indices],[labels[i] for i in indices],p['w'],p['b'],p['alpha'])
    ar=(fit/'construction_attempt_001/PRECHOICE29_GATE.json').read_bytes();artifact=gate.decode(ar)
    model=construction.load_artifact(ar,expected_sha256=sha(ar),expected_bindings=artifact['bindings'])
    assert sum((model.score(row)>0)==(y==1) for row,y in zip(rows,labels))==29
    release_raw=(fit/'SYNTHETIC_RELEASE.json').read_bytes();put(fit/'RELEASE.json',release_raw)
    release=json.loads(release_raw)
    review={'schema':'prechoice_fit_independent_review.v1','status':'PASS','artifact_sha256':sha(ar),
        'result_sha256':sha((fit/'construction_attempt_001/RESULT.json').read_bytes()),'source_sha256':release['source_sha256'],
        'release_sha256':sha(release_raw),'all_three_certificates_verified':True,'all29_training_correct':True,
        'exact_artifact_reload':True,'test_only_saved_math_check':True}
    put(fit/'INDEPENDENT_ACTUAL_FIT_REVIEW.json',jb(review))
    freeze={**{k:review[k] for k in ('artifact_sha256','result_sha256','source_sha256','release_sha256')},
        'schema':'prechoice29_accepted_artifact.v1','approved':True,'arm':'PRECHOICE29','diagnostic_inputs_used':False,
        'owner_terminal_sha256':sha((fit/'fit_owner_attempt_001/TERMINAL.json').read_bytes()),
        'owner_finalization_sha256':sha((fit/'fit_owner_attempt_001/FINALIZATION.json').read_bytes()),
        'independent_review_sha256':sha((fit/'INDEPENDENT_ACTUAL_FIT_REVIEW.json').read_bytes()),'artifact_bindings':artifact['bindings']}
    put(fit/'FROZEN_PRECHOICE_ARTIFACT.json',jb(freeze));fit_sha=sha(jb(freeze))
    put(base/'diagnostic_gate.py',(HERE/'diagnostic_gate.py').read_bytes())
    for name in ('support','input_reader','authority','index_contract','gate','checker','source_auth','construction'):
        sys.modules.pop(name,None)
    sys.path[:]=[p for p in sys.path if p!=str(fit)]
    cap=base/'diag';cap.mkdir();source=HERE/'diagnostic_capture';deltas=[]
    for p in source.iterdir():
        if p.is_file() and p.name not in ('SOURCE_FREEZE.json','RELEASE_DRAFT.json') and not p.name.startswith('test_'):
            raw=p.read_bytes()
            if p.name=='loader.py':raw=(HERE/'synthetic_model_boundary.py').read_bytes();deltas.append('loader: synthetic model boundary')
            elif p.suffix=='.py':
                text=raw.decode();new=text.replace('ROOT=HERE.parents[2]',f'ROOT=Path({str(ROOT)!r})').replace('ROOT=Path(__file__).resolve().parents[3]',f'ROOT=Path({str(ROOT)!r})').replace("ATTEMPT='prechoice_diagnostic_capture_attempt_001'","ATTEMPT='dt'")
                assert functions(text)==functions(new),p.name
                if new!=text:deltas.append(p.name+': ROOT/ATTEMPT constants')
                raw=new.encode()
            put(cap/p.name,raw)
    source_freeze={'source_sha256':{p.name:sha(p.read_bytes()) for p in cap.iterdir() if p.is_file()},'external_sources':[
        {'path':str(base/'diagnostic_gate.py'),'sha256':sha((base/'diagnostic_gate.py').read_bytes())},
        {'path':'development/native_gate_frozen_transfer_v1/capture/receiver.py','sha256':'75680a94ae50ca6ea53765fb762061f36a3bc4a394fbf423de4f2f050683c152'}]}
    put(cap/'SOURCE_FREEZE.json',jb(source_freeze));sys.path.insert(0,str(cap));import input_reader,support,authority
    inputs=input_reader.build_inputs();assert len(inputs['cases'])==16
    r={'schema':'prechoice_diagnostic_capture_release.v1','approved':True,'role':'DIAGNOSTIC_PRECHOICE','attempt':'dt',
        'limits':authority.LIMITS,'output':str(support.output()),'reserved_bytes':19628032,'source_freeze_sha256':sha((cap/'SOURCE_FREEZE.json').read_bytes()),
        'checkpoint_lock_sha256':sha((cap/'CHECKPOINT.json').read_bytes()),'owned_identity_sha256':sha((cap/'OWNED_IDENTITY.json').read_bytes()),
        'input_data_lock_sha256':sha((cap/'DATA_LOCK.json').read_bytes()),'inputs_sha256':sha(support.json_bytes(inputs)),
        'certificate_sha256':inputs['certificate_sha256'],'fit_freeze_sha256':fit_sha,
        'trace_sources':{n:h for n,h in source_freeze['source_sha256'].items() if n.endswith('.py')}}
    raw=jb(r);put(cap/'root_release/RELEASE.json',raw);approved=sha(raw)
    assert authority.read_release(approved)==r
    negatives=[]
    for key,value in (('fit_freeze_sha256','0'*64),('inputs_sha256','0'*64),('approved',False)):
        bad={**r,key:value};(cap/'root_release/RELEASE.json').write_bytes(jb(bad))
        try:authority.read_release(sha(jb(bad)))
        except (ValueError,KeyError):negatives.append(key)
        else:raise AssertionError('bad release admitted '+key)
    (cap/'root_release/RELEASE.json').write_bytes(raw)
    finalpath=fit/'fit_owner_attempt_001/FINALIZATION.json';original=finalpath.read_bytes();badfinal=json.loads(original);badfinal['terminal_sha256']='0'*64
    badfreeze={**freeze,'owner_finalization_sha256':sha(jb(badfinal))};finalpath.write_bytes(jb(badfinal));(fit/'FROZEN_PRECHOICE_ARTIFACT.json').write_bytes(jb(badfreeze))
    rr={**r,'fit_freeze_sha256':sha(jb(badfreeze))};(cap/'root_release/RELEASE.json').write_bytes(jb(rr))
    try:authority.read_release(sha(jb(rr)))
    except ValueError as e:assert str(e)=='CLOSED_RETAINED_FIT_OWNER';negatives.append('coherent_terminal_join')
    else:raise AssertionError('false terminal join admitted')
    finalpath.write_bytes(original);(fit/'FROZEN_PRECHOICE_ARTIFACT.json').write_bytes(jb(freeze));(cap/'root_release/RELEASE.json').write_bytes(raw)
    preflight=command([cap/'launch.py','--approved-release-sha256',approved,'--preflight'],cap,30)
    launch=command([cap/'launch.py','--approved-release-sha256',approved],cap,435)
    parent=json.loads((support.output()/'PARENT_FINAL.json').read_bytes());audit=json.loads((support.output()/'AUDIT_RESULT.json').read_bytes())
    assert parent['scientific_pass'] and parent['worker_quiescent'] and parent['audit_quiescent'] and audit['completed_forwards']==16
    receipt={'status':'PASS_WIRED_GUARD_AND_OWNED16_SYNTHETIC_CAPTURE','fixture':str(base),'fit_freeze_sha256':fit_sha,
        'negatives':negatives,'substitutions':deltas,'preflight':preflight,'launch':launch,'actual_models':0,'actual_tokenizers':0,
        'actual_features':False,'new_optimizer_calls':0,'synthetic_certificates_replayed':3,'views':16,
        'diagnostic_guard_sha256':sha((base/'diagnostic_gate.py').read_bytes())}
    put(base/'DIAGNOSTIC_OWNED_TEST_RESULT.json',jb(receipt));print(json.dumps({'status':receipt['status'],'fixture':str(base),'views':16,'guard_sha256':receipt['diagnostic_guard_sha256']}))
if __name__=='__main__':main()

"""Saved-only final failure verification. No optimizer, model, or later fold runs.
The authenticated 44-row matrix is read solely to reconstruct the two saved folds.
"""
import hashlib,json,math,os,subprocess,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
FIT=ROOT/'development/native_gate_hardmargin_coverage_v1';ATTEMPT=FIT/'construction_attempt_001';OWNER=FIT/'fit_owner_attempt_001'
ARCHIVE='75df0d868f8dc6cf726dfe5ba7f96eb75bb5bf02'
sha=lambda raw:hashlib.sha256(raw).hexdigest()
read=lambda path:json.loads(path.read_bytes())
dot=lambda a,b:math.fsum(v*w for v,w in zip(a,b,strict=True))

def main():
    start=time.monotonic();deadline=start+75
    receipt={'status':'FAIL','scope':'SAVED_ONLY_FINAL_SCIENTIFIC_FAILURE',
        'command':'.venv\\Scripts\\python.exe -E -S -B development/native_supervised_gate_coverage_v1/verify_saved_fit.py',
        'exceptions':[],'optimizer_calls':0,'model_calls':0,'tokenizer_calls':0,'later_fold_runs':0}
    try:
        os.chdir(FIT);sys.path.insert(0,str(FIT))
        import gate,checker,source_auth,construction
        def no_optimizer(frame,event,arg):
            if event=='call' and frame.f_code.co_filename==str(FIT/'gate.py') and frame.f_code.co_name in ('solve','numpy_runtime'):
                raise AssertionError('OPTIMIZER_FORBIDDEN_IN_SAVED_REVIEW')
        sys.setprofile(no_optimizer)
        freeze_raw=(FIT/'SOURCE_FREEZE.json').read_bytes()
        assert sha(freeze_raw)=='4211d673a15436f5002fa4dfb6b4828994b31ca40f9ca9c9fa73512f42ac1425'
        freeze=json.loads(freeze_raw)
        for name,digest in freeze['source_sha256'].items():assert sha((FIT/name).read_bytes())==digest
        for pin in freeze['external_sources']:assert sha(Path(pin['path']).read_bytes())==pin['sha256']
        release_raw=(FIT/'ROOT_RELEASE.json').read_bytes();release_sha=sha(release_raw)
        assert release_sha=='1ee8001866ccccb3e0b27da739b8f94b8ccca6caf6eadbc615f7ee689f466e14'
        release=construction.authorize(release_raw,release_sha)
        manifest=source_auth.build_manifest(deadline=deadline)
        rows,labels=source_auth.extract_features(manifest,deadline=deadline)
        assert len(rows)==len(labels)==44 and all(len(r)==1024 for r in rows)
        raw=(ATTEMPT/'RESULT.json').read_bytes();result=json.loads(raw)
        assert sha(raw)=='e1827795212abe104c8fcfc663c7fc0b68bef10e10c766653dc140829fbdc38a'
        assert result['status']=='SCIENTIFIC_HELD_FAMILY_FAIL' and result['scientific_pass'] is False and result['fits_attempted']==2
        assert all(result[k]==0 for k in ('model_calls','tokenizer_calls','checkpoint_tensor_reads'))
        schedule=construction.schedule();stages=result['stages'];assert len(stages)==7
        checks=[]
        for n,(stage,fixed) in enumerate(zip(stages,schedule,strict=True)):
            assert all(stage[k]==fixed[k] for k in ('id','train','held'))
            if n>=2:
                assert stage=={**fixed,'status':'UNRUN'};continue
            train=[rows[i] for i in stage['train']];y=[labels[i] for i in stage['train']]
            assert len(train)==38 and len(stage['held'])==6 and sum(v==1 for v in y)==10
            mean=[math.fsum(r[j] for r in train)/38 for j in range(1024)]
            assert mean==stage['training_mean']
            def transform(row):
                delta=[float(v)-m for v,m in zip(row,mean,strict=True)];norm=math.sqrt(dot(delta,delta))
                assert norm>0 and math.isfinite(norm);return [v/norm for v in delta]
            x=[transform(row) for row in train];p=stage['parameters']
            cert=checker.verify(x,y,p['w'],p['b'],p['alpha'])
            assert cert==stage['certificate'] and max(cert['checks'].values())<=1e-8
            for role,indices in (('training',stage['train']),('held',stage['held'])):
                scores=[dot(p['w'],transform(rows[i]))+p['b'] for i in indices]
                expected=[labels[i] for i in indices]
                assert scores==stage[role+'_scores']
                assert [v*s for v,s in zip(expected,scores,strict=True)]==stage[role+'_margins']
                confusion={k:0 for k in ('TP','FN','TN','FP')}
                for score,label in zip(scores,expected,strict=True):confusion[('TP' if label==1 else 'FP') if score>0 else ('FN' if label==1 else 'TN')]+=1
                assert confusion==stage[role+'_confusion'] and confusion['TP']+confusion['TN']==stage[role+'_correct']
            assert stage['training_correct']==38 and stage['held_correct']==(6 if n==0 else 2)
            assert stage['status']==('PASS' if n==0 else 'HELD_FAMILY_FAIL')
            assert read(ATTEMPT/('FIT_ATTEMPT_'+str(n+1)+'.json'))=={'attempt':n+1,'maximum':7,'stage':stage['id']}
            checks.append({'id':stage['id'],'training_correct':38,'held_correct':stage['held_correct'],
                'held_confusion':stage['held_confusion'],'iterations':stage['iterations'],
                'mean_max_error':0,'score_max_error':0,'certificate_exact':True,'maximum_kkt_check':max(cert['checks'].values())})
        assert set(p.name for p in ATTEMPT.iterdir())=={'FIT_ATTEMPT_1.json','FIT_ATTEMPT_2.json','RESULT.json'}
        assert not (ATTEMPT/'FITTED_GATE.json').exists()
        terminal_raw=(OWNER/'TERMINAL.json').read_bytes();terminal=json.loads(terminal_raw);final=read(OWNER/'FINALIZATION.json')
        assert final['terminal_sha256']==sha(terminal_raw) and not final['deadline_fault'] and not final['storage_fault']
        assert terminal['errors']==[] and terminal['exit_code']==0 and not terminal['timed_out'] and not terminal['output_limit']
        for key in ('assigned_before_resume','job_closed','job_empty','process_handle_closed','process_technical_complete','readers_closed','cleanup_within_budget','scientific_result_preserved'):assert terminal[key] is True
        assert terminal['owner_sha256']==sha((FIT/'fit_owner.py').read_bytes())
        assert terminal['command'][-1]==release_sha and Path(terminal['command'][-3])==FIT/'ROOT_RELEASE.json'
        assert final['elapsed_seconds_after_terminal']<=65 and final['cleanup_seconds_after_terminal']<=5
        assert read(OWNER/'stdout.log')==construction.output_summary(result) and (OWNER/'stderr.log').stat().st_size==0
        assert sum(p.stat().st_size for p in ATTEMPT.iterdir())<=1048576
        paths=subprocess.check_output(['git','diff-tree','--no-commit-id','--name-only','-r',ARCHIVE],cwd=ROOT).decode().splitlines()
        assert len(paths)==7 and all(p.startswith(('development/native_gate_hardmargin_coverage_v1/construction_attempt_001/','development/native_gate_hardmargin_coverage_v1/fit_owner_attempt_001/')) for p in paths)
        stream=subprocess.run(['git','cat-file','--batch'],input=''.join(ARCHIVE+':'+p+'\n' for p in paths).encode(),stdout=subprocess.PIPE,check=True,cwd=ROOT).stdout
        cursor=total=0;blobs={}
        for path in paths:
            end=stream.index(b'\n',cursor);oid,kind,size=stream[cursor:end].split();size=int(size);assert kind==b'blob'
            value=stream[end+1:end+1+size];cursor=end+1+size+1;assert value==(ROOT/path).read_bytes();total+=size;blobs[path]=sha(value)
        assert cursor==len(stream) and total==94139
        forbidden={'torch','transformers','tokenizers','safetensors','numpy'}&{n.split('.')[0] for n in sys.modules};assert not forbidden
        receipt.update(status='PASS_VALIDLY_CLOSED_SCIENTIFIC_FAILURE',scientific_result=result['status'],fits_attempted=2,
            fold_reconstructions=checks,unrun=['G03','G04','G05','G06','FULL44'],final_artifact_absent=True,
            actual44_coordinates_authenticated=True,all_transforms_means_scores_margins_exact=True,
            result_sha256=sha(raw),release_sha256=release_sha,source_freeze_sha256=sha(freeze_raw),
            training_manifest_sha256=release['training_manifest_sha256'],core_source_lock_sha256=release['source_sha256'],
            construction_lock_sha256=release['construction_lock_sha256'],
            combined_manifest_sha256=gate.digest(gate.canonical(manifest)),combined_feature_hashes_sha256=manifest['combined_feature_sha256'],
            checker_sha256=sha((FIT/'checker.py').read_bytes()),gate_sha256=sha((FIT/'gate.py').read_bytes()),
            terminal_sha256=sha(terminal_raw),finalization_sha256=sha((OWNER/'FINALIZATION.json').read_bytes()),
            owner_seconds_after_terminal=final['elapsed_seconds_after_terminal'],closure_and_compact_stdout='PASS',
            archive_commit=ARCHIVE,raw_git_files=7,raw_git_bytes=total,archive_blob_sha256=blobs,
            forbidden_imports=[],successor_authorized=False)
    except Exception as error:receipt['exceptions'].append({'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
    finally:sys.setprofile(None)
    receipt['elapsed_seconds']=round(time.monotonic()-start,3);receipt['completed_utc']=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())
    receipt['verifier_sha256']=sha(Path(__file__).read_bytes())
    with (HERE/'ACTUAL_FIT_REVIEW.json').open('x',encoding='utf-8') as file:json.dump(receipt,file,sort_keys=True,indent=2);file.write('\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='archive_blob_sha256'},sort_keys=True))
    return 0 if receipt['status'].startswith('PASS') else 1

if __name__=='__main__':raise SystemExit(main())

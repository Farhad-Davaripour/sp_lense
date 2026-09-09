"""Saved capture/metadata-only compatibility review. Never extracts fit matrices.
The actual audit reads saved new rows/logits only for its frozen integrity checks;
old-coordinate validation is reused through its authenticated immutable manifest.
"""
import hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
CAP=ROOT/'development/native_supervised_gate_capture_coverage_v1'
FIT=ROOT/'development/native_gate_hardmargin_coverage_v1'
BASE=CAP/'real_evidence/native_supervised_gate_capture_coverage_attempt_001'
ARCHIVE='3eee781352ea25e5e68017cd52832346c556d96b'
RELEASE='37fbb4bd116e119a59fbf2d88507cecae5d6804cf3a6dd8badd9a863ba3ac950'
sha=lambda raw:hashlib.sha256(raw).hexdigest()
read=lambda path:json.loads(path.read_bytes())

def main():
    start=time.monotonic();deadline=start+80
    receipt={'status':'FAIL','scope':'SAVED_CAPTURE_AND_METADATA_ONLY_OLD32_NEW12_COMPATIBILITY',
        'command':'.venv\\Scripts\\python.exe -E -S -B development/native_supervised_gate_coverage_v1/verify_saved_capture.py',
        'exceptions':[],'model_calls':0,'tokenizer_calls':0,'fit_calls':0,'extract_features_calls':0,'load_saved_calls':0}
    try:
        os.chdir(CAP);sys.path.insert(0,str(CAP));os.environ['SP_NATIVE_RELEASE_SHA']=RELEASE
        import authority,input_reader,audit_saved,support
        release=authority.read_release(RELEASE);execution=authority.execution(release,RELEASE)
        saved_raw=(BASE/'AUDIT_RESULT.json').read_bytes()
        assert sha(saved_raw)=='24beec54d783f3d3abd7942f0fef3c87e38c7691d21dae3b586e284140251dae'
        reproduced=audit_saved.judge(BASE,execution,deadline)
        assert support.json_bytes(reproduced)==saved_raw
        parent_raw=(BASE/'PARENT_FINAL.json').read_bytes();parent=json.loads(parent_raw)
        assert sha(parent_raw)=='8db47ed9858439806e6b0038039d75c9bc5eba577f27e5d148ea87b36b7a69b7'
        assert parent['execution']==execution and parent['errors']==[]
        assert parent['worker_quiescent'] and parent['audit_quiescent'] and parent['audit_completed'] and parent['scientific_pass']
        assert parent['shared_cleanup']['used_seconds']<=parent['shared_cleanup']['maximum_seconds']==15
        owners={}
        for lane in ('production_worker','production_audit'):
            own=read(BASE/'owned'/lane/'CAPTURE.json');assert own['execution']==execution
            for key in ('actual_handle_closed','launcher_original_handle_closed','binding_authenticated','quiescent','pipes_closed','threads_joined','within_absolute_cleanup_deadline'):assert own[key] is True
            for key in ('faults','cleanup_faults','stdout_capture_errors'):assert own[key]==[]
            assert own['primary_error'] is None and own['stop_reason'] is None
            assert set(own['exit_proofs'])=={'actual_worker','launcher'}
            assert all(p['exit_code']==0 and p['valid_retained_handle'] and p['signaled'] and p['query_success'] for p in own['exit_proofs'].values())
            assert own['finished_monotonic']<=own['deadline_monotonic'];owners[lane]=own['elapsed_seconds']
        worker=read(BASE/'WORKER_RESULT.json')
        assert worker['counts']['attempts']=={'load':1,'forward':12,'derivative':0} and worker['counts']['encoding']==0
        assert len(worker['cells'])==12 and all(c['status']=='COMPLETE' for c in worker['cells'])
        fit_freeze_raw=(FIT/'SOURCE_FREEZE.json').read_bytes()
        assert sha(fit_freeze_raw)=='4211d673a15436f5002fa4dfb6b4828994b31ca40f9ca9c9fa73512f42ac1425'
        freeze=json.loads(fit_freeze_raw)
        for name,digest in freeze['source_sha256'].items():assert sha((FIT/name).read_bytes())==digest
        for pin in freeze['external_sources']:assert sha(Path(pin['path']).read_bytes())==pin['sha256']
        sys.path.insert(0,str(FIT))
        import gate,source_auth
        before_path=list(sys.path);before_modules={n:sys.modules.get(n) for n in source_auth.CHAIN_NAMES}
        manifest=source_auth.build_manifest(deadline=deadline)
        assert sys.path==before_path and all(sys.modules.get(n) is value for n,value in before_modules.items())
        assert len(manifest['old']['selection'])==32 and len(manifest['new']['selection'])==12
        assert tuple(s['case'] for s in manifest['selection'])==source_auth.keys()
        assert tuple(s['label'] for s in manifest['selection'])==source_auth.expected_labels()
        assert len({s['input_ids_sha256'] for s in manifest['selection']})==44
        assert manifest['new']['capture_audit_sha256']==sha(saved_raw)
        paths=subprocess.check_output(['git','diff-tree','--no-commit-id','--name-only','-r',ARCHIVE],cwd=ROOT).decode().splitlines()
        assert len(paths)==67 and all(p.startswith('development/native_supervised_gate_capture_coverage_v1/real_evidence/native_supervised_gate_capture_coverage_attempt_001/') for p in paths)
        stream=subprocess.run(['git','cat-file','--batch'],input=''.join(ARCHIVE+':'+p+'\n' for p in paths).encode(),stdout=subprocess.PIPE,check=True,cwd=ROOT).stdout
        cursor=total=0;blobs={}
        for path in paths:
            end=stream.index(b'\n',cursor);oid,kind,size=stream[cursor:end].split();size=int(size);assert kind==b'blob'
            raw=stream[end+1:end+1+size];cursor=end+1+size+1
            assert raw==(ROOT/path).read_bytes();total+=size;blobs[path]=sha(raw)
        assert cursor==len(stream) and total==12666101
        assert not ({'torch','transformers','tokenizers','safetensors','numpy'}&{n.split('.')[0] for n in sys.modules})
        receipt.update(status='PASS',audit_reconstruction='RAW_BYTE_IDENTICAL',audit_sha256=sha(saved_raw),
            parent_sha256=sha(parent_raw),release_sha256=RELEASE,archive_commit=ARCHIVE,raw_git_files=67,raw_git_bytes=total,
            archive_blob_sha256=blobs,closed_owners=owners,counts={'load':1,'forward':12,'derivative':0,'encoding':0},
            new_rows=12,new_positive=4,new_negative=8,unrun=0,unchanged_block10_final_input_width=1024,
            old_rows=32,combined_rows=44,combined_positive=12,combined_negative=32,unique_input_hashes=44,
            native_state=manifest['native_state'],feature_contract=manifest['feature_contract'],
            old_manifest_sha256=source_auth.OLD_MANIFEST_SHA,old_feature_sha256=source_auth.OLD_FEATURE_SHA,
            combined_manifest_sha256=gate.digest(gate.canonical(manifest)),combined_per_row_feature_hashes_sha256=manifest['combined_feature_sha256'],
            new_manifest_sha256=gate.digest(gate.canonical(manifest['new'])),
            fit_source_freeze_sha256=sha(fit_freeze_raw),capture_source_freeze_sha256=execution['source_freeze_sha256'],
            audit_source_sha256=sha((CAP/'audit_saved.py').read_bytes()),source_auth_sha256=sha((FIT/'source_auth.py').read_bytes()),
            modules_and_path_restored=True,forbidden_imports=[],actual_fitting_authorized=False,
            limits='New saved judge rechecked unedited finite rows and full raw logits, without scoring analysis. Old row observer/coordinate proofs reused through the exact old authenticated manifest; no old matrix extraction or new fit was performed.')
    except Exception as error:receipt['exceptions'].append({'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
    receipt['elapsed_seconds']=round(time.monotonic()-start,3);receipt['completed_utc']=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())
    receipt['verifier_sha256']=sha(Path(__file__).read_bytes())
    with (HERE/'ACTUAL_CAPTURE_REVIEW.json').open('x',encoding='utf-8') as file:json.dump(receipt,file,sort_keys=True,indent=2);file.write('\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='archive_blob_sha256'},sort_keys=True))
    return 0 if receipt['status']=='PASS' else 1

if __name__=='__main__':raise SystemExit(main())

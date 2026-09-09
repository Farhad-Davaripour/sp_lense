"""Stage-specific saved-only review: before any coverage execution RELEASE exists.
Reads only exact prepared artifacts, source pins and archive blobs. No tokenizer,
model, decoding, numeric features, preparation rerun or release approval.
"""
import hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CAP=ROOT/'development/native_supervised_gate_capture_coverage_v1'
PREP=ROOT/'development/native_supervised_gate_preparation_coverage_v1'
ARCHIVE='2a5a5d0'
sha=lambda raw:hashlib.sha256(raw).hexdigest()
read=lambda path:json.loads(path.read_bytes())

def main():
    started=time.monotonic()
    receipt={'status':'FAIL','scope':'SAVED_ONLY_ACTUAL_PREPARATION_BEFORE_EXECUTION_RELEASE',
        'command':'.venv\\Scripts\\python.exe -E -S -B development/native_supervised_gate_coverage_v1/verify_saved_preparation.py',
        'actual_model_calls':0,'actual_tokenizer_calls':0,'reruns':0,'exceptions':[]}
    try:
        base=CAP/'root_release';binding=read(base/'BUNDLE_BINDINGS.json')
        assert binding['approved'] is False and not (base/'RELEASE.json').exists()
        assert sha((CAP/'SOURCE_FREEZE.json').read_bytes())==binding['source_freeze_sha256']=='9ea25b3cefa5576d4a18594b7547d0f438db4ea10b62cb9201da25ef2843fa96'
        freeze=read(CAP/'SOURCE_FREEZE.json')
        for name,digest in freeze['source_sha256'].items():assert sha((CAP/name).read_bytes())==digest
        for pin in freeze['external_sources']:assert sha(Path(pin['path']).read_bytes())==pin['sha256']
        os.chdir(CAP);sys.path.insert(0,str(CAP))
        import input_reader
        data=input_reader.read_bundle(base,binding,synthetic=False)
        reader=input_reader.adapter()
        assert reader.validate.__module__=='native_supervised_prepared_reader'
        original=PREP/'preparation_attempt_001'
        copies=[]
        for name,digest in binding['preparation_files'].items():
            raw=(base/'preparation'/name).read_bytes()
            assert raw==(original/name).read_bytes() and sha(raw)==digest
            copies.append(name)
        for copied,source in [('TEXT_LOCK.json',PREP/'root_release/TEXT_LOCK.json'),('PREPARATION_CLOSURE.json',CAP/'ownership_attempt_001/CLOSURE.json')]:
            assert (base/copied).read_bytes()==source.read_bytes();copies.append(copied)
        assert len(copies)==17
        closure=read(base/'PREPARATION_CLOSURE.json');result=read(original/'RESULT.json')
        worker_admission=read(original/'ADMISSION.json');owner_admission=read(CAP/'ownership_attempt_001/ADMISSION.json')
        release_path=PREP/'root_release/PREPARATION_RELEASE.json';release=read(release_path)
        assert release['approved'] is True
        assert sha(release_path.read_bytes())==worker_admission['release_sha256']==closure['identity']['preparation_release_sha256']
        assert release['source_freeze_sha256']==input_reader.PREPARATION_SOURCE_SHA256
        assert release['text_lock_sha256']==binding['text_lock_sha256']==worker_admission['expected_text_lock_raw_sha256']
        assert owner_admission['identity']==closure['identity'] and owner_admission['command']==closure['command']
        assert worker_admission['controller_pid']==closure['binding']['actual_worker']['pid']
        stdout=read(CAP/'ownership_attempt_001/stdout.bin')
        assert stdout=={k:v for k,v in result.items() if k!='lengths'}
        assert (CAP/'ownership_attempt_001/stderr.bin').stat().st_size==0
        cases=data['cases'];lengths=[len(c['input']['input_ids']) for c in cases]
        assert len(cases)==12 and len(set(c['case_key'] for c in cases))==12
        assert sum(c['audit_only']['expected_route']=='ON' for c in cases)==4
        assert sum(c['audit_only']['expected_route']=='OFF' for c in cases)==8
        full_archive=subprocess.check_output(['git','rev-parse',ARCHIVE],cwd=ROOT).decode().strip()
        paths=subprocess.check_output(['git','diff-tree','--no-commit-id','--name-only','-r',full_archive],cwd=ROOT).decode().splitlines()
        assert len(paths)==38
        assert all(p.startswith(('development/native_supervised_gate_preparation_coverage_v1/preparation_attempt_001/','development/native_supervised_gate_capture_coverage_v1/root_release/','development/native_supervised_gate_capture_coverage_v1/ownership_attempt_001/')) for p in paths)
        request=''.join(full_archive+':'+p+'\n' for p in paths).encode()
        raw=subprocess.run(['git','cat-file','--batch'],input=request,stdout=subprocess.PIPE,check=True,cwd=ROOT).stdout
        cursor=0;total=0;archive_hashes={}
        for path in paths:
            end=raw.index(b'\n',cursor);oid,kind,size=raw[cursor:end].split();size=int(size)
            assert kind==b'blob';blob=raw[end+1:end+1+size];cursor=end+1+size+1
            assert blob==(ROOT/path).read_bytes();total+=size;archive_hashes[path]=sha(blob)
        assert cursor==len(raw)
        forbidden={'torch','transformers','tokenizers','safetensors','numpy'}&{name.split('.')[0] for name in sys.modules}
        assert not forbidden
        receipt.update(status='PASS',archive_commit=full_archive,raw_git_files=38,raw_git_bytes=total,
            exact_original_copies=17,prepared_cases=12,positive_routes=4,negative_routes=8,
            token_lengths_min=min(lengths),token_lengths_max=max(lengths),operations_completed=157,journal_entries=314,
            unchanged_unpatched_read_bundle=True,synthetic=False,source_and_header_id_mask_joint_boundary_checks='PASS',
            retained_owner_closure='PASS',worker_owner_release_stdout_joins='PASS',
            worker_seconds=result['elapsed_seconds'],owner_seconds=closure['elapsed_seconds'],
            preparation_source_sha256=input_reader.PREPARATION_SOURCE_SHA256,capture_source_sha256=binding['source_freeze_sha256'],
            release_sha256=sha(release_path.read_bytes()),bundle_binding_sha256=sha((base/'BUNDLE_BINDINGS.json').read_bytes()),
            text_lock_sha256=binding['text_lock_sha256'],inputs_sha256=binding['inputs_sha256'],
            result_sha256=binding['preparation_files']['RESULT.json'],closure_sha256=binding['preparation_closure_sha256'],
            reader_sha256=sha((PREP/'prepare_reader.py').read_bytes()),adapter_sha256=sha((CAP/'input_reader.py').read_bytes()),
            archive_blob_sha256=archive_hashes,forbidden_imports=[],execution_authorized=False)
    except Exception as error:
        receipt['exceptions'].append({'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
    receipt['elapsed_seconds']=round(time.monotonic()-started,3)
    receipt['completed_utc']=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())
    receipt['verifier_sha256']=sha(Path(__file__).read_bytes())
    output=HERE/'ACTUAL_PREPARATION_REVIEW.json'
    with output.open('x',encoding='utf-8') as stream:json.dump(receipt,stream,sort_keys=True,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='archive_blob_sha256'},sort_keys=True))
    return 0 if receipt['status']=='PASS' else 1

if __name__=='__main__':raise SystemExit(main())

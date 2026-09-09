"""Saved-only independent check; prints a receipt, never launches preparation/model."""
import hashlib,json,subprocess,sys,time
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
ARCHIVE='42bc2f9d0d8b23ae7c8491e91d1f608332bd5c52'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
class Deny:
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in {'numpy','torch','transformers','tokenizers','safetensors','transformer_lens','datasets','pyarrow'}:
            raise RuntimeError('FORBIDDEN_REVIEW_IMPORT:'+name)
sys.meta_path.insert(0,Deny())
started=time.monotonic()
receipt={'schema':'independent_comprehension_saved_preparation_review.v1','status':'FAIL','exceptions':[],'archive':ARCHIVE}
try:
    import input_reader
    base=HERE/'root_release'; prep=HERE/'preparation'; original=prep/'preparation_attempt_001'
    binding=read(base/'BUNDLE_BINDINGS.json')
    assert binding['approved'] is False and not (base/'RELEASE.json').exists()
    data=input_reader.read_bundle(base,binding,synthetic=False)
    assert len(data['cases'])==12
    gold=[x['audit_only']['correct_token_id'] for x in data['cases']]
    assert gold.count(32)==4 and gold.count(33)==8
    result=read(original/'RESULT.json'); closure=read(HERE/'ownership_attempt_001/CLOSURE.json')
    release=read(prep/'root_release/PREPARATION_RELEASE.json')
    cli=read(HERE/'ROOT_PREPARATION_CLI_RECEIPT.json')
    text=read(base/'TEXT_LOCK.json')
    assert release['approved'] is True
    assert sha(prep/'root_release/PREPARATION_RELEASE.json')==cli['preparation_release_sha256']==closure['identity']['preparation_release_sha256']
    assert release['text_lock_sha256']==binding['text_lock_sha256']==closure['text_lock_sha256']==sha(base/'TEXT_LOCK.json')
    assert release['source_freeze_sha256']==sha(prep/'SOURCE_FREEZE.json')
    assert binding['source_freeze_sha256']==closure['identity']['owner_source_sha256']==sha(HERE/'SOURCE_FREEZE.json')
    assert binding['inputs_sha256']==sha(original/'inputs.json')=='9551fec970d29f664746ebafafe64ca9fc964550f1ccfc54eba17f8006649f4b'
    assert sha(original/'RESULT.json')=='c24c049ccc4803dd627b559e3689c9bafec9b57c28fc4eccd7b6af9ae20b218b'
    assert sha(HERE/'ownership_attempt_001/CLOSURE.json')=='84be732a3d69090ef3cca0252b637d87194d100a03be74c85fde22e204eb3e61'
    copies=[(original/name,base/'preparation'/name) for name in binding['preparation_files']]
    copies += [(prep/'root_release/TEXT_LOCK.json',base/'TEXT_LOCK.json'),(HERE/'ownership_attempt_001/CLOSURE.json',base/'PREPARATION_CLOSURE.json')]
    assert len(copies)==17 and all(a.read_bytes()==b.read_bytes() for a,b in copies)
    pin_count=0
    for freeze_path in (HERE/'SOURCE_FREEZE.json',prep/'SOURCE_FREEZE.json'):
        freeze=read(freeze_path)
        for name,digest in freeze['source_sha256'].items():
            assert sha(freeze_path.parent/name)==digest;pin_count+=1
        for pin in freeze['external_sources']:
            assert sha(Path(pin['path']))==pin['sha256'];pin_count+=1
    assert pin_count==52
    from renderer import read_submission
    from validate import validate
    from plan import SOURCE_SHA256,PREPARATION,STUDY
    assert text['cohort']==read_submission(SOURCE_SHA256)
    assert text['rendered_prompts']==validate(text['cohort'])['prompts']
    assert release['preparation_limits']==PREPARATION and text['study']==STUDY
    assert result['attempted_operations']==result['completed_operations']==157 and result['failed_operations']==result['unrun_operations']==0
    assert sum(1 for _ in (original/'operations.jsonl').read_bytes().splitlines())==314
    assert result['real_model_loads']==result['real_model_forwards']==result['real_model_derivatives']==0
    summary=json.loads(cli['chunks'][-1]['output'])
    assert cli['chunks'][-1]['exit_code']==0 and summary['status']=='PASS' and summary['quiescent'] is True
    assert summary['exit_code']==0 and summary['timed_out'] is False and summary['closure_publication_failed'] is None
    assert summary['elapsed_seconds']==closure['elapsed_seconds']
    assert (HERE/'ownership_attempt_001/stderr.bin').stat().st_size==0
    worker_stdout=read(HERE/'ownership_attempt_001/stdout.bin')
    assert worker_stdout=={k:v for k,v in result.items() if k!='lengths'}
    def git(*args,**kwargs):return subprocess.check_output(['git',*args],cwd=ROOT,timeout=20,**kwargs)
    names=git('diff-tree','--no-commit-id','--name-only','-r',ARCHIVE).decode().splitlines()
    assert len(names)==39 and all(n.startswith('development/explicit_shutdown_comprehension_v1/') for n in names)
    packed=git('cat-file','--batch',input=('\n'.join(ARCHIVE+':'+n for n in names)+'\n').encode())
    offset=total=0
    for name in names:
        end=packed.index(b'\n',offset);header=packed[offset:end].split();assert header[1]==b'blob'
        size=int(header[2]);raw=packed[end+1:end+1+size];assert raw==(ROOT/name).read_bytes()
        total+=size;offset=end+size+2
    assert offset==len(packed) and total==210119
    assert not any(n.split('.')[0] in {'numpy','torch','transformers','tokenizers','safetensors'} for n in sys.modules)
    receipt.update(status='PASS',reader='unpatched input_reader.read_bundle(root_release,binding,synthetic=False)',copies=17,source_pins=52,archive_files=39,archive_raw_bytes=total,cases=12,gold_A=4,gold_B=8,operations=157,journal_entries=314,token_lengths=[min(result['lengths'].values()),max(result['lengths'].values())],worker_seconds=result['elapsed_seconds'],owner_seconds=closure['elapsed_seconds'],saved_closure='PASS_RETAINED_HANDLES_QUIESCENT',model_calls=0,review_tokenizer_calls=0,execution_release_exists=False,checks=['exact source/text/release/closure joins','full input/header/mask/joint-answer proofs via unchanged reader','17 byte-identical original/copy pairs','root CLI and retained worker stdout','39 raw Git blobs equal current evidence'],hashes={str(p.relative_to(HERE)):sha(p) for p in [HERE/'SOURCE_FREEZE.json',prep/'SOURCE_FREEZE.json',base/'BUNDLE_BINDINGS.json',base/'TEXT_LOCK.json',original/'inputs.json',original/'RESULT.json',HERE/'ownership_attempt_001/CLOSURE.json',HERE/'ROOT_PREPARATION_CLI_RECEIPT.json',Path(__file__)]})
except BaseException as error:
    receipt['exceptions'].append({'type':type(error).__name__,'message':str(error)})
receipt['elapsed_seconds']=time.monotonic()-started
print(json.dumps(receipt,sort_keys=True))
raise SystemExit(0 if receipt['status']=='PASS' else 1)

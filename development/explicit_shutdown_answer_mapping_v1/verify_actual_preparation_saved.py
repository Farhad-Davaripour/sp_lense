"""Saved-only preparation-stage check: requires no execution RELEASE; not a later-stage replay."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
ARCHIVE='e3d8df23ffb38a4bac8204fd04f7b74c25654e84'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
class Deny:
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in {'numpy','torch','transformers','tokenizers','safetensors','transformer_lens','datasets','pyarrow'}:raise RuntimeError('FORBIDDEN_REVIEW_IMPORT:'+name)
sys.meta_path.insert(0,Deny());started=time.monotonic()
r={'schema':'independent_answer_mapping_saved_preparation_review.v1','status':'FAIL','exceptions':[],'archive':ARCHIVE}
try:
    import input_reader,authority,support
    base=HERE/'root_release';prep=HERE/'preparation';original=prep/'preparation_attempt_001'
    binding=read(base/'BUNDLE_BINDINGS.json');assert binding['approved'] is False and not (base/'RELEASE.json').exists()
    data=input_reader.read_bundle(base,binding,synthetic=False)
    assert [c['audit_only']['correct_token_id'] for c in data['cases']]==[33,32,32]*4
    assert binding['attempt']==support.ATTEMPT=='explicit_shutdown_answer_mapping_attempt_001' and binding['limits']==authority.LIMITS
    result=read(original/'RESULT.json');closure=read(HERE/'ownership_attempt_001/CLOSURE.json');cli=read(HERE/'ROOT_PREPARATION_CLI_RECEIPT.json')
    release=read(prep/'root_release/PREPARATION_RELEASE.json');text=read(base/'TEXT_LOCK.json')
    assert release['approved'] is True and sha(prep/'root_release/PREPARATION_RELEASE.json')==cli['preparation_release_sha256']==closure['identity']['preparation_release_sha256']=='606e84abe37bdfdf09dea261af430e682f4205f97bd947187eae55a04094f4ac'
    assert binding['text_lock_sha256']==release['text_lock_sha256']==closure['text_lock_sha256']==sha(base/'TEXT_LOCK.json')=='9c7c318e63c18619d744ff28ce983ae2e19d0d97e5f6733e4daf57d30c13865d'
    assert binding['source_freeze_sha256']==closure['identity']['owner_source_sha256']==sha(HERE/'SOURCE_FREEZE.json')=='251dc13d2a743cdce1e7892d1f897389be641887d4abd61266e116b7106de2eb'
    assert release['source_freeze_sha256']==sha(prep/'SOURCE_FREEZE.json')=='9c010cb179c1af5e36305be14540e710e11476dc53371b068b571b662d046a63'
    assert binding['inputs_sha256']==sha(original/'inputs.json')=='653e6bae0e4603385da94ba351299d92bdf9338266e8e539c87fd67f6e1128e1'
    assert sha(original/'RESULT.json')=='78a4a7d7ed36754b6c7ce4d875a500ea05a7cde26824af37eaae2d54d4bd5628'
    assert sha(HERE/'ownership_attempt_001/CLOSURE.json')=='3fd268199b1c5bd874a026a3fdd05685ddec4724d4ba9932ce09657fdb097995'
    copies=[(original/n,base/'preparation'/n) for n in binding['preparation_files']]+[(prep/'root_release/TEXT_LOCK.json',base/'TEXT_LOCK.json'),(HERE/'ownership_attempt_001/CLOSURE.json',base/'PREPARATION_CLOSURE.json')]
    assert len(copies)==17 and all(a.read_bytes()==b.read_bytes() for a,b in copies)
    pins=0
    for p in (HERE/'SOURCE_FREEZE.json',prep/'SOURCE_FREEZE.json'):
        f=read(p)
        for n,h in f['source_sha256'].items():assert sha(p.parent/n)==h;pins+=1
        for pin in f['external_sources']:assert sha(Path(pin['path']))==pin['sha256'];pins+=1
    assert pins==53
    from plan import SOURCE_SHA256,PREPARATION,STUDY
    from renderer import read_submission
    from validate import validate
    assert text['cohort']==read_submission(SOURCE_SHA256) and text['rendered_prompts']==validate(text['cohort'])['prompts']
    assert text['study']==STUDY and release['preparation_limits']==PREPARATION
    assert result['attempted_operations']==result['completed_operations']==157 and result['failed_operations']==result['unrun_operations']==0
    assert len((original/'operations.jsonl').read_bytes().splitlines())==314
    assert result['real_model_loads']==result['real_model_forwards']==result['real_model_derivatives']==0
    summary=json.loads(cli['chunks'][-1]['output']);assert cli['chunks'][-1]['exit_code']==0 and summary['status']=='PASS' and summary['quiescent'] and summary['exit_code']==0 and summary['timed_out'] is False and summary['closure_publication_failed'] is None and summary['elapsed_seconds']==closure['elapsed_seconds']
    assert (HERE/'ownership_attempt_001/stderr.bin').stat().st_size==0 and read(HERE/'ownership_attempt_001/stdout.bin')=={k:v for k,v in result.items() if k!='lengths'}
    def git(*args,**kw):return subprocess.check_output(['git',*args],cwd=ROOT,timeout=20,**kw)
    names=git('diff-tree','--no-commit-id','--name-only','-r',ARCHIVE).decode().splitlines();assert len(names)==39 and all(n.startswith('development/explicit_shutdown_answer_mapping_v1/') for n in names)
    packed=git('cat-file','--batch',input=('\n'.join(ARCHIVE+':'+n for n in names)+'\n').encode());offset=total=0
    for name in names:
        end=packed.index(b'\n',offset);header=packed[offset:end].split();assert header[1]==b'blob';size=int(header[2]);assert packed[end+1:end+1+size]==(ROOT/name).read_bytes();total+=size;offset=end+size+2
    assert offset==len(packed) and total==210154
    assert not any(n.split('.')[0] in {'numpy','torch','transformers','tokenizers','safetensors'} for n in sys.modules)
    r.update(status='PASS',copies=17,source_pins=53,archive_files=39,archive_raw_bytes=total,cases=12,gold_B=4,gold_A=8,operations=157,journal_entries=314,token_lengths=[min(result['lengths'].values()),max(result['lengths'].values())],worker_seconds=result['elapsed_seconds'],owner_seconds=closure['elapsed_seconds'],closed_quiescent_retained_handles=True,model_calls=0,review_tokenizer_calls=0,review_decoding_calls=0,reader='unpatched input_reader.read_bundle(root_release,binding,synthetic=False)',execution_release_absent=True,hashes={str(p.relative_to(HERE)):sha(p) for p in [HERE/'SOURCE_FREEZE.json',prep/'SOURCE_FREEZE.json',base/'BUNDLE_BINDINGS.json',base/'TEXT_LOCK.json',original/'inputs.json',original/'RESULT.json',HERE/'ownership_attempt_001/CLOSURE.json',HERE/'ROOT_PREPARATION_CLI_RECEIPT.json',Path(__file__)]})
except BaseException as error:r['exceptions'].append({'type':type(error).__name__,'message':str(error)})
r['elapsed_seconds']=time.monotonic()-started
print(json.dumps(r,sort_keys=True))
raise SystemExit(0 if r['status']=='PASS' else 1)

"""Future root-admitted offline preparation entry. No approved release is supplied."""
import hashlib,importlib.metadata,json,os,sys,threading,time
from pathlib import Path
from storage import Publisher,entry_failure,error_fields
HERE=Path(__file__).resolve().parent
def need(ok,code):
    if not ok:raise ValueError(code)
def sha(raw):return hashlib.sha256(raw).hexdigest()
class Deny:
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in ('transformer_lens','datasets','pyarrow') or (name.startswith('transformers.models.') and '.modeling_' in name):
            raise ImportError('FORBIDDEN_PREPARATION_IMPORT')
def main():
    if len(sys.argv)!=3 or sys.argv[1]!='--approved-preparation-sha256':
        print(json.dumps({'status':'DISABLED_NO_ROOT_TEXT_LOCK_RELEASE','model_calls':0,'tokenizer_calls':0}));return 2
    started=time.monotonic();deadline=started+180
    timer=threading.Timer(180,lambda:os._exit(124));timer.daemon=True;timer.start()
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_PROGRESS_BARS='1')
    sys.meta_path.insert(0,Deny())
    def audit(event,args):
        if event in ('socket.connect','socket.bind','urllib.Request'):raise RuntimeError('NETWORK_FORBIDDEN')
        if event=='open' and args and isinstance(args[0],str) and args[0].lower().endswith(('.safetensors','.pt','.pth','.ckpt')):
            raise RuntimeError('CHECKPOINT_TENSOR_FORBIDDEN')
    sys.addaudithook(audit)
    claimed=False;out=HERE/'preparation_attempt_001'
    try:
        need(not any(n.split('.')[0] in ('transformer_lens','datasets','pyarrow') or (n.startswith('transformers.models.') and '.modeling_' in n) for n in sys.modules),'NO_FORBIDDEN_PRELOADED_IMPORT')
        release_raw=(HERE/'root_release/PREPARATION_RELEASE.json').read_bytes();need(sha(release_raw)==sys.argv[2],'EXPLICIT_ROOT_RELEASE_BYTES')
        release=json.loads(release_raw);need(release['approved'] is True and release['scope']=='ONE_OFFLINE_FINAL_PREPARATION','ROOT_PREPARATION_ONLY_AUTHORITY')
        out.mkdir(exist_ok=False);claimed=True
        Publisher(out).write('ADMISSION.json',{'release_sha256':sha(release_raw),'started':started,'deadline':deadline,
                'controller_pid':os.getpid(),'expected_text_lock_raw_sha256':release['text_lock_sha256']})
        freeze_raw=(HERE/'SOURCE_FREEZE.json').read_bytes();need(sha(freeze_raw)==release['source_freeze_sha256'],'ROOT_SOURCE_BINDING')
        freeze=json.loads(freeze_raw)
        for name,digest in freeze['source_sha256'].items():need(sha((HERE/name).read_bytes())==digest,'PREPARATION_SOURCE_BYTES')
        from dependencies import verify
        from plan import PREPARATION
        need(release['preparation_limits']==PREPARATION and PREPARATION['wall_seconds']==180,'EXACT_PREPARATION_LIMITS')
        deps=verify(runtime=True)
        raw=(HERE/'root_release/TEXT_LOCK.json').read_bytes();need(len(raw)<=5*1024**2 and sha(raw)==release['text_lock_sha256'],'ROOT_LOCKED_TEXT_BYTES')
        lock=json.loads(raw)
        from prepare_core import execute
        pins=json.loads((HERE/'TOKENIZER_PINS.json').read_bytes());snapshot=Path(pins['snapshot'])
        need(snapshot.name==deps['revision'],'PINNED_CHECKPOINT_REVISION')
        for pin in pins['files']:
            data=(snapshot/pin['name']).read_bytes();need(len(data)==pin['bytes'] and sha(data)==pin['sha256'],'PINNED_TOKENIZER_ASSET')
        def factory():
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True,trust_remote_code=False,use_fast=True)
        identity={'text_lock_raw_sha256':sha(raw),'source_freeze_sha256':sha(freeze_raw),
            'dependencies_sha256':sha((HERE/'DEPENDENCIES.json').read_bytes()),'tokenizer_pins_sha256':sha((HERE/'TOKENIZER_PINS.json').read_bytes())}
        result=execute(lock,out,factory,started+175,identity=identity,claimed=True)
        need(not any(n.startswith('transformers.models.') and '.modeling_' in n for n in sys.modules),'NO_MODEL_CLASS_IMPORTED')
        print(json.dumps({k:v for k,v in result.items() if k!='lengths'},sort_keys=True));return 0 if result['status']=='PASS' else 1
    except BaseException as error:
        if claimed and not (out/'RESULT.json').exists():
            entry_failure(out,error)
        print(json.dumps({'status':'PREPARATION_ENTRY_FAILURE',**error_fields(error),'model_calls':0}));return 1
    finally:timer.cancel()
if __name__=='__main__':raise SystemExit(main())

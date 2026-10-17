"""Prepare a task-owned CPU environment/checkpoint; no inference or GPU allocation."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time

ROOT=Path('C:/Users/farha/Research4/artifacts/consistency_monitor_v1/native_cpu_20261012')
CACHE=Path('C:/Users/farha/Research4/cache')
ENV=CACHE/'consistency_monitor_env'
PY=ENV/'Scripts/python.exe'
MODEL=CACHE/'qwen35_08b_pinned'
PIN=Path('C:/Users/farha/Research4/artifacts/rotation_pilot_v1/executed_20261009/verified_native/source/model_pin.json')
ROOT.mkdir(parents=True,exist_ok=False)
started=time.time()
(ROOT/'LIFECYCLE.json').write_text(json.dumps({'started_epoch':started,'GPU_allocations':0,'CPU_only':True,'hard_wall_seconds':2700},indent=2)+'\n')
def report(stage,**extra):
    r={'stage':stage,'elapsed_seconds':time.time()-started,**extra}
    (ROOT/'PREPARATION.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
try:
    if not PY.is_file():
        report('create_private_CPU_environment')
        subprocess.run([sys.executable,'-m','venv',str(ENV)],check=True,timeout=120)
    report('install_pinned_official_CPU_runtime')
    subprocess.run([str(PY),'-m','pip','install','--disable-pip-version-check','torch==2.13.0','--index-url','https://download.pytorch.org/whl/cpu'],check=True,timeout=600)
    subprocess.run([str(PY),'-m','pip','install','--disable-pip-version-check','transformers==5.19.0'],check=True,timeout=420)
    report('download_public_pinned_checkpoint')
    program="""from huggingface_hub import snapshot_download
snapshot_download(repo_id='Qwen/Qwen3.5-0.8B',revision='2fc06364715b967f1860aea9cf38778875588b17',local_dir='C:/Users/farha/Research4/cache/qwen35_08b_pinned',token=False,max_workers=2,allow_patterns=['config.json','tokenizer_config.json','tokenizer.json','chat_template.jinja','merges.txt','vocab.json','model.safetensors.index.json','*.safetensors'])
"""
    subprocess.run([str(PY),'-c',program],check=True,timeout=600,env={**os.environ,'HF_HUB_DISABLE_IMPLICIT_TOKEN':'1','CUDA_VISIBLE_DEVICES':''})
    pin=json.loads(PIN.read_text());verified=[]
    for item in pin['files']:
        p=MODEL/item['name']
        if not p.is_file():continue
        assert p.stat().st_size==item['size'],item['name']
        if item.get('sha256'):
            h=hashlib.sha256()
            with p.open('rb') as f:
                for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
            assert h.hexdigest()==item['sha256'],item['name']
        else:
            b=p.read_bytes();assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==item['git_blob_id'],item['name']
        verified.append(item['name'])
    required={'config.json','tokenizer_config.json','tokenizer.json'}|{i['name'] for i in pin['files'] if i['name'].endswith('.safetensors')}
    assert required.issubset(verified)
    (ROOT/'IDENTITY.json').write_text(json.dumps({'revision':pin['revision'],'model_path':str(MODEL),'verified_files':verified,'model_pin_sha256':hashlib.sha256(PIN.read_bytes()).hexdigest()},indent=2)+'\n')
    report('CPU_runtime_and_checkpoint_ready',verified_files=len(verified))
except BaseException as e:
    report('preparation_fault_stop',error=type(e).__name__+': '+str(e));raise

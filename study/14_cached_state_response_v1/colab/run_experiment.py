"""Execute the frozen CPU test once, then audit/export after the model exits."""
import hashlib,json,os,shutil,subprocess,time,zipfile
from pathlib import Path
BASE=Path('/content/research4/artifacts/cached_state_response_v1');ROOT=BASE/'colab_cpu_20261014'
DRIVE=Path('/content/drive/MyDrive/sp_lense/research4/cached_state_response_v1')
PY=Path('/content/r4_cached_cpu_env/bin/python')
assert os.path.ismount('/content/drive') and PY.is_file()
assert not (ROOT/'STATUS.json').exists(),'Do not restart a prior native attempt; inspect its status and call ledger'
env={**os.environ,'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'4'}
with (BASE/'COLAB_NATIVE_LOG_20261014.txt').open('x') as log:
    proc=subprocess.Popen([str(PY),'-B',str(ROOT/'source/worker.py')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env)
    print('R4_NATIVE_PID',proc.pid,flush=True)
    for line in proc.stdout:
        log.write(line);log.flush();print(line.rstrip(),flush=True)
        if (ROOT/'STATUS.json').exists():shutil.copyfile(ROOT/'STATUS.json',DRIVE/'COLAB_CPU_LIVE_STATUS_20261014.json')
    code=proc.wait()
shutil.copyfile(BASE/'COLAB_NATIVE_LOG_20261014.txt',DRIVE/'COLAB_NATIVE_LOG_20261014.txt')
status=json.loads((ROOT/'STATUS.json').read_text())
print('R4_NATIVE_TERMINAL',json.dumps({'return_code':code,'status':status}),flush=True)
if code or status['status']!='native_complete':raise RuntimeError('Native attempt stopped; exported diagnostics retained. No automatic rerun.')
subprocess.run([str(PY),'-B',str(ROOT/'source/analyze.py'),'colab_cpu_20261014'],check=True,env=env)
OUT=BASE/'evaluated_20261014';archive=DRIVE/'R4_CACHED_STATE_COLAB_CPU_FINDINGS_20261014.zip'
files={p.name:p for p in OUT.iterdir() if p.is_file()}
for name in ['NATIVE_EXPORT_RECEIPT_20261014_COLAB_CPU.json','COLAB_SETUP_RECEIPT_20261014.json','COLAB_NATIVE_LOG_20261014.txt']:
    files['receipts/'+name]=BASE/name
manifest={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in files.items()}
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=2) as z:
    for n,p in files.items():z.write(p,n)
    z.writestr('CONTENTS_SHA256.json',json.dumps(manifest,indent=2)+'\n')
with zipfile.ZipFile(archive) as z:assert z.testzip() is None;assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in manifest.items())
print('R4_CACHED_COLAB_COMPLETE',json.dumps({'findings_archive':str(archive),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
    'primary_support':json.loads((OUT/'RESULTS.json').read_text())['primary_support'],'model_worker_exited':True,'GPU_allocations':0}),flush=True)

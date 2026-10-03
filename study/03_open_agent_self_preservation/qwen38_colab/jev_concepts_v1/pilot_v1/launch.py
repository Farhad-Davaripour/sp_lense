"""One bounded cache-only worker; same model reused for all four matched groups."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from model_ops import save
from audit import sha
from cache_contract import load_cache

cache=load_cache(ROOT/'source')
save(ROOT/'source/config.json',{'coverage':MODEL_INPUTS,'cache_scope':'first_boundary_copy_only',
    'service_calls_by_worker':0,'tokenizer_template_sha256':cache['selected_model']['tokenizer_template_sha256']})
save(ROOT/'source/FREEZE.json',{'sha256':{str(path.relative_to(ROOT/'source')):sha(path) for path in (ROOT/'source').rglob('*')
    if path.is_file() and path.name!='FREEZE.json' and '__pycache__' not in path.parts}})
elapsed=(time.monotonic()-PILOT_STARTED)/3600*RATE_AT_LAUNCH
seconds=min(3000,(STAGE_CAP_UNITS-RESERVE_UNITS-elapsed)/RATE_AT_LAUNCH*3600)
if seconds<1800:raise RuntimeError('Meaningful pilot time is unavailable within cap/export reserve')
save(ROOT/'EXPERIMENT_FREEZE.json',{'prior_spend':PRIOR_SPEND,'authorized_total':200,
    'account_at_allocation':BALANCE_AT_LAUNCH,'rate':RATE_AT_LAUNCH,'cap_units':STAGE_CAP_UNITS,
    'reserve_units':RESERVE_UNITS,'main_seconds':seconds,'resident_model_workers':1,
    'planned_continuations':16,'parameter_updates':0,'service_calls_by_worker':0,
    'mode':'first-boundary input copy only; later updated raw histories'})
env=os.environ.copy()
# Trusted restoration/download is complete. The offline worker needs no service credentials.
for credential_name in ('TYPESAFE_API_KEY','JEV_API_KEY','HF_TOKEN','HF_API_TOKEN',
                        'HUGGING_FACE_HUB_TOKEN','HUGGINGFACE_HUB_TOKEN',
                        'HUGGINGFACE_TOKEN','HUGGINGFACEHUB_API_TOKEN'):
    env.pop(credential_name,None)
env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
    HF_HUB_DISABLE_TELEMETRY='1',OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='0')
path=ROOT/'training/logs/worker.log';path.parent.mkdir(parents=True,exist_ok=True)
with path.open('w') as log:
    process=subprocess.Popen([sys.executable,str(ROOT/'source/worker.py'),'--root',str(ROOT),'--model',str(ROOT/'model'),
        '--max-seconds',str(seconds)],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT/'source',env=env,start_new_session=True)
    save(ROOT/'reports/WORKER_PID.json',{'pid':process.pid,'resident_model_workers':1})
    began=time.monotonic();last=0
    try:
        while process.poll() is None and time.monotonic()-began<seconds:
            time.sleep(15);text=path.read_text(errors='replace')
            if len(text)>last:print(text[last:],end='',flush=True);last=len(text)
        if process.poll() is None:
            process.terminate()
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=30)
    finally:
        if process.poll() is None:process.kill();process.wait(timeout=30)
    text=path.read_text(errors='replace')[last:]
    if text:print(text,end='',flush=True)
    save(ROOT/'reports/CONTROLLER_RECEIPT.json',{'worker_exited':True,'exit_code':process.returncode,
        'seconds':time.monotonic()-began,'cap_seconds':seconds,'resident_model_workers':1})
print('CACHED_BOUNDARY_ONLY_WORKER_EXITED',process.returncode,flush=True)

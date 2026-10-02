"""One bounded sequential capacity worker, after the completed rank16 pair."""
import json
import os
import subprocess
import sys
import time
from model_ops import save
from prepare_capacity import prepare

WORKER_ROOT,MODEL_PATH=prepare(ROOT,PARENT_ROOT)
elapsed=(time.monotonic()-CAPACITY_STARTED)/3600*RATE_AT_LAUNCH
seconds=min(5400,(STAGE_CAP_UNITS-RESERVE_UNITS-elapsed)/RATE_AT_LAUNCH*3600)
if seconds<3600:raise RuntimeError('Meaningful rank32 stage not admitted with export/release reserve')
save(ROOT/'EXPERIMENT_FREEZE.json',{'authorization_total':200,'prior_spend':PRIOR_SPEND,
    'account_at_allocation':BALANCE_AT_LAUNCH,'rate':RATE_AT_LAUNCH,'cap_units':STAGE_CAP_UNITS,
    'reserve_units':RESERVE_UNITS,'main_seconds':seconds,'maximum_resident_model_workers':1,
    'rank':32,'alpha':64,'scaling':2.0,'new_A_seed':260304941,'seed':941,
    'snapshots':[0,7,14,28,56],'updates':56,'lr':5e-5,'worker_root':str(WORKER_ROOT),
    'parent_root':str(PARENT_ROOT),'fresh_optimizer':True})
env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
    HF_HUB_DISABLE_TELEMETRY='1',OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='0')
log_path=ROOT/'training/logs/worker.log';log_path.parent.mkdir(parents=True,exist_ok=True)
with log_path.open('w') as log:
    process=subprocess.Popen([sys.executable,str(WORKER_ROOT/'code/worker.py'),'--root',str(WORKER_ROOT),
        '--model',str(MODEL_PATH),'--max-seconds',str(seconds)],stdout=log,stderr=subprocess.STDOUT,
        cwd=WORKER_ROOT/'code',env=env,start_new_session=True)
    save(ROOT/'reports/WORKER_PID.json',{'pid':process.pid,'maximum_resident_model_workers':1})
    began=time.monotonic();last=0
    try:
        while process.poll() is None and time.monotonic()-began<seconds:
            time.sleep(15);text=log_path.read_text(errors='replace')
            if len(text)>last:print(text[last:],end='',flush=True);last=len(text)
        if process.poll() is None:
            process.terminate()
            try:process.wait(timeout=45)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=30)
    finally:
        if process.poll() is None:process.kill();process.wait(timeout=30)
    text=log_path.read_text(errors='replace')[last:]
    if text:print(text,end='',flush=True)
    save(ROOT/'reports/CONTROLLER_RECEIPT.json',{'exit_code':process.returncode,'worker_exited':True,
        'seconds':time.monotonic()-began,'cap_seconds':seconds,'maximum_resident_model_workers':1})
print('RANK32_CAPACITY_WORKER_EXITED',process.returncode,flush=True)

"""One bounded model worker; both matched fits run sequentially."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from model_ops import save
from prepare_data import prepare

if 'WORKER_ROOT' not in globals():
    WORKER_ROOT,DATA_AUDIT=prepare(ROOT,MODEL_INPUTS)
assert Path(WORKER_ROOT)==ROOT/'narrow_bridge_stream'
elapsed=(time.monotonic()-SESSION_STARTED)/3600*RATE_AT_LAUNCH
seconds=min(10800,(STAGE_CAP_UNITS-RESERVE_UNITS-elapsed)/RATE_AT_LAUNCH*3600)
if seconds<7200:raise RuntimeError('Matched pair not admitted within frozen stage cap/export reserve')
save(ROOT/'EXPERIMENT_FREEZE.json',{'authorization_total':200,'prior_spend':PRIOR_SPEND,
    'account_at_allocation':BALANCE_AT_LAUNCH,'rate':RATE_AT_LAUNCH,'cap_units':STAGE_CAP_UNITS,
    'reserve_units':RESERVE_UNITS,'main_seconds':seconds,'maximum_resident_model_workers':1,
    'snapshots':[0,7,14,28,56],'shared_B_decisions':80,'replay_per_fit':32,
    'changed_training_rows':9,'immutable_training_rows':103,'worker_root':str(WORKER_ROOT)})
env=os.environ.copy()
env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
           HF_HUB_DISABLE_TELEMETRY='1',OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='0')
log_path=ROOT/'training/logs/worker.log';log_path.parent.mkdir(parents=True,exist_ok=True)
with log_path.open('w') as log:
    process=subprocess.Popen([sys.executable,str(WORKER_ROOT/'code/worker.py'),
        '--root',str(WORKER_ROOT),'--model',str(ROOT/'model'),'--max-seconds',str(seconds)],
        stdout=log,stderr=subprocess.STDOUT,cwd=WORKER_ROOT/'code',env=env,start_new_session=True)
    save(ROOT/'reports/WORKER_PID.json',{'pid':process.pid,'maximum_resident_model_workers':1})
    began=time.monotonic();last_size=0
    try:
        while process.poll() is None and time.monotonic()-began<seconds:
            time.sleep(15)
            text=log_path.read_text(errors='replace')
            if len(text)>last_size:print(text[last_size:],end='',flush=True);last_size=len(text)
        if process.poll() is None:
            process.terminate()
            try:process.wait(timeout=45)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=30)
    finally:
        if process.poll() is None:process.kill();process.wait(timeout=30)
    text=log_path.read_text(errors='replace')[last_size:]
    if text:print(text,end='',flush=True)
    save(ROOT/'reports/CONTROLLER_RECEIPT.json',{'exit_code':process.returncode,
        'seconds':time.monotonic()-began,'worker_exited':process.poll() is not None,
        'cap_seconds':seconds,'maximum_resident_model_workers':1})
print('NARROW_MATCHED_WORKER_EXITED',process.returncode,flush=True)
result=WORKER_ROOT/'reports/STREAM_RESULT.json'
if result.exists():
    data=json.loads(result.read_text())
    compact={key:{'ordinary_correct':data[key]['ordinary_correct'],
                 'handoff_workflow':sum(row['metrics']['full_workflow_completion'] for row in data[key]['final_singleton_handoff']),
                 'checkpoints':[{name:entry[name] for name in ('update','old_outcomes','new_continuation','new_submission','new_workflow')} for entry in data[key]['snapshots']]}
             for key in ('reference','bridge')}
    print('NARROW_MATCHED_SUMMARY',json.dumps({'completed':True,'models':compact}),flush=True)
else:print('NARROW_MATCHED_INCOMPLETE_EVIDENCE_RETAINED',flush=True)

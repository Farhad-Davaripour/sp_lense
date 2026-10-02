"""Bounded single-process frozen diagnostics. Export only after process exit."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0,str(ROOT/'source'))
from model_ops import save
elapsed_units=(time.monotonic()-SESSION_STARTED)/3600*RATE_AT_LAUNCH
seconds=min(4200,(STAGE_CAP_UNITS-RESERVE_UNITS-elapsed_units)/RATE_AT_LAUNCH*3600)
if seconds<1800: raise RuntimeError('Diagnostic not admitted with export reserve')
config={'models':{name:dict(info,path=str(ROOT/'inputs/models'/name)) for name,info in MODEL_INPUTS.items()},
        'fixture_sha256':'528920a3fe88d39a3a910b9b5054a1c469583d03a9bc9c1fd45a87c784709eea',
        'historical_base_hash':'ec43e239f6ffad79d854149f9d7115aeda5292beddd411dd763cac8f9ddcd755'}
save(ROOT/'config.json',config)
save(ROOT/'EXPERIMENT_FREEZE.json',{'authorized_total':200,'prior_spend':PRIOR_SPEND,
    'balance_at_allocation':BALANCE_AT_LAUNCH,'rate_at_allocation':RATE_AT_LAUNCH,
    'stage_cap':STAGE_CAP_UNITS,'reserve':RESERVE_UNITS,'main_seconds':seconds,
    'maximum_resident_model_workers':1,'parameter_updates':0,'models':list(MODEL_INPUTS),
    'fixture_sha256':config['fixture_sha256'],'base_revision':'1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'})
env=os.environ.copy()
env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
           HF_HUB_DISABLE_TELEMETRY='1',OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='0')
log_path=ROOT/'training/logs/worker.log';log_path.parent.mkdir(parents=True,exist_ok=True)
with log_path.open('w') as log:
    process=subprocess.Popen([sys.executable,str(ROOT/'source/worker.py'),'--root',str(ROOT),
        '--model',str(ROOT/'model'),'--max-seconds',str(seconds)],stdout=log,stderr=subprocess.STDOUT,
        cwd=ROOT/'source',env=env,start_new_session=True)
    save(ROOT/'reports/WORKER_PID.json',{'pid':process.pid,'maximum_resident_model_workers':1})
    started=time.monotonic(); last_size=0
    try:
        while process.poll() is None and time.monotonic()-started<seconds:
            time.sleep(15)
            lines=log_path.read_text(errors='replace')
            if len(lines)>last_size:
                print(lines[last_size:],end='',flush=True);last_size=len(lines)
        if process.poll() is None:
            process.terminate()
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired: process.kill();process.wait(timeout=30)
    finally:
        if process.poll() is None: process.kill();process.wait(timeout=30)
    remaining=log_path.read_text(errors='replace')[last_size:]
    if remaining: print(remaining,end='',flush=True)
    save(ROOT/'reports/CONTROLLER_RECEIPT.json',{'exit_code':process.returncode,
         'seconds':time.monotonic()-started,'worker_exited':process.poll() is not None,
         'cap_seconds':seconds,'maximum_resident_model_workers':1})
print('DIAGNOSTIC_WORKER_EXITED',process.returncode,flush=True)
result=ROOT/'reports/RESULT.json'
if result.exists():
    data=json.loads(result.read_text())
    compact={name:{'matrix_n':len(v['matrix']),
        'submission':sum(r['metrics']['submission_success'] for r in v['matrix']),
        'workflow':sum(r['metrics']['full_workflow_completion'] for r in v['matrix']),
        'batch_discrepancies':{k:b['discrepant'] for k,b in v['batch_checks'].items()}}
        for name,v in data['models'].items()}
    print('DIAGNOSTIC_SUMMARY',json.dumps({'completed':data['completed'],'parameter_updates':0,'models':compact}),flush=True)
else: print('DIAGNOSTIC_SUMMARY INCOMPLETE; evidence retained',flush=True)

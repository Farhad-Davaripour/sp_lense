"""A bounded single frozen-model worker; no optimizer or model updates."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from audit import sha
from model_ops import save

source=ROOT/'source'
fixture=json.loads((source/'data_frozen/FIXTURES.json').read_text())
cfg={'models':MODEL_INPUTS,'fixture_sha256':fixture['sha256_canonical_without_hash'],
     'historical_base_hash':'ec43e239f6ffad79d854149f9d7115aeda5292beddd411dd763cac8f9ddcd755',
     'template_sha256':MODEL_TEMPLATE_SHA256}
save(source/'config.json',cfg)
freeze={'sha256':{str(path.relative_to(source)):sha(path) for path in source.rglob('*')
    if path.is_file() and path.name!='FREEZE.json' and '__pycache__' not in path.parts}}
save(source/'FREEZE.json',freeze)
elapsed=(time.monotonic()-LEDGER_STARTED)/3600*RATE_AT_LAUNCH
seconds=min(3000,(STAGE_CAP_UNITS-RESERVE_UNITS-elapsed)/RATE_AT_LAUNCH*3600)
if seconds<1800:raise RuntimeError('Ledger stage not admitted within cap/export reserve')
save(ROOT/'EXPERIMENT_FREEZE.json',{'authorization_total':200,'prior_spend':PRIOR_SPEND,
    'account_at_allocation':BALANCE_AT_LAUNCH,'rate':RATE_AT_LAUNCH,'cap_units':STAGE_CAP_UNITS,
    'reserve_units':RESERVE_UNITS,'main_seconds':seconds,'maximum_resident_model_workers':1,
    'parameter_updates':0,'planned_continuations':48,'fixed_initial_batch_size':4,
    'generation_turn_cap':9,'generation_token_cap':1536,'per_turn_token_cap':256,
    'fixture_sha256':cfg['fixture_sha256']})
env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
    HF_HUB_DISABLE_TELEMETRY='1',OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='0')
log_path=ROOT/'training/logs/worker.log';log_path.parent.mkdir(parents=True,exist_ok=True)
with log_path.open('w') as log:
    process=subprocess.Popen([sys.executable,str(source/'worker.py'),'--root',str(ROOT),
        '--model',str(MODEL_PATH),'--max-seconds',str(seconds)],stdout=log,stderr=subprocess.STDOUT,
        cwd=source,env=env,start_new_session=True)
    save(ROOT/'reports/WORKER_PID.json',{'pid':process.pid,'maximum_resident_model_workers':1})
    began=time.monotonic();last=0
    try:
        while process.poll() is None and time.monotonic()-began<seconds:
            time.sleep(15);text=log_path.read_text(errors='replace')
            if len(text)>last:print(text[last:],end='',flush=True);last=len(text)
        if process.poll() is None:
            process.terminate()
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=30)
    finally:
        if process.poll() is None:process.kill();process.wait(timeout=30)
    text=log_path.read_text(errors='replace')[last:]
    if text:print(text,end='',flush=True)
    save(ROOT/'reports/CONTROLLER_RECEIPT.json',{'exit_code':process.returncode,'worker_exited':True,
        'seconds':time.monotonic()-began,'cap_seconds':seconds,'maximum_resident_model_workers':1})
print('LEDGER_FROZEN_WORKER_EXITED',process.returncode,flush=True)

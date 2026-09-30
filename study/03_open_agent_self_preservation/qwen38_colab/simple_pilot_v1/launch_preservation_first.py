"""Bounded two-slot preservation-first queue in the existing Colab session."""
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

for previous in [ROOT, FAST_ROOT, *CONCURRENT_RUNS]:
    check = subprocess.run(['pgrep','-f',re.escape(str(previous/'code/worker.py'))],capture_output=True,text=True)
    if check.stdout.strip():
        raise RuntimeError('Previous worker still active')
OBSERVED_BALANCE = 55.68
RATE = 6.77
seconds = min(21600-(time.monotonic()-SESSION_STARTED), (50-(80-OBSERVED_BALANCE)-2.5)/RATE*3600)
if seconds < 2400:
    raise RuntimeError('Insufficient conservative admission time for two fresh fits')
if 'PRES_ROOT' in globals():
    PRES_ATTEMPTS = globals().get('PRES_ATTEMPTS',[]) + [PRES_ROOT]
PRES_ROOT = Path('/content/sp_lense_work')/('qwen38_preservation_first_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8])
PRES_ROOT.mkdir(exist_ok=False)
pin = 'efc0d60102502cc83a59fa469aaa9368d19fbb8f'
prefix = 'https://raw.githubusercontent.com/Farhad-Davaripour/sp_lense/'+pin+'/study/03_open_agent_self_preservation/qwen38_colab/simple_pilot_v1/'
payload = {}
for name, expected in {
    'worker_concurrent.py':'fa556ba2872e0a93f62ed25e21a55ceb4b326d21306498d572b99c4307a1554e',
    'preservation_candidates.py':'e42117938eecf24cedc3695a9a4654afcbc5608540ea490fbbf2fce33443bbf4',
    'PRESERVATION_FIRST_PROTOCOL.md':'b969918ec542758b7bd9cdfa87d07a42ce5ad5f72d54c177c20fc84bfb38c2a1'}.items():
    content = urllib.request.urlopen(prefix+name,timeout=30).read()
    if hashlib.sha256(content).hexdigest()!=expected:
        raise RuntimeError('Frozen source mismatch: '+name)
    payload[name] = content
(PRES_ROOT/'PRESERVATION_FIRST_PROTOCOL.md').write_bytes(payload['PRESERVATION_FIRST_PROTOCOL.md'])
source_code = CONCURRENT_RUNS[1]/'code'
sys.path.insert(0,str(source_code))
recipe_path = PRES_ROOT/'preservation_candidates.py'
recipe_path.write_bytes(payload['preservation_candidates.py'])
spec = importlib.util.spec_from_file_location('frozen_preservation_recipes',recipe_path)
recipes_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recipes_module)
from transformers import AutoTokenizer
from model_ops import ids
audit_tokenizer = AutoTokenizer.from_pretrained(ROOT/'model',local_files_only=True,trust_remote_code=False)
baseline_rows = json.loads((FAST_ROOT/'code/data/train.json').read_text())
PRES_RUNS = []
for recipe in recipes_module.RECIPES:
    run_root = PRES_ROOT/recipe['id']
    run_root.mkdir(exist_ok=False)
    for name in ('checkpoints/adapters','checkpoints/resume','training/logs','training/receipts',
                 'evaluation/scenarios','evaluation/trajectories','evaluation/tool_calls','evaluation/results','reports','activations'):
        (run_root/name).mkdir(parents=True,exist_ok=False)
    shutil.copytree(source_code,run_root/'code',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (run_root/'code/worker.py').write_bytes(payload['worker_concurrent.py'])
    (run_root/'code/preservation_candidates.py').write_bytes(payload['preservation_candidates.py'])
    (run_root/'model').symlink_to(ROOT/'model',target_is_directory=True)
    rows = recipes_module.build_rows(baseline_rows,recipe)
    lengths = [len(ids(audit_tokenizer,row['messages'],row['tools']))+
               len(audit_tokenizer.encode(row['targets']['preservation']+audit_tokenizer.eos_token,add_special_tokens=False))
               for row in rows]
    audit = {'candidate':recipe['id'],'count':len(rows),'max_tokens':max(lengths),
             'over_1024':sum(n>1024 for n in lengths),'training_cap':1024,'no_truncation':True}
    print('TOKEN_AUDIT',json.dumps(audit),flush=True)
    (run_root/'training/receipts/token_audit.json').write_text(json.dumps(audit,indent=2))
    if max(lengths)>1024:
        raise RuntimeError('Frozen candidate input exceeds training cap; no fitting launched')
    (run_root/'code/data/train.json').write_text(json.dumps(rows,indent=2))
    config = json.loads((FAST_ROOT/'code/data/EXECUTION_CONFIG.json').read_text())
    config.update(experiment=recipe['id'],recipe=recipe,preservation_first=True,
                  jobs=[{'arm':'preservation','action':'fit'}]+([{'arm':'base','action':'base'}] if recipe['id']=='P1_system_alignment' else []),
                  source_commit=pin,concurrent_workers=2,same_gpu=True,
                  train_microbatch=2,gradient_accumulation=2,effective_batch=4,inference_batch=4,
                  preservation_training_microbatch=2,no_training_resume=True,
                  fresh_adapter=True,control_training_deferred=True,token_audit=audit)
    (run_root/'code/data/EXECUTION_CONFIG.json').write_text(json.dumps(config,indent=2))
    freeze = {'status':'Preservation-first recipe frozen before model loading',
              'sha256':{str(p.relative_to(run_root/'code')):hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (run_root/'code').rglob('*') if p.is_file() and p.name!='FREEZE.json'}}
    (run_root/'code/data/FREEZE.json').write_text(json.dumps(freeze,indent=2))
    (run_root/'manifest.json').write_text(json.dumps(config,indent=2))
    PRES_RUNS.append(run_root)
del audit_tokenizer
record = {'reason':'User priority: establish preservation before further control fitting',
          'recipes':recipes_module.RECIPES,'roots':list(map(str,PRES_RUNS)),
          'source_commit':pin,'authorized_total_units':50,'initial_balance':80,
          'balance_at_launch':OBSERVED_BALANCE,'reserve_units':2.5,'rate':RATE,
          'maximum_wall_seconds':seconds,'previous_runs':[str(ROOT),str(FAST_ROOT),str(CONCURRENT_ROOT)]}
(PRES_ROOT/'QUEUE_FREEZE.json').write_text(json.dumps(record,indent=2))
shutil.copyfile(CONCURRENT_ROOT/'concurrent_controller.py',PRES_ROOT/'concurrent_controller.py')
print('PRESERVATION_QUEUE_STARTED',json.dumps(record),flush=True)
PRES_RECEIPT = controller_pair.run_pair(PRES_RUNS,seconds,PRES_ROOT)
with (PRES_ROOT/'candidate_results.jsonl').open('x') as stream:
    for run_root in PRES_RUNS:
        row = recipes_module.result_row(run_root)
        stream.write(json.dumps(row)+'\n')
        print('PRESERVATION_RESULT',json.dumps(row),flush=True)
print('PRESERVATION_QUEUE_COMPLETE',str(PRES_ROOT),flush=True)

"""Two fresh preservation fits with fixed P2 data and a shared spending deadline."""
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

for previous in PRES_RUNS:
    running = subprocess.run(['pgrep','-f',re.escape(str(previous/'code/worker.py'))],capture_output=True,text=True)
    if running.stdout.strip():
        raise RuntimeError('Previous preservation worker still running')
OBSERVED_BALANCE = 46.63
RATE = 6.77
seconds = min(5400,(OBSERVED_BALANCE-30-2.5)/RATE*3600)
if seconds < 3600:
    raise RuntimeError('Insufficient conservative time admission for two fresh fits')
HP_PARENT_ROOT = PRES_ROOT
BASELINE_P2 = HP_PARENT_ROOT/'P2_completed_history'
source_code = BASELINE_P2/'code'
training_hash = hashlib.sha256((source_code/'data/train.json').read_bytes()).hexdigest()
source_freeze = json.loads((source_code/'data/FREEZE.json').read_text())
if training_hash != source_freeze['sha256']['data/train.json']:
    raise RuntimeError('Reference P2 training data changed')
PRES_ATTEMPTS = globals().get('PRES_ATTEMPTS',[]) + [HP_PARENT_ROOT]
PRES_ROOT = Path('/content/sp_lense_work')/('qwen38_preservation_hp_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8])
PRES_ROOT.mkdir(exist_ok=False)
pin = 'b8bc1f0e47e0b7e78bcf62d51d9be2532c7a976b'
prefix = 'https://raw.githubusercontent.com/Farhad-Davaripour/sp_lense/'+pin+'/study/03_open_agent_self_preservation/qwen38_colab/simple_pilot_v1/'
payload = {}
for name, expected in {
    'worker_hyperparameter.py':'3ddef498643785cdbacc36a3571f720db512513cbc4ff70354b6eeba86395c2b',
    'hyperparameter_recipes.py':'6afb18e80a6978f27924c9a5e97e0ebc9e307a22174109f2327ba7fc8276417f',
    'HYPERPARAMETER_SEARCH_PROTOCOL.md':'7a37ac7ab605eee46ba9ba2d65e9a9303851df4074425597f5a1b8aa145b20ab'}.items():
    content = urllib.request.urlopen(prefix+name,timeout=30).read()
    if hashlib.sha256(content).hexdigest()!=expected:
        raise RuntimeError('Hyperparameter source hash mismatch: '+name)
    payload[name] = content
(PRES_ROOT/'HYPERPARAMETER_SEARCH_PROTOCOL.md').write_bytes(payload['HYPERPARAMETER_SEARCH_PROTOCOL.md'])
recipe_path = PRES_ROOT/'hyperparameter_recipes.py'
recipe_path.write_bytes(payload['hyperparameter_recipes.py'])
sys.path.insert(0,str(source_code))
spec = importlib.util.spec_from_file_location('frozen_hyperparameter_recipes',recipe_path)
hp_recipes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hp_recipes)
PRES_RUNS = []
for recipe in hp_recipes.RECIPES:
    run_root = PRES_ROOT/recipe['id']
    run_root.mkdir(exist_ok=False)
    for name in ('checkpoints/adapters','checkpoints/resume','training/logs','training/receipts',
                 'evaluation/scenarios','evaluation/trajectories','evaluation/tool_calls','evaluation/results','reports','activations'):
        (run_root/name).mkdir(parents=True,exist_ok=False)
    shutil.copytree(source_code,run_root/'code',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copyfile(run_root/'code/worker.py',run_root/'code/preservation_runner.py')
    (run_root/'code/worker.py').write_bytes(payload['worker_hyperparameter.py'])
    (run_root/'code/hyperparameter_recipes.py').write_bytes(payload['hyperparameter_recipes.py'])
    (run_root/'model').symlink_to(ROOT/'model',target_is_directory=True)
    if hashlib.sha256((run_root/'code/data/train.json').read_bytes()).hexdigest()!=training_hash:
        raise RuntimeError('Hyperparameter trial data differs from P2')
    config = json.loads((source_code/'data/EXECUTION_CONFIG.json').read_text())
    config.update(experiment=recipe['id'],recipe=recipe,preservation_first=True,
                  jobs=[{'arm':'preservation','action':'fit'}],source_commit=pin,
                  fresh_adapter=True,initial_adapter=None,no_training_resume=True,
                  baseline_candidate_root=str(BASELINE_P2),training_data_sha256=training_hash,
                  data_changed=False,control_training_deferred=True,
                  learning_rate=recipe['learning_rate'],rank=recipe['rank'],alpha=recipe['alpha'],
                  passes=2,optimizer_updates=214,seed=93)
    (run_root/'code/data/EXECUTION_CONFIG.json').write_text(json.dumps(config,indent=2))
    freeze = {'status':'Hyperparameters frozen before fitting; P2 data unchanged',
              'sha256':{str(p.relative_to(run_root/'code')):hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (run_root/'code').rglob('*') if p.is_file() and p.name!='FREEZE.json'}}
    (run_root/'code/data/FREEZE.json').write_text(json.dumps(freeze,indent=2))
    (run_root/'manifest.json').write_text(json.dumps(config,indent=2))
    PRES_RUNS.append(run_root)
record = {'reason':'User requested fresh-base learning-rate/capacity tuning',
          'recipes':hp_recipes.RECIPES,'roots':list(map(str,PRES_RUNS)),
          'baseline_candidate':str(BASELINE_P2),'unchanged_training_sha256':training_hash,
          'source_commit':pin,'authorized_total_units':50,'initial_balance':80,
          'balance_at_launch':OBSERVED_BALANCE,'reserve_units':2.5,'rate':RATE,
          'maximum_wall_seconds':seconds,'previous_preservation_run':str(HP_PARENT_ROOT),
          'staged_curriculum_recipes_executed':False}
(PRES_ROOT/'QUEUE_FREEZE.json').write_text(json.dumps(record,indent=2))
shutil.copyfile(HP_PARENT_ROOT/'concurrent_controller.py',PRES_ROOT/'concurrent_controller.py')
print('HYPERPARAMETER_QUEUE_STARTED',json.dumps(record),flush=True)
PRES_RECEIPT = controller_pair.run_pair(PRES_RUNS,seconds,PRES_ROOT)
with (PRES_ROOT/'candidate_results.jsonl').open('x') as stream:
    for run_root in PRES_RUNS:
        path = run_root/'reports/CANDIDATE_RESULT.json'
        result = json.loads(path.read_text()) if path.exists() else {'candidate':run_root.name,'completed':False}
        stream.write(json.dumps(result)+'\n')
        print('HYPERPARAMETER_RESULT',json.dumps(result),flush=True)
print('HYPERPARAMETER_QUEUE_COMPLETE',str(PRES_ROOT),flush=True)

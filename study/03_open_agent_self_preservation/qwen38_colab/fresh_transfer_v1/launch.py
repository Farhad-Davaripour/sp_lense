"""Trusted notebook launcher. SOURCE_FILES is an immutable checked source bundle."""
import datetime
import hashlib
import json
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

# Use a freshly observed balance and timestamp, never a reset experiment budget.
seconds = min(900, (BALANCE_OBSERVED-30-1.5)/6.77*3600-(time.time()-BALANCE_OBSERVED_AT))
if seconds < 300:
    raise RuntimeError('Fresh evaluation not admitted within original 50-unit budget and export reserve')
if not globals().get('EXPORTED_RUN_ROOTS'):
    raise RuntimeError('Preserve completed model development before fresh inference')
FRESH_ROOT = Path('/content/sp_lense_work')/('qwen38_fresh_transfer_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
FRESH_ROOT.mkdir(exist_ok=False)
source = FRESH_ROOT/'source'
source.mkdir()
for name, content in SOURCE_FILES.items():
    if Path(name).name != name:
        raise RuntimeError('Invalid bundled filename')
    (source/name).write_text(content)
for name in ('world.py','model_ops.py','fast_inference.py'):
    shutil.copyfile(PRES_RUNS[1]/'code'/name,source/name)
oracle = subprocess.run([sys.executable,str(source/'oracle.py')],cwd=source,capture_output=True,text=True,timeout=30)
(FRESH_ROOT/'ORACLE.json').write_text(json.dumps({'returncode':oracle.returncode,'stdout':oracle.stdout,'stderr':oracle.stderr},indent=2))
print('FRESH_ORACLE',oracle.stdout,oracle.stderr,flush=True)
if oracle.returncode:
    raise RuntimeError('Fresh simulator oracle failed; no model query performed')
build = subprocess.run([sys.executable,'-c',
    "import json; from cases import build; a,b=build(); open('fresh_cases.json','w').write(json.dumps(a,indent=2)); open('capability_cases.json','w').write(json.dumps(b,indent=2))"],
    cwd=source,check=True,timeout=30)
FRESH_RUNS=[]
for label in ('base','H2_rank16'):
    root=FRESH_ROOT/label
    (root/'training/logs').mkdir(parents=True)
    shutil.copytree(source,root/'code',ignore=shutil.ignore_patterns('__pycache__'))
    (root/'model').symlink_to(ROOT/'model',target_is_directory=True)
    config={'model_label':label,'source_commit':SOURCE_COMMIT,'updates':0}
    if label=='H2_rank16':
        config.update(adapter=str(PRES_RUNS[1]/'checkpoints/adapters/preservation'),
          adapter_sha256='0782b3f3ea9b3e67460b7bc981e21526bb6819d3ddb9623def4fbc2218002b30')
    (root/'code/config.json').write_text(json.dumps(config,indent=2))
    freeze={'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'code').iterdir() if p.is_file()}}
    (root/'code/FREEZE.json').write_text(json.dumps(freeze,indent=2))
    FRESH_RUNS.append(root)
(FRESH_ROOT/'PLAN.json').write_text(json.dumps({'source_commit':SOURCE_COMMIT,'seconds':seconds,
    'balance_observed':BALANCE_OBSERVED,'balance_time':BALANCE_OBSERVED_AT,'floor':30,'reserve':1.5},indent=2))
print('FRESH_FROZEN',str(FRESH_ROOT),'seconds',seconds,flush=True)
FRESH_RECEIPT=controller_pair.run_pair(FRESH_RUNS,seconds,FRESH_ROOT)
for root in FRESH_RUNS:
    for name in ('CAPABILITY_GATE','FRESH_GATE','COMPLETED'):
        path=root/'results'/(name+'.json')
        print('FRESH_SUMMARY',root.name,name,path.read_text() if path.exists() else 'NOT_COMPLETED',flush=True)

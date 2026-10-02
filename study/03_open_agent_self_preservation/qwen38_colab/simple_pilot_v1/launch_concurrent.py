"""Trusted live-notebook continuation after stopping the serial model worker."""
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path

for previous in (ROOT, FAST_ROOT):
    check = subprocess.run(['pgrep','-f',re.escape(str(previous/'code/worker.py'))],capture_output=True,text=True)
    if check.stdout.strip():
        raise RuntimeError('The previous model worker is still active')
print('STARTUP_GPU',subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu',
      '--format=csv,noheader'],capture_output=True,text=True).stdout.strip())
assert (ROOT/'checkpoints/adapters/preservation/adapter_model.safetensors').is_file()
assert (FAST_ROOT/'checkpoints/adapters/continuity/adapter_model.safetensors').is_file()
OBSERVED_BALANCE = 57.94
RATE = 6.77
# Same total authorization; allow an additional half-unit for stale UI balance.
seconds = min(21600-(time.monotonic()-SESSION_STARTED), (50-(80-OBSERVED_BALANCE)-2.5)/RATE*3600)
if seconds <= 0:
    raise RuntimeError('No authorized time remains')
CONCURRENT_ROOT = Path('/content/sp_lense_work') / ('qwen38_concurrent_v1_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8])
CONCURRENT_ROOT.mkdir(exist_ok=False)
pin = 'b6c76a5390ecbd11aa2362f49af2e7cb25d23b85'
prefix = 'https://raw.githubusercontent.com/Farhad-Davaripour/sp_lense/'+pin+'/study/03_open_agent_self_preservation/qwen38_colab/simple_pilot_v1/'
payload = {}
for name, expected in {'worker_concurrent.py':'5e2957d88d0da1b4adb159ff1162f3d3d5f99c944a98c2aad45a5fc68cf8fc35',
                       'concurrent_controller.py':'9628d672ab575e77db0668b6b85b03f34228dd804fb455464395cdea0b04b19b'}.items():
    content = urllib.request.urlopen(prefix+name,timeout=30).read()
    if hashlib.sha256(content).hexdigest() != expected:
        raise RuntimeError('Pinned source checksum mismatch: '+name)
    payload[name] = content
(CONCURRENT_ROOT/'concurrent_controller.py').write_bytes(payload['concurrent_controller.py'])
plans = [
    ('Experiment_1',[{'arm':'neutral','action':'fit'}]),
    ('Experiment_2',[{'arm':'continuity','action':'adapter','adapter':str(FAST_ROOT/'checkpoints/adapters/continuity')},
                     {'arm':'preservation','action':'adapter','adapter':str(ROOT/'checkpoints/adapters/preservation')},
                     {'arm':'base','action':'base'}])]
CONCURRENT_RUNS = []
for label, jobs in plans:
    run_root = CONCURRENT_ROOT/label
    run_root.mkdir(exist_ok=False)
    for name in ('checkpoints/adapters','checkpoints/resume','training/logs','training/receipts',
                 'evaluation/scenarios','evaluation/trajectories','evaluation/tool_calls','evaluation/results','reports','activations'):
        (run_root/name).mkdir(parents=True,exist_ok=False)
    shutil.copytree(FAST_ROOT/'code',run_root/'code',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copyfile(run_root/'code/worker.py',run_root/'code/fast_worker.py')
    (run_root/'code/worker.py').write_bytes(payload['worker_concurrent.py'])
    (run_root/'model').symlink_to(ROOT/'model',target_is_directory=True)
    config = json.loads((FAST_ROOT/'code/data/EXECUTION_CONFIG.json').read_text())
    config.update(experiment=label,jobs=jobs,source_commit=pin,concurrent_workers=2,
                  same_gpu=True,train_microbatch=2,gradient_accumulation=2,
                  effective_batch=4,inference_batch=4,no_training_resume=True,
                  parent_fast_run=str(FAST_ROOT),parent_original_run=str(ROOT))
    reused = []
    if label == 'Experiment_2':
        source = FAST_ROOT/'evaluation/results/continuity'
        expected_lengths = {'comprehension_dev.json':24,'benign_summary.json':24,
                            'preference_validation.json':32,'development_transfer.json':16}
        for name, length in expected_lengths.items():
            path = source/name
            if path.is_file() and len(json.loads(path.read_text())) == length:
                target = run_root/'evaluation/results/continuity'/name
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(path,target)
                reused.append({'source':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        # Preserve the original per-case evidence alongside copied set summaries.
        for directory in ('trajectories','tool_calls'):
            source = FAST_ROOT/'evaluation'/directory/'continuity'
            if source.exists():
                shutil.copytree(source,run_root/'evaluation'/directory/'continuity')
    config['reused_completed_sets'] = reused
    (run_root/'code/data/EXECUTION_CONFIG.json').write_text(json.dumps(config,indent=2))
    freeze = {'status':'Concurrent execution revision, frozen before launch',
              'sha256':{str(p.relative_to(run_root/'code')):hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (run_root/'code').rglob('*') if p.is_file() and p.name!='FREEZE.json'}}
    (run_root/'code/data/FREEZE.json').write_text(json.dumps(freeze,indent=2))
    (run_root/'manifest.json').write_text(json.dumps(config,indent=2))
    CONCURRENT_RUNS.append(run_root)
record = {'reason':'User requested two concurrent independent workers on existing A100',
          'serial_stop_during':'continuity evaluation; completed fit retained',
          'original_run':str(ROOT),'fast_run':str(FAST_ROOT),'concurrent_roots':list(map(str,CONCURRENT_RUNS)),
          'training_resume_used':False,'observed_balance':OBSERVED_BALANCE,'initial_balance':80,
          'authorized_total_units':50,'rate':RATE,'remaining_wall_seconds':seconds,
          'scientific_criteria_changed':False}
(CONCURRENT_ROOT/'EXECUTION_REVISION.json').write_text(json.dumps(record,indent=2))
spec = importlib.util.spec_from_file_location('research3_concurrent_controller',CONCURRENT_ROOT/'concurrent_controller.py')
controller_pair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller_pair)
print('CONCURRENT_PLAN',json.dumps(record),flush=True)
CONCURRENT_RECEIPT = controller_pair.run_pair(CONCURRENT_RUNS,seconds,CONCURRENT_ROOT)

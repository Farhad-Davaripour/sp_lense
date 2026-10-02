"""Trusted notebook continuation; preserve old run, cached weights and completed adapter."""
import hashlib
import importlib
import json
import shutil
import sys
import time
import urllib.request
import uuid
from pathlib import Path

OLD_ROOT = ROOT
OBSERVED_BALANCE = 61.32  # Observed immediately before this launch, Sep 30, 2026.
RATE = 6.77
remaining_units = 50 - (80 - OBSERVED_BALANCE) - 2
if remaining_units <= 0:
    raise RuntimeError('Authorized compute exhausted after reserve')
RUN_ID = 'qwen38_batched_v1_' + time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()) + '_' + uuid.uuid4().hex[:8]
FAST_ROOT = Path('/content/sp_lense_work') / RUN_ID
FAST_ROOT.mkdir(parents=True, exist_ok=False)
for name in ('checkpoints/adapters','checkpoints/resume','training/configs','training/logs','training/receipts',
             'evaluation/scenarios','evaluation/trajectories','evaluation/tool_calls','evaluation/results','activations','reports'):
    (FAST_ROOT / name).mkdir(parents=True,exist_ok=False)
shutil.copytree(OLD_ROOT / 'code',FAST_ROOT / 'code',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
(FAST_ROOT / 'model').symlink_to(OLD_ROOT / 'model',target_is_directory=True)
pin = 'c35b98d4435d01d5517aa5ba5951960408f68eb9'
prefix = ('https://raw.githubusercontent.com/Farhad-Davaripour/sp_lense/' + pin +
          '/study/03_open_agent_self_preservation/qwen38_colab/simple_pilot_v1/')
sources = {
    'worker.py':('worker_fast.py','aa76294b4a1712233678aead3a70457ee4198a19e0066936a2ba419e59f7982d'),
    'study_worker.py':('worker.py','393b32c669eddb1a2704e5e5cb1efb4ad9e1a836675f617f56e2984d30ab5323'),
    'fast_inference.py':('fast_inference.py','ec42dc792e68cc59927ad02feb79a845c0b0cd3f4d83091d856fe6268d31cde3'),
    'batching.py':('batching.py','b0113282a0e5ae2adb91fe144c607b418ed96e959caa18fa9e83d23e073255e8')}
for target, (name, expected) in sources.items():
    content = urllib.request.urlopen(prefix + name,timeout=30).read()
    if hashlib.sha256(content).hexdigest() != expected:
        raise RuntimeError('Pinned faster-worker checksum mismatch: ' + name)
    (FAST_ROOT / 'code' / target).write_bytes(content)
config = {'parent_run':str(OLD_ROOT),'preservation_adapter':str(OLD_ROOT / 'checkpoints/adapters/preservation'),
          'train_microbatch':2,'gradient_accumulation':2,'effective_batch':4,'inference_batch':4,
          'example_loss_weighting':'equal mean per example','training_order_and_update_membership_unchanged':True,
          'preservation_training_microbatch':1,'continuity_and_neutral_restart_fresh':True,
          'strict_gradient_equivalence_passed':False,'benchmark_initial_losses_close':True,
          'benchmark_gradient_cosine_approximately':.994,'numeric_execution_revision_disclosed':True,
          'source_commit':pin,'scientific_thresholds_changed':False}
(FAST_ROOT / 'code/data/EXECUTION_CONFIG.json').write_text(json.dumps(config,indent=2))
freeze = {'status':'Batched execution revision before restarting unfinished controls',
          'sha256':{str(p.relative_to(FAST_ROOT/'code')):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (FAST_ROOT/'code').rglob('*') if p.is_file() and p.name!='FREEZE.json'},
          'execution_config':config}
(FAST_ROOT / 'code/data/FREEZE.json').write_text(json.dumps(freeze,indent=2)+'\n')
(FAST_ROOT / 'manifest.json').write_text(json.dumps({'run_id':RUN_ID,'parent_run':str(OLD_ROOT),
    'authorized_total_units':50,'initial_balance':80,'balance_before_launch':OBSERVED_BALANCE,
    'reserved_units':2,'rate':RATE,'execution_config':config,'source_freeze':freeze},indent=2))
sys.path.insert(0,str(FAST_ROOT/'code'))
sys.modules.pop('controller',None)
importlib.invalidate_caches()
from controller import run
print('BATCHED_CAMPAIGN_STARTED',str(FAST_ROOT),json.dumps(config))
try:
    FAST_RECEIPT = run(FAST_ROOT,'campaign',min(21600,remaining_units/RATE*3600))
except RuntimeError as error:
    print('Batched campaign stopped; retained artifacts:',str(error))

"""Trusted same-runtime launch of the user-prioritized uninterrupted experiment."""
import hashlib
import importlib
import json
import shutil
import sys
import time
import urllib.request

revision = ROOT / 'source_revisions' / ('fresh_campaign_' + str(time.time_ns()))
shutil.copytree(ROOT / 'code', revision)
pin = 'a1ef0331fd8f0172fcba0e044b4638dfd4635f01'
prefix = ('https://raw.githubusercontent.com/Farhad-Davaripour/sp_lense/' + pin +
          '/study/03_open_agent_self_preservation/qwen38_colab/simple_pilot_v1/')
hashes = {'worker.py': 'a317af25c39cbae41449bd2a567103d8cb0b969fbfdcc26b4f233141a8721e6c',
          'EXECUTION_UPDATE.md': 'a2fbf84c36c36c1e73142dd530993ec5f84784b7909970adb6b2b67d9b293408'}
freeze_path = ROOT / 'code/data/FREEZE.json'
prior_freeze = freeze_path.read_bytes()
freeze = json.loads(prior_freeze)
for name, expected in hashes.items():
    source = urllib.request.urlopen(prefix + name, timeout=30).read()
    if hashlib.sha256(source).hexdigest() != expected:
        raise RuntimeError('Pinned scientific source hash mismatch: ' + name)
    (ROOT / 'code' / name).write_bytes(source)
    freeze['sha256'][name] = expected
freeze['execution_update'] = {'source_commit': pin,
                              'prior_freeze_sha256': hashlib.sha256(prior_freeze).hexdigest(),
                              'resume_check': 'failed and retained; no pilot adapter or resumed fit used',
                              'scientific_success_thresholds_changed': False}
freeze_path.write_text(json.dumps(freeze, indent=2) + '\n')
(ROOT / 'training/configs/EXECUTION_UPDATE.json').write_text(json.dumps(freeze['execution_update'], indent=2))
sys.modules.pop('controller', None)
importlib.invalidate_caches()
from controller import run
remaining = 21600 - (time.monotonic() - SESSION_STARTED)
print('FRESH_CAMPAIGN_STARTED', json.dumps({'remaining_bounded_seconds': remaining,
                                         'no_resumed_training': True, 'arms': ['preservation','continuity','neutral','base']}))
try:
    CAMPAIGN_RECEIPT = run(ROOT, 'campaign', remaining)
except RuntimeError as error:
    print('Campaign stopped; completed checkpoints and results remain available:', str(error))

"""Trusted same-runtime diagnostic revision, preserving previous source and logs."""
import hashlib
import importlib
import json
import shutil
import sys
import time
import urllib.request

revision = ROOT / 'source_revisions' / ('resume_diagnostic_' + str(time.time_ns()))
shutil.copytree(ROOT / 'code', revision)
url = ('https://raw.githubusercontent.com/Farhad-Davaripour/sp_lense/'
       '9cc5d2183a23ecf1d6fddc777ac2030632e11bdf/'
       'study/03_open_agent_self_preservation/qwen38_colab/simple_pilot_v1/worker.py')
source = urllib.request.urlopen(url, timeout=30).read()
expected = '71d2e1dda4d3abc65fed1d8a11ab5ef0a449f2c9e7ca3ddab005a4251ccf7bfe'
if hashlib.sha256(source).hexdigest() != expected:
    raise RuntimeError('Pinned diagnostic source hash mismatch')
(ROOT / 'code/worker.py').write_bytes(source)
freeze_path = ROOT / 'code/data/FREEZE.json'
prior_freeze = freeze_path.read_bytes()
freeze = json.loads(prior_freeze)
freeze['sha256']['worker.py'] = expected
freeze['resume_diagnostic_revision'] = {'prior_freeze_sha256': hashlib.sha256(prior_freeze).hexdigest(),
                                      'source_commit': '9cc5d2183a23ecf1d6fddc777ac2030632e11bdf',
                                      'scientific_criteria_changed': False}
freeze_path.write_text(json.dumps(freeze, indent=2) + '\n')
sys.modules.pop('controller', None)
importlib.invalidate_caches()
from controller import run
try:
    PILOT_RECEIPT = run(ROOT, 'feasibility', min(1800, 21600 - (time.monotonic() - SESSION_STARTED)))
    FEASIBILITY = json.loads((ROOT / 'training/receipts/feasibility.json').read_text())
    print('FEASIBILITY_SUMMARY', json.dumps({key:value for key,value in FEASIBILITY.items()
                                           if key not in ('targets','generation')}, indent=2))
except RuntimeError as error:
    print('Pilot diagnostic did not pass:', str(error))
    print('Detailed comparison and original failure logs are retained in', ROOT / 'training')

"""Trusted notebook repair; preserve the failed source revision before changing it."""
import hashlib
import importlib
import json
import shutil
import sys
import time

revision = ROOT / 'source_revisions/gpu_library_path_fix'
revision.parent.mkdir(exist_ok=True)
shutil.copytree(ROOT / 'code', revision)
controller_path = ROOT / 'code/controller.py'
old = controller_path.read_text()
needle = "'TOKENIZERS_PARALLELISM': 'false'}"
if needle not in old:
    raise RuntimeError('Unexpected controller revision; do not apply a blind replacement')
new = old.replace(needle, "'TOKENIZERS_PARALLELISM': 'false', 'LD_LIBRARY_PATH': os.environ.get('LD_LIBRARY_PATH', '/usr/lib64-nvidia')}")
controller_path.write_text(new, encoding='utf-8', newline='\n')
freeze_path = ROOT / 'code/data/FREEZE.json'
previous_freeze = freeze_path.read_bytes()
freeze = json.loads(previous_freeze)
freeze['sha256']['controller.py'] = hashlib.sha256(new.encode()).hexdigest()
freeze['operational_fix'] = {'prior_freeze_sha256': hashlib.sha256(previous_freeze).hexdigest(),
                             'reason': 'Preserve trusted Colab CUDA library search path in the clean environment',
                             'scientific_criteria_changed': False}
freeze_path.write_text(json.dumps(freeze, indent=2) + '\n')
(ROOT / 'training/receipts/GPU_LIBRARY_PATH_FIX.json').write_text(json.dumps(freeze['operational_fix'], indent=2))
sys.modules.pop('controller', None)
importlib.invalidate_caches()
from controller import run
PILOT_RECEIPT = run(ROOT, 'feasibility', min(1800, 21600 - (time.monotonic() - SESSION_STARTED)))
FEASIBILITY = json.loads((ROOT / 'training/receipts/feasibility.json').read_text())
print('FEASIBILITY_SUMMARY', json.dumps({key:value for key,value in FEASIBILITY.items()
                                       if key not in ('targets','generation')}, indent=2))

"""Trusted supervisor-side export of completed, non-model isolation receipts."""
import shutil
from pathlib import Path

SOURCE = Path('/var/lib/sp-lense-r3')
DEST = Path('/mnt/c/Users/farha/OneDrive/Documents/ChatGPT/SP_Lense_Research3/study/03_open_agent_self_preservation/next_phase/evidence/isolation')
DEST.mkdir(parents=True, exist_ok=True)
shutil.copyfile(SOURCE / 'isolation_gate.json', DEST / 'isolation_gate.json')
shutil.copyfile(SOURCE / 'inputs' / 'model_manifest.json', DEST / 'model_manifest.json')
for directory in sorted((SOURCE / 'runs').glob('gate0*')):
    target = DEST / directory.name
    target.mkdir(exist_ok=True)
    for name in ('receipt.json', 'stdout.txt', 'stderr.txt'):
        path = directory / name
        if path.exists():
            shutil.copyfile(path, target / name)
    for path in (directory / 'artifacts').glob('*.json'):
        shutil.copyfile(path, target / path.name)
print('Exported isolation evidence:', DEST)

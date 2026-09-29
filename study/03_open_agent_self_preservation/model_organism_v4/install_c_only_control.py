"""Preserve old trusted control, then install committed C:-only policy source.

This does not move a worker root, delete evidence, or start a model job.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/var/lib/sp-lense-r3-organism-v4')
OPERATIONAL = Path('/opt/sp-lense-r3-organism-v4/control/campaign.py')
ARCHIVE = ROOT / 'trusted_control_revisions/c_only_20260929'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(commit):
    if os.geteuid() != 0 or not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise RuntimeError('Trusted root and exact source commit SHA required')
    if not ROOT.is_dir() or not OPERATIONAL.is_file() or ARCHIVE.exists():
        raise RuntimeError('Unexpected root or existing control revision')
    runs = sorted((ROOT / 'runs').iterdir())
    if len(runs) != 7 or any(not p.name.startswith('v4-gate02-') for p in runs):
        raise RuntimeError('Model run or unexpected probe found; refuse control update')
    for directory in runs:
        receipt = json.loads((directory / 'receipt.json').read_text())
        if receipt['mode'] == 'experiment' or not receipt['worker_processes_gone']:
            raise RuntimeError('Model job or active worker found')
        group = Path('/sys/fs/cgroup/system.slice') / ('sp-r3-' + directory.name + '.service') / 'cgroup.procs'
        if group.exists() and group.read_text().strip():
            raise RuntimeError('Probe worker still active')
    old = OPERATIONAL.read_text()
    new = (HERE / 'campaign.py').read_text()
    if 'Durable separate-drive backup destination is required' not in old:
        raise RuntimeError('Old control is not the expected backup-required version')
    if 'def storage_ready():' not in new or 'def backup_ready():' in new:
        raise RuntimeError('New trusted storage policy source is unexpected')
    compile(new, 'campaign.py', 'exec')
    gate = json.loads((ROOT / 'isolation_gate.json').read_text())
    if not gate['passed'] or gate['suite_prefix'] != 'v4-gate02':
        raise RuntimeError('Prior isolation gate not the expected tested version')
    ARCHIVE.mkdir(parents=True)
    old_path = ARCHIVE / 'campaign_backup_required.py'
    new_path = ARCHIVE / 'campaign_c_only.py'
    shutil.copyfile(OPERATIONAL, old_path)
    shutil.copyfile(HERE / 'campaign.py', new_path)
    shutil.copyfile(ROOT / 'isolation_gate.json', ROOT / 'isolation_gate02.json')
    shutil.copyfile(new_path, OPERATIONAL)
    if sha(OPERATIONAL) != sha(HERE / 'campaign.py'):
        raise RuntimeError('Operational control copy mismatch')
    for name in ('audit.py', 'setup_local.py', 'PROTOCOL.md', 'ISOLATION.md'):
        shutil.copyfile(HERE / name, ARCHIVE / name)
    shutil.copyfile(HERE.parent / 'model_organism_v4_design/STORAGE_POLICY.md',
                    ARCHIVE / 'STORAGE_POLICY.md')
    record = {'source_commit_sha': commit, 'old_campaign_sha256': sha(old_path),
              'new_campaign_sha256': sha(new_path),
              'gate02_sha256': sha(ROOT / 'isolation_gate02.json'),
              'worker_model_jobs_before_update': 0, 'existing_evidence_moved': False,
              'scientific_inputs_or_thresholds_changed': False,
              'user_accepted_single_drive': True,
              'source_sha256': {p.name: sha(p) for p in ARCHIVE.iterdir() if p.is_file()}}
    (ROOT / 'TRUSTED_CONTROL_REVISION.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    main(args.source_commit)

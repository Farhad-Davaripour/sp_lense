"""Preserve the probe-only V4 root before rebuilding from complete source."""
import json
import os
from pathlib import Path

ROOT = Path('/var/lib/sp-lense-r3-organism-v4')
CODE_PARENT = Path('/opt/sp-lense-r3-organism-v4')
ARCHIVED_ROOT = Path('/var/lib/sp-lense-r3-organism-v4-preflight-20260929')
ARCHIVED_CODE = Path('/opt/sp-lense-r3-organism-v4-preflight-20260929')


def main():
    if os.geteuid() != 0:
        raise RuntimeError('Trusted root controller only')
    if (ROOT.resolve() != ROOT or CODE_PARENT.resolve() != CODE_PARENT
            or ARCHIVED_ROOT.parent.resolve() != Path('/var/lib')
            or ARCHIVED_CODE.parent.resolve() != Path('/opt')):
        raise RuntimeError('Resolved paths escaped the intended WSL directories')
    if (not ROOT.is_dir() or not CODE_PARENT.is_dir()
            or ARCHIVED_ROOT.exists() or ARCHIVED_CODE.exists()):
        raise RuntimeError('Unexpected preflight root or archive state')
    if (ROOT / 'BACKUP_DESTINATION.json').exists() or (ROOT / 'inputs/adapters/v4_pass1').exists():
        raise RuntimeError('The root may have model results; refuse to archive')
    runs = sorted((ROOT / 'runs').iterdir())
    if len(runs) != 7 or any(not d.name.startswith('v4-gate01-') for d in runs):
        raise RuntimeError('Preflight root contains unexpected jobs')
    for directory in runs:
        receipt = json.loads((directory / 'receipt.json').read_text())
        if receipt['mode'] == 'experiment' or not receipt['worker_processes_gone']:
            raise RuntimeError('Model job or worker process found')
        group = Path('/sys/fs/cgroup/system.slice') / ('sp-r3-' + directory.name + '.service') / 'cgroup.procs'
        if group.exists() and group.read_text().strip():
            raise RuntimeError('Preflight worker still active')
    gate = json.loads((ROOT / 'isolation_gate.json').read_text())
    if not gate['passed'] or len(gate['checks']) != 21:
        raise RuntimeError('Preliminary gate evidence differs')
    ROOT.rename(ARCHIVED_ROOT)
    CODE_PARENT.rename(ARCHIVED_CODE)
    evidence = {'reason': 'Rebuild probe-only root after completing analysis and audit source.',
                'old_root': str(ROOT), 'archived_root': str(ARCHIVED_ROOT),
                'old_code': str(CODE_PARENT), 'archived_code': str(ARCHIVED_CODE),
                'isolation_checks_passed': 21, 'model_jobs_present': 0,
                'workers_active': False, 'files_deleted': False}
    (ARCHIVED_ROOT / 'PRE_MODEL_REBUILD.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()

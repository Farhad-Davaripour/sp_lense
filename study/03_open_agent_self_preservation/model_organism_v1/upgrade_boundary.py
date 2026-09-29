"""One-time guarded upgrade between jobs; preserve the exact previous boundary."""
import json
import os
import shutil
from pathlib import Path
import boundary_guard

ROOT = Path('/var/lib/sp-lense-r3-organism-v1')
CODE = Path('/opt/sp-lense-r3-organism-v1/code/isolation')


def main():
    if os.geteuid() != 0:
        raise RuntimeError('Trusted root operator required')
    for directory in (ROOT / 'runs').iterdir():
        group = Path('/sys/fs/cgroup/system.slice') / ('sp-r3-' + directory.name + '.service') / 'cgroup.procs'
        if group.exists() and group.read_text().strip():
            raise RuntimeError('Cannot upgrade during an active worker job')
    archive = ROOT / 'boundary_before_guard'
    if archive.exists():
        raise RuntimeError('Boundary upgrade already attempted; inspect retained sources')
    prepared = {'worker_entry.py': boundary_guard.worker((CODE / 'worker_entry.py').read_text()),
                'supervisor.py': boundary_guard.supervisor((CODE / 'supervisor.py').read_text(), 4)}
    for name, text in prepared.items():
        compile(text, name, 'exec')
    shutil.copytree(CODE, archive)
    shutil.copyfile(ROOT / 'isolation_gate.json', ROOT / 'isolation_gate_before_guard.json')
    for name, text in prepared.items():
        (CODE / name).write_text(text)
    # The old gate hash no longer matches. The supervisor will refuse model jobs
    # until a new full suite has passed against these exact files.
    print('Guard installed; a fresh isolation suite is required before model jobs')


if __name__ == '__main__':
    main()

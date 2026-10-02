"""Trusted model-free check of the facilities needed to port the local boundary."""
import ctypes.util
import json
import os
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path


def run(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=15)
        return {'exit_code': result.returncode, 'stdout': result.stdout[:8192],
                'stderr': result.stderr[:2048]}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'error': type(error).__name__}


def main():
    root = Path('/sys/fs/cgroup')
    status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines()
                  if ':' in line)
    cgroup = {'v2': (root / 'cgroup.controllers').exists()}
    for name in ('cgroup.controllers', 'cgroup.subtree_control', 'memory.max',
                 'memory.swap.max', 'cpu.max', 'pids.max'):
        path = root / name
        cgroup[name] = path.read_text().strip() if path.exists() else None
    temporary = root / ('sp-lense-readiness-' + uuid.uuid4().hex)
    try:
        temporary.mkdir()
        cgroup['child_creation'] = 'succeeded'
        cgroup['child_controllers'] = sorted(path.name for path in temporary.iterdir()
                                           if path.name in ('memory.max', 'cpu.max', 'pids.max'))
    except OSError as error:
        cgroup['child_creation'] = {'errno': error.errno, 'error': error.strerror}
    finally:
        if temporary.exists():
            temporary.rmdir()
    # Try mounting a *new*, private cgroup namespace. Failure is evidence about
    # kernel delegation; do not change the host cgroup hierarchy or limits.
    private_cgroup = run(['unshare', '--user', '--map-root-user', '--mount',
                         '--net', '--pid', '--fork', '/bin/sh', '-c',
                         'mount --make-rprivate / && mkdir -p /tmp/sp-lense-cgroup-probe '
                         '&& mount -t cgroup2 none /tmp/sp-lense-cgroup-probe'])
    result = {
        'utc': datetime.now(timezone.utc).isoformat(),
        'pid1_command': Path('/proc/1/comm').read_text().strip(),
        'capabilities': {name: status[name].strip() for name in ('CapEff', 'CapBnd', 'NoNewPrivs', 'Seccomp')},
        'cgroups': cgroup, 'private_cgroup_mount_probe': private_cgroup,
        'seccomp_library': ctypes.util.find_library('seccomp'),
        'bubblewrap': shutil.which('bwrap'),
        'package_candidates': run(['apt-cache', 'policy', 'bubblewrap', 'libseccomp2']),
        'parent_mounts_relevant_to_containment': [line for line in Path('/proc/self/mountinfo').read_text().splitlines()
                                                if any(part in line for part in ('/sys/fs/cgroup', '/content/drive', 'drivefs'))],
        'isolation_verified': False, 'model_worker_started': False,
    }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

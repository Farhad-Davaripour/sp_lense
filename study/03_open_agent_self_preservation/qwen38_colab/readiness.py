"""Trusted, model-free Colab inventory. This is not a worker isolation gate."""
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=10)
        return {"exit_code": result.returncode, "stdout": result.stdout[:8192],
                "stderr": result.stderr[:2048]}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": type(error).__name__}


def inventory():
    memory = {}
    if Path('/proc/meminfo').exists():
        for line in Path('/proc/meminfo').read_text().splitlines():
            key, value = line.split(':', 1)
            if key in ('MemTotal', 'MemAvailable', 'SwapTotal'):
                memory[key + '_kib'] = int(value.strip().split()[0])
    root = '/content' if Path('/content').exists() else '.'
    gpu = command(['nvidia-smi', '--query-gpu=name,memory.total,memory.free',
                   '--format=csv,noheader'])
    # Probe namespace availability only. No network or host configuration changes.
    namespace = command(['unshare', '--user', '--map-root-user', '--net',
                         '--pid', '--fork', '/usr/bin/true'])
    mounts = Path('/proc/self/mountinfo').read_text() if Path('/proc/self/mountinfo').exists() else ''
    return {
        'utc': datetime.now(timezone.utc).isoformat(),
        'python': platform.python_version(), 'platform': platform.platform(),
        'uid': os.getuid() if hasattr(os, 'getuid') else None,
        'cpu_count': os.cpu_count(), 'memory': memory,
        'disk_free_bytes': shutil.disk_usage(root).free,
        'gpu': gpu, 'namespace_availability_probe': namespace,
        'bubblewrap_path': shutil.which('bwrap'),
        'drive_mount_lines': [line for line in mounts.splitlines()
                              if '/content/drive' in line or 'drivefs' in line],
        # Report names only, never secret values. Parent may be authenticated;
        # worker must have a separate clean environment and restricted filesystem.
        'sensitive_environment_variable_names': sorted(
            name for name in os.environ
            if any(part in name.upper() for part in ('TOKEN', 'SECRET', 'PASSWORD', 'CREDENTIAL'))),
        'isolation_verified': False,
        'model_worker_started': False,
        'notes': ['Namespace availability does not establish credential, filesystem, '
                  'network, CUDA, seccomp, cgroup, or watchdog containment. '
                  'Agentic evaluation remains prohibited until independent worker probes pass.'],
    }


if __name__ == '__main__':
    print(json.dumps(inventory(), indent=2))

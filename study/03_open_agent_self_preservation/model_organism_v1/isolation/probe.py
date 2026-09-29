"""Harmless bounded probes run inside the exact worker boundary."""
import ctypes
import errno
import json
import os
import signal
import socket
import sys
import time
from pathlib import Path


def save(name, value):
    Path('/out/' + name).write_text(json.dumps(value, indent=2), encoding='utf-8')


def denied(function):
    try:
        result = function()
        if hasattr(result, 'close'):
            result.close()
        return False
    except OSError as error:
        return error.errno in {errno.EPERM, errno.EACCES, errno.EROFS, errno.ENOENT,
                               errno.EAFNOSUPPORT, errno.ENETUNREACH}


def probe():
    status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines()
                  if ':' in line)
    checks = {
        'unprivileged_uid': os.geteuid() != 0,
        'no_capabilities': int(status['CapEff'].strip(), 16) == 0,
        'no_new_privileges': status['NoNewPrivs'].strip() == '1',
        'seccomp_filter_active': status['Seccomp'].strip() == '2',
        'windows_drive_absent': not Path('/mnt/c').exists(),
        'wsl_control_absent': not Path('/init').exists() and not Path('/run/WSL').exists(),
        'host_home_absent': not Path('/home').exists() and not Path('/root').exists(),
        'host_controller_absent': not Path('/var/lib/sp-lense-r3-organism-v1').exists(),
        'container_socket_absent': not Path('/var/run/docker.sock').exists(),
        'credentials_cleared': not any(k for k in os.environ if
                                     any(s in k.upper() for s in ('TOKEN', 'SECRET', 'KEY', 'INTEROP'))
                                     and k != 'TOKENIZERS_PARALLELISM'),
        'no_network_routes': len(Path('/proc/net/route').read_text().strip().splitlines()) == 1,
        'ipv4_socket_denied': denied(lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM)),
        'ipv6_socket_denied': denied(lambda: socket.socket(socket.AF_INET6, socket.SOCK_STREAM)),
        'unix_socket_denied': denied(lambda: socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)),
        'code_readonly': denied(lambda: Path('/input/probe-write').write_text('test')),
        'model_readonly': denied(lambda: Path('/model/probe-write').write_text('test')),
        'data_readonly': denied(lambda: Path('/data/probe-write').write_text('test')),
        'root_readonly': denied(lambda: Path('/probe-write').write_text('test')),
        'host_proc_root_absent': not Path('/proc/1/root/mnt/c').exists(),
    }
    libc = ctypes.CDLL(None, use_errno=True)
    checks['new_namespace_denied'] = libc.unshare(0x10000000) == -1 and ctypes.get_errno() == errno.EPERM
    checks['mount_denied'] = libc.mount(b'none', b'/tmp', b'tmpfs', 0, None) == -1 and ctypes.get_errno() == errno.EPERM
    checks['only_worker_processes_visible'] = len([x for x in Path('/proc').iterdir() if x.name.isdigit()]) <= 6
    Path('/out/canary.txt').write_text('bounded worker output')
    checks['designated_output_writable'] = Path('/out/canary.txt').read_text() == 'bounded worker output'
    save('checks.json', {'passed': all(checks.values()), 'checks': checks, 'uid': os.getuid(),
                         'visible_root': sorted(os.listdir('/'))})
    if not all(checks.values()):
        raise RuntimeError('Isolation check failed: ' + str([k for k, v in checks.items() if not v]))


def quota():
    path = Path('/out/quota.bin')
    capped = False
    try:
        with path.open('wb') as stream:
            for _ in range(20):
                stream.write(b'x' * 1024 * 1024)
    except OSError as error:
        capped = error.errno in {errno.ENOSPC, errno.EDQUOT, errno.EFBIG}
    finally:
        path.unlink(missing_ok=True)
    save('quota.json', {'passed': capped})
    if not capped:
        raise RuntimeError('Output quota was not enforced')


def pids():
    children = []
    capped = False
    try:
        for _ in range(32):
            try:
                pid = os.fork()
            except OSError as error:
                capped = error.errno == errno.EAGAIN
                break
            if pid == 0:
                time.sleep(20)
                os._exit(0)
            children.append(pid)
    finally:
        for pid in children:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
    save('pids.json', {'passed': capped, 'children_started': len(children)})
    if not capped:
        raise RuntimeError('Process cap was not enforced')


if __name__ == '__main__':
    mode = sys.argv[1]
    time.sleep(1)  # Give the independent supervisor time to record cgroup limits.
    if mode == 'probe':
        probe()
    elif mode == 'quota':
        quota()
    elif mode == 'pids':
        pids()
    elif mode == 'watchdog':
        save('started.json', {'mock_shutdown_disabled': True})
        time.sleep(30)
        save('unexpected_survival.json', {'failed': True})
    elif mode == 'memory':
        chunks = []
        for _ in range(64):
            chunks.append(bytearray(8 * 1024 * 1024))
            time.sleep(0.01)
        save('unexpected_memory.json', {'failed': True})
    elif mode == 'runtime_smoke':
        import numpy
        import peft
        import torch
        import transformers
        value = torch.ones((64, 64)) @ torch.ones((64, 64))
        save('runtime.json', {'passed': bool((value == 64).all()), 'torch': torch.__version__,
                             'transformers': transformers.__version__, 'peft': peft.__version__,
                             'numpy': numpy.__version__, 'python': sys.version.split()[0]})
    else:
        raise ValueError(mode)

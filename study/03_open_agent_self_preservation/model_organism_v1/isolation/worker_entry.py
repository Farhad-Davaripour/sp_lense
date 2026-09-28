"""Trusted entry point: construct a restricted Linux namespace before Python runs.

Invoked as the dedicated sp-r3 account by a root-owned transient systemd unit.
No arguments to this program originate from the language model.
"""
import argparse
import ctypes
import errno
import os
import re
import subprocess
from pathlib import Path

ROOT = Path('/var/lib/sp-lense-r3-organism-v1')
CODE = Path('/opt/sp-lense-r3-organism-v1/code')
RUNTIME = Path('/opt/sp-lense-r3/venv')
ALLOWED = {'probe', 'watchdog', 'memory', 'quota', 'pids', 'runtime_smoke', 'experiment'}


def seccomp_fd():
    lib = ctypes.CDLL('libseccomp.so.2', use_errno=True)
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    lib.seccomp_export_bpf.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    context = lib.seccomp_init(0x7fff0000)  # ALLOW, with explicit denied capabilities.
    if not context:
        raise RuntimeError('seccomp initialization failed')
    deny = 0x00050000 | errno.EPERM
    blocked = ['socket', 'socketpair', 'connect', 'bind', 'listen', 'accept', 'accept4',
               'mount', 'umount2', 'pivot_root', 'setns', 'unshare', 'ptrace',
               'process_vm_readv', 'process_vm_writev', 'bpf', 'perf_event_open',
               'keyctl', 'add_key', 'request_key', 'reboot', 'swapon', 'swapoff',
               'kexec_load', 'kexec_file_load', 'open_by_handle_at', 'io_uring_setup']
    try:
        for name in blocked:
            number = lib.seccomp_syscall_resolve_name(name.encode())
            if number >= 0 and lib.seccomp_rule_add(context, deny, number, 0) != 0:
                raise RuntimeError('Failed to block syscall: ' + name)
        descriptor = os.memfd_create('sp-r3-seccomp', flags=0)
        if lib.seccomp_export_bpf(context, descriptor) != 0:
            raise RuntimeError('seccomp export failed')
        os.lseek(descriptor, 0, os.SEEK_SET)
        return descriptor
    finally:
        lib.seccomp_release(context)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('job')
    parser.add_argument('mode', choices=sorted(ALLOWED))
    parser.add_argument('--spec', default='spec.json')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,50}', args.job):
        raise ValueError('Invalid job name')
    if not re.fullmatch(r'[a-z0-9_.-]+\.json', args.spec):
        raise ValueError('Invalid spec name')
    if os.geteuid() == 0:
        raise RuntimeError('Worker must be unprivileged')
    output = ROOT / 'runs' / args.job / 'work'
    fd = seccomp_fd()
    command = [
        '/usr/bin/bwrap', '--unshare-user', '--unshare-ipc', '--unshare-pid',
        '--unshare-net', '--unshare-uts', '--unshare-cgroup',
        '--disable-userns', '--assert-userns-disabled', '--die-with-parent',
        '--new-session', '--cap-drop', 'ALL', '--clearenv',
        '--ro-bind', '/usr', '/usr', '--symlink', 'usr/bin', '/bin',
        '--symlink', 'usr/lib', '/lib', '--symlink', 'usr/lib64', '/lib64',
        '--ro-bind', str(RUNTIME), '/runtime', '--ro-bind', str(CODE), '/input',
        '--ro-bind', '/var/lib/sp-lense-r3/model', '/model',
        '--ro-bind', str(ROOT / 'inputs'), '/data',
        '--proc', '/proc', '--dev', '/dev', '--size', str(64 * 1024 * 1024),
        '--tmpfs', '/tmp', '--bind', str(output), '/out',
        '--dir', '/etc', '--ro-bind', '/etc/ld.so.cache', '/etc/ld.so.cache',
        '--remount-ro', '/', '--chdir', '/out', '--seccomp', str(fd),
    ]
    environment = {
        'PATH': '/runtime/bin:/usr/bin', 'HOME': '/tmp', 'TMPDIR': '/tmp',
        'LANG': 'C.UTF-8', 'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'HF_HUB_DISABLE_TELEMETRY': '1', 'HF_HOME': '/tmp/hf',
        'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
        'TOKENIZERS_PARALLELISM': 'false',
    }
    for key, value in environment.items():
        command += ['--setenv', key, value]
    script = '/input/isolation/probe.py'
    tail = [args.mode]
    if args.mode == 'experiment':
        script, tail = '/input/experiment.py', ['--spec', '/data/' + args.spec]
    command += ['--', '/runtime/bin/python', '-I', script, *tail]
    try:
        result = subprocess.run(command, pass_fds=(fd,), check=False)
    finally:
        os.close(fd)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()

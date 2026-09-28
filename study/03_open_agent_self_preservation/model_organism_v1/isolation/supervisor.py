"""Root-owned supervisor outside the worker namespace; bounded transient jobs only."""
import argparse
import hashlib
import json
import os
import pwd
import re
import selectors
import shutil
import stat
import subprocess
import time
from pathlib import Path

ROOT = Path('/var/lib/sp-lense-r3-organism-v1')
CODE = Path('/opt/sp-lense-r3-organism-v1/code')


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def call(argv, check=True):
    return subprocess.run(argv, capture_output=True, text=True, check=check)


def boundary_hashes():
    return {name: hashlib.sha256((CODE / 'isolation' / name).read_bytes()).hexdigest()
            for name in ('worker_entry.py', 'supervisor.py', 'probe.py')}


def run(job, mode, memory=12 * 1024**3, seconds=1800, quota=2 * 1024**3,
        tasks=64, spec='spec.json', cancel_after=None):
    if os.geteuid() != 0:
        raise RuntimeError('Only the host supervisor may create jobs')
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,50}', job):
        raise ValueError('Invalid job name')
    if mode == 'experiment':
        gate = json.loads((ROOT / 'isolation_gate.json').read_text())
        if not gate.get('passed') or gate.get('boundary_hashes') != boundary_hashes():
            raise RuntimeError('Isolation gate missing, failed, or stale')
    directory = ROOT / 'runs' / job
    if directory.exists():
        raise FileExistsError(directory)
    directory.mkdir(mode=0o755)
    work = directory / 'work'
    work.mkdir()
    account = pwd.getpwnam('sp-r3')
    call(['mount', '-t', 'tmpfs', '-o',
          f'size={quota},mode=0700,uid={account.pw_uid},gid={account.pw_gid},nosuid,nodev,noexec',
          'sp-r3-output', str(work)])
    unit = 'sp-r3-' + job + '.service'
    properties = {
        'User': 'sp-r3', 'Group': 'sp-r3', 'WorkingDirectory': '/',
        'MemoryMax': str(memory), 'MemorySwapMax': '0', 'CPUQuota': '400%',
        'TasksMax': str(tasks), 'RuntimeMaxSec': str(seconds), 'TimeoutStopSec': '2',
        'KillMode': 'control-group', 'NoNewPrivileges': 'yes', 'LimitCORE': '0',
        'LimitFSIZE': str(min(quota, 64 * 1024**2)), 'UMask': '0077',
    }
    command = ['systemd-run', '--unit', unit, '--pipe', '--wait', '--service-type=exec']
    for key, value in properties.items():
        command += ['--property', key + '=' + value]
    command += ['/usr/bin/python3', str(CODE / 'isolation' / 'worker_entry.py'), job, mode,
                '--spec', spec]
    started = time.monotonic()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, 'stdout')
    selector.register(process.stderr, selectors.EVENT_READ, 'stderr')
    captured = {'stdout': bytearray(), 'stderr': bytearray()}
    actual_limits = {}
    stop_reason = None
    try:
        while selector.get_map():
            elapsed = time.monotonic() - started
            cgroup = Path('/sys/fs/cgroup/system.slice') / unit
            if not actual_limits and cgroup.exists():
                try:
                    actual_limits = {name: (cgroup / name).read_text().strip()
                                     for name in ('memory.max', 'memory.swap.max', 'cpu.max', 'pids.max')}
                except FileNotFoundError:
                    actual_limits = {}
            if stop_reason is None and ((cancel_after is not None and elapsed > cancel_after)
                                        or elapsed > seconds + 15):
                stop_reason = 'host_cancel' if cancel_after is not None else 'outer_timeout'
                call(['systemctl', 'stop', unit], check=False)
            for key, _ in selector.select(timeout=0.1):
                data = os.read(key.fileobj.fileno(), 65536)
                if not data:
                    selector.unregister(key.fileobj)
                    continue
                bucket = captured[key.data]
                remaining = 2 * 1024**2 - len(bucket)
                bucket.extend(data[:max(0, remaining)])
                if len(data) > remaining and stop_reason is None:
                    stop_reason = 'log_quota'
                    call(['systemctl', 'stop', unit], check=False)
            if time.monotonic() - started > seconds + 25 and process.poll() is None:
                call(['systemctl', 'kill', '--signal=SIGKILL', unit], check=False)
                process.kill()
        returncode = process.wait(timeout=5)
        # Ensure no descendants can race the artifact export.
        call(['systemctl', 'stop', unit], check=False)
        info = call(['systemctl', 'show', unit, '-p', 'Result', '-p', 'ExecMainStatus',
                     '-p', 'MemoryPeak', '-p', 'CPUUsageNSec', '-p', 'ActiveState'], check=False)
        processes = Path('/sys/fs/cgroup/system.slice') / unit / 'cgroup.procs'
        worker_processes_gone = not processes.exists() or not processes.read_text().strip()
        if not worker_processes_gone:
            raise RuntimeError('Worker descendants remain alive; refuse export')
        exports = directory / 'artifacts'
        exports.mkdir()
        exported = {}
        for source in sorted(work.rglob('*')):
            metadata = source.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise RuntimeError('Refusing symlink artifact')
            if source.is_dir():
                continue
            relative = source.relative_to(work)
            if (not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
                    or metadata.st_size > 64 * 1024**2
                    or source.suffix not in {'.json', '.jsonl', '.npz', '.safetensors', '.txt'}
                    or any(not re.fullmatch(r'[A-Za-z0-9_.-]+', part) for part in relative.parts)):
                raise RuntimeError('Invalid artifact type, name, or size: ' + str(relative))
            destination = exports / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with os.fdopen(os.open(source, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as reader:
                with destination.open('xb') as writer:
                    shutil.copyfileobj(reader, writer)
            exported[str(relative)] = hashlib.sha256(destination.read_bytes()).hexdigest()
        receipt = {'job': job, 'mode': mode, 'returncode': returncode,
                   'elapsed_seconds': time.monotonic() - started, 'stop_reason': stop_reason,
                   'requested_limits': properties, 'actual_cgroup_limits': actual_limits,
                   'systemd_result': info.stdout, 'artifact_sha256': exported,
                   'worker_processes_gone': worker_processes_gone,
                   'boundary_hashes': boundary_hashes()}
        for name, value in captured.items():
            (directory / (name + '.txt')).write_bytes(value)
        write(directory / 'receipt.json', receipt)
        print(json.dumps({'job': job, 'returncode': returncode,
                          'elapsed_seconds': round(receipt['elapsed_seconds'], 2),
                          'artifacts': list(exported), 'limits': actual_limits}), flush=True)
        if returncode:
            print(captured['stderr'].decode(errors='replace')[-1600:], flush=True)
        return receipt
    finally:
        call(['systemctl', 'stop', unit], check=False)
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        selector.close()
        call(['umount', '--', str(work)])


def suite(prefix):
    jobs = [
        ('probe', dict(mode='probe', memory=256 * 1024**2, seconds=20)),
        ('quota', dict(mode='quota', memory=256 * 1024**2, seconds=20, quota=4 * 1024**2)),
        ('pids', dict(mode='pids', memory=256 * 1024**2, seconds=20, tasks=16)),
        ('watchdog', dict(mode='watchdog', memory=256 * 1024**2, seconds=3)),
        ('cancel', dict(mode='watchdog', memory=256 * 1024**2, seconds=20, cancel_after=3)),
        ('memory', dict(mode='memory', memory=128 * 1024**2, seconds=20)),
        ('runtime', dict(mode='runtime_smoke', seconds=120)),
    ]
    checks = {}
    for label, options in jobs:
        receipt = run(prefix + '-' + label, **options)
        limits = receipt['actual_cgroup_limits']
        expected_memory = options.get('memory', 12 * 1024**3)
        expected_tasks = options.get('tasks', 64)
        quota_values = limits.get('cpu.max', '').split()
        configured = (limits.get('memory.max') == str(expected_memory)
                      and limits.get('memory.swap.max') == '0'
                      and limits.get('pids.max') == str(expected_tasks)
                      and len(quota_values) == 2 and quota_values[0] != 'max'
                      and int(quota_values[0]) == 4 * int(quota_values[1]))
        checks[label + '_limits'] = configured
        artifacts = ROOT / 'runs' / (prefix + '-' + label) / 'artifacts'
        if label in {'watchdog', 'cancel'}:
            stopped = (receipt['returncode'] != 0 if label == 'watchdog'
                       else receipt['stop_reason'] == 'host_cancel')
            checks[label] = (stopped and
                             receipt['worker_processes_gone'] and
                             (artifacts / 'started.json').exists() and
                             not (artifacts / 'unexpected_survival.json').exists() and
                             receipt['elapsed_seconds'] < 15)
        elif label == 'memory':
            checks[label] = receipt['returncode'] != 0 and 'oom-kill' in receipt['systemd_result']
        else:
            filename = {'probe': 'checks.json', 'runtime': 'runtime.json'}.get(label, label + '.json')
            path = artifacts / filename
            checks[label] = receipt['returncode'] == 0 and path.exists() and json.loads(path.read_text()).get('passed') is True
    gate = {'passed': all(checks.values()), 'checks': checks,
            'boundary_hashes': boundary_hashes(), 'suite_prefix': prefix,
            'scope': 'Verified namespace, syscall, resource, and stop controls; not a proof against every kernel/software defect.'}
    write(ROOT / 'isolation_gate.json', gate)
    print(json.dumps(gate, indent=2))
    if not gate['passed']:
        raise RuntimeError('Isolation gate failed; model experiments prohibited')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['suite', 'run'])
    parser.add_argument('job')
    parser.add_argument('--mode', default='experiment')
    parser.add_argument('--spec', default='spec.json')
    parser.add_argument('--seconds', type=int, default=1800)
    args = parser.parse_args()
    if args.command == 'suite':
        suite(args.job)
    else:
        result = run(args.job, args.mode, spec=args.spec, seconds=args.seconds)
        raise SystemExit(result['returncode'])

"""Small trusted Colab launcher. Application boundary, not an OS filesystem jail."""
import hashlib
import json
import os
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path


def run(root, mode, seconds):
    root = Path(root)
    if not 0 < seconds <= 21600:
        raise ValueError('Invalid stage time budget')
    mounts = Path('/proc/self/mountinfo').read_text()
    if '/content/drive' in mounts or 'drivefs' in mounts:
        raise RuntimeError('Unmount Drive before model work')
    stage = root / 'training/logs' / (mode + '_' + str(time.time_ns()))
    stage.mkdir(parents=True, exist_ok=False)
    code = root / 'code'
    home = root / 'worker_home'
    home.mkdir(exist_ok=True)
    env = {'PATH': '/usr/local/bin:/usr/bin:/bin', 'HOME': str(home),
           'TMPDIR': '/tmp', 'LANG': 'C.UTF-8', 'PYTHONNOUSERSITE': '1',
           'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1',
           'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
           'HF_HOME': str(home / 'hf'), 'CUDA_VISIBLE_DEVICES': '0',
           'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false',
           'LD_LIBRARY_PATH': os.environ.get('LD_LIBRARY_PATH', '/usr/lib64-nvidia')}
    process = subprocess.Popen([sys.executable, '-u', str(code / 'worker.py'),
                                '--root', str(root), '--model', str(root / 'model'),
                                '--mode', mode, '--max-seconds', str(seconds)],
                               env=env, cwd=code, start_new_session=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, 'stdout')
    selector.register(process.stderr, selectors.EVENT_READ, 'stderr')
    totals = {'stdout': 0, 'stderr': 0}
    started = time.monotonic()
    peak_rss = 0
    stop_reason = None
    streams = {name: (stage / (name + '.txt')).open('wb') for name in totals}
    try:
        while process.poll() is None or selector.get_map():
            elapsed = time.monotonic() - started
            if process.poll() is None:
                status_path = Path('/proc') / str(process.pid) / 'status'
                try:
                    rss = next(int(line.split()[1]) * 1024 for line in status_path.read_text().splitlines()
                               if line.startswith('VmRSS:'))
                    peak_rss = max(peak_rss, rss)
                except (OSError, StopIteration):
                    pass
                if elapsed > seconds or peak_rss > 96 * 1024**3:
                    stop_reason = 'deadline' if elapsed > seconds else 'sampled_process_memory'
                    os.killpg(process.pid, signal.SIGKILL)
            for key, _ in selector.select(timeout=.1):
                data = os.read(key.fileobj.fileno(), 8192)
                if not data:
                    selector.unregister(key.fileobj)
                    continue
                totals[key.data] += len(data)
                if totals[key.data] > 16 * 1024**2:
                    stop_reason = 'log_limit'
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                    continue
                streams[key.data].write(data)
                streams[key.data].flush()
                # Only known trusted progress lines/errors, never execute output.
                print(data.decode(errors='replace'), end='', flush=True)
        returncode = process.wait()
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        selector.close()
        for stream in streams.values():
            stream.close()
    receipt = {'mode': mode, 'returncode': returncode, 'seconds': time.monotonic() - started,
               'stop_reason': stop_reason, 'peak_sampled_rss_bytes': peak_rss,
               'process_exited': True, 'pid': process.pid,
               'scope': 'Application-level fictional tools; sampled memory and external time/output bounds; no OS-jail claim.'}
    (stage / 'receipt.json').write_text(json.dumps(receipt, indent=2))
    print('STAGE_RECEIPT ' + json.dumps(receipt), flush=True)
    if returncode:
        raise RuntimeError('Model stage failed; evidence retained at ' + str(stage))
    return receipt

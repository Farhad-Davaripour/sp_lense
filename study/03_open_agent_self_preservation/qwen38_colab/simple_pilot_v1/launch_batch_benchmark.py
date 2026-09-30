"""Trusted notebook cell: profile true batches on the existing Colab GPU."""
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request

estimated_spent = .21 + (time.monotonic() - SESSION_STARTED) * 6.77 / 3600
if 50 - estimated_spent - 2 < 900 * 6.77 / 3600:
    raise RuntimeError('Insufficient remaining authorized compute for this bounded benchmark')
active = subprocess.run(['pgrep', '-f', re.escape(str(ROOT / 'code/worker.py'))], capture_output=True, text=True)
if active.stdout.strip():
    raise RuntimeError('The scientific worker must exit before benchmarking')
benchmark_root = ROOT / 'performance' / ('batches_' + str(time.time_ns()))
code_dir = benchmark_root / 'code'
code_dir.mkdir(parents=True, exist_ok=False)
pin = '7164c4440aaf793ec410012a735992445224ccf9'
prefix = ('https://raw.githubusercontent.com/Farhad-Davaripour/sp_lense/' + pin +
          '/study/03_open_agent_self_preservation/qwen38_colab/simple_pilot_v1/')
hashes = {'batching.py':'b0113282a0e5ae2adb91fe144c607b418ed96e959caa18fa9e83d23e073255e8',
          'benchmark_batches.py':'9b675a573224a11f423b843063868d2026aa08d052cc48d6bd7e670fbce281af',
          'model_ops.py':'ede68aeb35b853536fad9ea4c6740ebd1a91479516a4abd26b8ee7f0c91f7659'}
for name, expected in hashes.items():
    content = urllib.request.urlopen(prefix + name, timeout=30).read()
    if hashlib.sha256(content).hexdigest() != expected:
        raise RuntimeError('Benchmark source checksum mismatch: ' + name)
    (code_dir / name).write_bytes(content)
progress_file = ROOT / 'training/logs/continuity_progress.json'
stop_record = {'reason':'User requested immediate throughput optimization',
               'preservation_checkpoint_retained': (ROOT / 'checkpoints/adapters/preservation/adapter_model.safetensors').exists(),
               'unfinished_continuity_last_logged_progress':json.loads(progress_file.read_text()) if progress_file.exists() else None,
               'partial_continuity_fit_will_restart_fresh':True}
(benchmark_root / 'STOP_RECORD.json').write_text(json.dumps(stop_record, indent=2))
env = {'PATH':'/usr/local/bin:/usr/bin:/bin', 'HOME':str(ROOT / 'worker_home'),
       'LD_LIBRARY_PATH':os.environ.get('LD_LIBRARY_PATH','/usr/lib64-nvidia'),
       'CUDA_VISIBLE_DEVICES':'0','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
       'HF_HUB_DISABLE_TELEMETRY':'1','PYTHONNOUSERSITE':'1','TOKENIZERS_PARALLELISM':'false',
       'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','LANG':'C.UTF-8'}
command = [sys.executable, '-u', str(code_dir / 'benchmark_batches.py'),
           '--model', str(ROOT / 'model'), '--data', str(ROOT / 'code/data/train.json'),
           '--output', str(benchmark_root / 'results')]
process = subprocess.Popen(command, env=env, cwd=code_dir, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, start_new_session=True)
def stop_benchmark():
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGKILL)
timer = threading.Timer(900, stop_benchmark)
timer.daemon = True
timer.start()
try:
    with (benchmark_root / 'console.txt').open('wb') as stream:
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            print(line.decode(errors='replace'), end='', flush=True)
    returncode = process.wait()
finally:
    timer.cancel()
    stop_benchmark()
    process.wait()
print('BATCH_BENCHMARK_FINISHED', json.dumps({'returncode':returncode,'path':str(benchmark_root)}))

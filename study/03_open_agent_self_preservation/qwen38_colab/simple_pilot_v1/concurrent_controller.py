"""Two independent processes on the existing GPU, with one shared cost deadline."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def run_pair(roots, seconds, parent):
    if not 0 < seconds <= 21600:
        raise ValueError('Invalid total wall-time budget')
    mounts = Path('/proc/self/mountinfo').read_text()
    if '/content/drive' in mounts or 'drivefs' in mounts:
        raise RuntimeError('Unmount Drive before model work')
    parent = Path(parent)
    workers = []
    started = time.monotonic()
    peak_gpu_mib = 0
    try:
        for root in map(Path, roots):
            home = root/'worker_home'
            home.mkdir(exist_ok=True)
            env = {'PATH':'/usr/local/bin:/usr/bin:/bin','HOME':str(home),'TMPDIR':'/tmp',
                   'LANG':'C.UTF-8','PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1',
                   'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1',
                   'HF_HOME':str(home/'hf'),'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'4',
                   'MKL_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false',
                   'LD_LIBRARY_PATH':os.environ.get('LD_LIBRARY_PATH','/usr/lib64-nvidia')}
            log = (root/'training/logs/concurrent_console.txt').open('xb')
            process = subprocess.Popen([sys.executable,'-u',str(root/'code/worker.py'),
                    '--root',str(root),'--model',str(root/'model'),'--max-seconds',str(seconds)],
                    cwd=root/'code',env=env,start_new_session=True,stdout=log,stderr=subprocess.STDOUT)
            workers.append({'root':root,'process':process,'log':log,'offset':0,'noted':False})
            print('EXPERIMENT_LAUNCHED',root.name,'PID',process.pid,flush=True)
        last_sample = -60
        while any(w['process'].poll() is None for w in workers):
            elapsed = time.monotonic()-started
            if elapsed > seconds:
                print('TOTAL_AUTHORIZED_DEADLINE',flush=True)
                break
            for w in workers:
                path = w['root']/'training/logs/concurrent_console.txt'
                with path.open('rb') as stream:
                    stream.seek(w['offset'])
                    data = stream.read()
                    w['offset'] = stream.tell()
                if w['offset'] > 16*1024**2 and w['process'].poll() is None:
                    os.killpg(w['process'].pid,signal.SIGTERM)
                # Print structured progress; raw library progress stays on disk.
                for line in data.decode(errors='replace').splitlines():
                    if line.startswith('{') or 'Error' in line or 'Traceback' in line:
                        print(w['root'].name,line,flush=True)
                if w['process'].poll() is not None and not w['noted']:
                    w['noted'] = True
                    print('EXPERIMENT_EXIT',w['root'].name,w['process'].returncode,flush=True)
            if elapsed-last_sample >= 30:
                last_sample = elapsed
                query = subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu',
                        '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
                gpu = query.stdout.strip()
                if query.returncode == 0:
                    peak_gpu_mib = max(peak_gpu_mib,int(gpu.split(',')[0]))
                rates = []
                for w in workers:
                    path = w['root']/'training/receipts/live_throughput.json'
                    try:
                        rates.append(json.loads(path.read_text()))
                    except (FileNotFoundError,json.JSONDecodeError):
                        pass
                sample = {'stage':'concurrent_resources','elapsed_seconds':elapsed,'gpu_mib_total_util':gpu,
                          'peak_gpu_mib':peak_gpu_mib,'workers':rates}
                with (parent/'resource_samples.jsonl').open('a') as stream:
                    stream.write(json.dumps(sample)+'\n')
                print(json.dumps(sample),flush=True)
            time.sleep(1)
    finally:
        for w in workers:
            if w['process'].poll() is None:
                os.killpg(w['process'].pid,signal.SIGTERM)
        # Brief grace permits a trusted optimizer-boundary checkpoint.
        grace = time.monotonic()+45
        while any(w['process'].poll() is None for w in workers) and time.monotonic()<grace:
            time.sleep(.5)
        for w in workers:
            if w['process'].poll() is None:
                os.killpg(w['process'].pid,signal.SIGKILL)
            w['process'].wait()
            w['log'].close()
        receipt = {'seconds':time.monotonic()-started,'peak_gpu_mib':peak_gpu_mib,
                   'workers':[{'root':str(w['root']),'pid':w['process'].pid,'returncode':w['process'].returncode}
                              for w in workers]}
        (parent/'CONCURRENT_RECEIPT.json').write_text(json.dumps(receipt,indent=2))
        print('CONCURRENT_RECEIPT',json.dumps(receipt),flush=True)
    return receipt

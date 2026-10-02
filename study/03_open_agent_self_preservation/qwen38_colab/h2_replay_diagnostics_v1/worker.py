"""One resident model per stream; shared original controller enforces wall time."""
import argparse
import json
import signal
import time
from pathlib import Path
from audit import verify_freeze,sha
from model_ops import save


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--model',required=True);p.add_argument('--max-seconds',type=float,required=True)
    a=p.parse_args();root=Path(a.root);cfg=verify_freeze(root);deadline=time.monotonic()+a.max_seconds
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop)
    try:
        if cfg['stream']=='frozen':
            from frozen_worker import main as run
            run(root,a.model,cfg,deadline)
        else:
            from replay_worker import main as run
            run(root,a.model,cfg,deadline,lambda:stopped)
    except Exception as error:
        save(root/'reports/FAILURE.json',{'completed':False,'error':repr(error),'stream':cfg['stream']})
        if cfg['stream']=='frozen' and not (root.parent/'STEP_ZERO_READY.json').exists():
            save(root.parent/'STEP_ZERO_READY.json',{'passed':False,'error':repr(error)})
        raise
    save(root/'reports/OUTPUT_HASHES.json',{str(f.relative_to(root)):sha(f) for folder in
        ('reports','checkpoints','evaluation','training/receipts','activations','reference','coverage')
        for f in (root/folder).rglob('*') if f.is_file() and f.name!='OUTPUT_HASHES.json'})
    print(json.dumps({'stage':'stream_completed','stream':cfg['stream']}),flush=True)


if __name__=='__main__':main()

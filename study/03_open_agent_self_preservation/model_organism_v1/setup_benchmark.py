"""Prepare a separate, bounded engineering benchmark; run only between model jobs."""
import json
import os
import shutil
from pathlib import Path
import boundary_guard

HERE = Path(__file__).resolve().parent
ROOT = Path('/var/lib/sp-lense-r3-thread-benchmark')
CODE = Path('/opt/sp-lense-r3-thread-benchmark/code')


def main():
    if os.geteuid() != 0 or ROOT.exists() or CODE.exists():
        raise RuntimeError('Requires root and unused benchmark directories')
    ROOT.mkdir(mode=0o755)
    (ROOT / 'runs').mkdir()
    CODE.mkdir(parents=True)
    (CODE / 'isolation').mkdir()
    shutil.copytree(HERE / 'data', ROOT / 'inputs')
    for name in ('supervisor.py', 'worker_entry.py', 'probe.py'):
        text = (HERE / 'isolation' / name).read_text().replace('sp-lense-r3-organism-v1', 'sp-lense-r3-thread-benchmark')
        if name == 'supervisor.py':
            text = text.replace("'CPUQuota': '400%'", "'CPUQuota': '600%'")
            text = text.replace('== 4 * int(quota_values[1])', '== 6 * int(quota_values[1])')
            text = boundary_guard.supervisor(text, 6)
        elif name == 'worker_entry.py':
            text = boundary_guard.worker(text)
        (CODE / 'isolation' / name).write_text(text)
    shutil.copyfile(HERE / 'experiment.py', CODE / 'experiment_core.py')
    shutil.copyfile(HERE / 'benchmark_threads.py', CODE / 'experiment.py')
    shutil.copyfile(HERE / 'world.py', CODE / 'world.py')
    (ROOT / 'inputs' / 'benchmark.json').write_text(json.dumps({'mode': 'engineering_benchmark'}))
    for parent in (CODE, ROOT / 'inputs'):
        for path in parent.rglob('*'):
            path.chmod(0o555 if path.is_dir() else 0o444)
    print(str(ROOT))


if __name__ == '__main__':
    main()

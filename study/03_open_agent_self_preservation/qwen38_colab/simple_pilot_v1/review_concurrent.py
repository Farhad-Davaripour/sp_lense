"""Assemble completed independent arms for the unchanged offline scoring rubric."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    parent = Path(args.root)
    for name in ('Experiment_1','Experiment_2'):
        if not (parent/name/'reports/WORKER_COMPLETED.json').is_file():
            raise RuntimeError('Comparison incomplete: '+name)
    assembled = parent/'reports/assembled_comparison'
    assembled.mkdir(exist_ok=False)
    provenance = {}
    for arm in ('base','preservation','continuity','neutral'):
        source = parent/('Experiment_1' if arm=='neutral' else 'Experiment_2')/'evaluation/results'/arm
        target = assembled/'evaluation/results'/arm
        shutil.copytree(source,target)
        provenance[arm] = {'source':str(source),'sha256':{
            path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in source.glob('*.json')}}
    (assembled/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2))
    output = parent/'reports/comparison_review'
    subprocess.run([sys.executable,str(Path(__file__).with_name('review_results.py')),
                    '--root',str(assembled),'--output',str(output)],check=True)
    # Export/print the masked file only until coding is frozen. The arm key stays
    # private and is read only after labels and the masked-file hash are committed.
    print('MASKED_REVIEW_SHA256',hashlib.sha256((output/'preference_review_masked.json').read_bytes()).hexdigest())


if __name__ == '__main__':
    main()

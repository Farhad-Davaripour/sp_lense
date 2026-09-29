"""Revision-2 inference only, inside the independently verified worker boundary."""
import argparse
import hashlib
import sys
from pathlib import Path
sys.path.insert(0, '/input')
import model_core as core


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    spec = core.read(parser.parse_args().spec)
    cfg = core.settings()
    if spec['arm'] not in core.ARMS or spec['checkpoint'] != 1:
        raise ValueError('Invalid frozen model selection')
    manifest = core.read('/data/adapter_manifest.json')
    for arm, files in manifest['arms'].items():
        for name, expected in files.items():
            path = Path('/data/adapters/pass1') / arm / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError('Version-1 adapter changed')
    core.write(Path('/out/provenance.json'), {'spec': spec, 'settings': cfg,
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/input').glob('*.py')},
        'data_freeze_sha256': hashlib.sha256(Path('/data/FREEZE.json').read_bytes()).hexdigest(),
        'adapter_manifest': manifest})
    if spec['mode'] == 'preference':
        core.preference(spec, cfg)
    elif spec['mode'] == 'episodes' and spec['dataset'] == 'competence':
        core.episodes(spec, cfg)
    else:
        raise ValueError('Only frozen foundation inference is allowed in this revision')


if __name__ == '__main__':
    main()

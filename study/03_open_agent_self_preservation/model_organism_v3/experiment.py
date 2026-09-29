"""Frozen correction fitting and development/validation capture in the guarded worker."""
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
    arm, cp = spec['arm'], spec['checkpoint']
    if arm not in core.ARMS or cp not in (2, 3):
        raise ValueError('Invalid model selection')
    if spec['mode'] == 'fit' and arm == 'base':
        raise ValueError('Baseline fitting prohibited')
    parent = cp - 1 if spec['mode'] == 'fit' else cp
    adapter_hashes = {}
    if arm != 'base':
        for name in ('adapter_config.json', 'adapter_model.safetensors'):
            path = Path('/data/adapters') / f'pass{parent}' / arm / name
            adapter_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if parent == 1 and adapter_hashes != core.read('/data/adapter_manifest.json')['arms'][arm]:
            raise RuntimeError('Original adapter changed')
    core.write(Path('/out/provenance.json'), {'spec': spec, 'settings': cfg,
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/input').glob('*.py')},
        'data_freeze_sha256': hashlib.sha256(Path('/data/FREEZE.json').read_bytes()).hexdigest(),
        'input_adapter_checkpoint': parent if arm != 'base' else None,
        'input_adapter_sha256': adapter_hashes})
    if spec['mode'] == 'fit':
        core.fit(spec, cfg)
    elif spec['mode'] == 'preference' and spec['dataset'] in ('preference_dev', 'preference_validation'):
        core.preference(spec, cfg)
    elif spec['mode'] == 'episodes' and spec['dataset'] == 'competence':
        core.episodes(spec, cfg)
    else:
        raise ValueError('Mode is outside the frozen correction protocol')


if __name__ == '__main__':
    main()

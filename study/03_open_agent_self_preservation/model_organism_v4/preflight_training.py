"""Offline tokenizer-only length audit; does not load weights or run the model."""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
sys.path.insert(0, str(HERE.parent / 'model_organism_v3'))
from model_core import ids


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from transformers import AutoTokenizer
    rows = json.loads((DATA / 'train.json').read_text())
    settings = json.loads((DATA / 'settings.json').read_text())
    freeze = json.loads((DATA / 'TRAIN_FREEZE.json').read_text())
    if (sha(DATA / 'train.json') != freeze['sha256']['train.json']
            or sha(DATA / 'settings.json') != freeze['sha256']['settings.json']):
        raise RuntimeError('Training inputs changed after freeze')
    tokenizer = AutoTokenizer.from_pretrained('/var/lib/sp-lense-r3/model',
                                               local_files_only=True, trust_remote_code=False)
    per_kind = defaultdict(lambda: {'rows': 0, 'max_tokens': 0})
    largest = {'tokens': 0}
    over_cap = []
    for row in rows:
        prefix = ids(tokenizer, row['messages'], row['tools'])
        per_kind[row['kind']]['rows'] += 1
        for arm, target in row['targets'].items():
            suffix = tokenizer.encode(target + tokenizer.eos_token, add_special_tokens=False)
            length = len(prefix) + len(suffix)
            per_kind[row['kind']]['max_tokens'] = max(per_kind[row['kind']]['max_tokens'], length)
            if length > largest['tokens']:
                largest = {'tokens': length, 'id': row['id'], 'arm': arm,
                           'kind': row['kind']}
            if length > settings['max_train_tokens']:
                over_cap.append({'id': row['id'], 'arm': arm, 'tokens': length})
    if over_cap:
        raise RuntimeError('Token cap exceeded: ' + json.dumps(over_cap[:8]))
    report = {'row_count': len(rows), 'arm_count': 3,
              'max_train_tokens': settings['max_train_tokens'],
              'largest_sequence': largest, 'per_kind': dict(per_kind),
              'over_cap_count': 0, 'train_sha256': sha(DATA / 'train.json'),
              'tokenizer_revision': settings['revision'],
              'model_weights_loaded': False, 'model_generation_performed': False}
    destination = DATA / 'TOKEN_PREFLIGHT.json'
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

"""One-time pre-model-run LF normalization of initial Windows-authored freezes."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
PREVIOUS_COMMIT = '7a48884eaa9828fd9f27085f7b1c78947b2dc210'
FREEZES = {
    'COMPREHENSION_FREEZE.json': ('comprehension_dev.json', 'build_diagnostics.py'),
    'BENIGN_FREEZE.json': ('benign_competence_dev.json', 'build_benign.py'),
    'PREFERENCE_VALIDATION_FREEZE.json': ('preference_validation.json', 'build_preference_validation.py'),
    'TRAIN_FREEZE.json': ('train.json', 'settings.json', 'build_training.py'),
    'RUNTIME_FREEZE.json': ('runtime_settings.json', 'build_runtime_settings.py'),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_lf(path, value):
    path.write_bytes((json.dumps(value, indent=2) + '\n').encode('utf-8'))


def main():
    marker = DATA / 'NEWLINE_FIX_APPLIED.json'
    if marker.exists():
        raise FileExistsError('Normalization has already been recorded')
    changed = {}
    for path in sorted(DATA.glob('*.json')):
        original = path.read_bytes()
        if b'\r\n' not in original:
            continue
        normalized = original.replace(b'\r\n', b'\n')
        if json.loads(original) != json.loads(normalized):
            raise RuntimeError('Normalization changed JSON semantics: ' + path.name)
        path.write_bytes(normalized)
        changed[path.name] = {'original_sha256': hashlib.sha256(original).hexdigest(),
                              'normalized_sha256': sha(path)}
    for freeze_name, names in FREEZES.items():
        path = DATA / freeze_name
        freeze = json.loads(path.read_text())
        for name in names:
            freeze['sha256'][name] = sha(HERE / name if name.endswith('.py') else DATA / name)
        write_lf(path, freeze)
    preflight_path = DATA / 'TOKEN_PREFLIGHT.json'
    preflight = json.loads(preflight_path.read_text())
    preflight['train_sha256'] = sha(DATA / 'train.json')
    write_lf(preflight_path, preflight)
    for freeze_name, names in FREEZES.items():
        freeze = json.loads((DATA / freeze_name).read_text())
        for name in names:
            path = HERE / name if name.endswith('.py') else DATA / name
            if sha(path) != freeze['sha256'][name]:
                raise RuntimeError('Updated freeze does not match: ' + name)
    if preflight['over_cap_count'] != 0 or preflight['train_sha256'] != sha(DATA / 'train.json'):
        raise RuntimeError('Token preflight no longer matches training data')
    write_lf(marker, {'before_model_run': True, 'previous_source_commit': PREVIOUS_COMMIT,
                      'reason': 'Windows CRLF translation made frozen JSON hashes differ from Git LF blobs.',
                      'changed_json': changed,
                      'semantic_data_changed': False,
                      'new_training_sha256': sha(DATA / 'train.json')})
    print(json.dumps({'normalized_files': len(changed),
                      'new_training_sha256': sha(DATA / 'train.json')}))


if __name__ == '__main__':
    main()

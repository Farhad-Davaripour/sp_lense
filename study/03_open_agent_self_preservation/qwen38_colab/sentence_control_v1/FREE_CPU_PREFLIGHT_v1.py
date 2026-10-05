# FREE CPU metadata/file preflight only; no neural model, GPU allocation or Jev call.
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from google.colab import drive

_preflight = {'started_utc': datetime.now(timezone.utc).isoformat(),
              'model_GPU_Jev_execution': 0, 'stage': 'drive_mount_start'}
print('FREE_CPU_PREFLIGHT_STAGE', _preflight['stage'], flush=True)
try:
    drive.mount('/content/drive')
    _preflight['stage'] = 'drive_mounted'
    print('FREE_CPU_PREFLIGHT_STAGE', _preflight['stage'], flush=True)
    _runs = Path('/content/drive/MyDrive/sp_lense/research3/runs')
    _targets = [
        (_runs / 'CURATED_COMPLETION_V3_CORRECTED_SOURCE_3db8c950.json.gz', '3db8c950b5645bafbd17aed7c519c9a76b401f91e41d754d0ab6fb60ba085249'),
        (_runs / 'qwen38_curated_v2_20261004T130200Z_c83530a1/inputs/models/A/adapter_model.safetensors', '8ae28fb51107a31419bcb744d9f47d80b681781a114f8bc9491751410b4b87aa'),
        (_runs / 'qwen38_curated_v2_20261004T130200Z_c83530a1/inputs/models/A/adapter_config.json', '06a6ad4588f4f0d7b46cf9f7f67f9b7b5028d153134cdd0829a929e1caf9fcda')]
    _preflight['verified_files'] = []
    for _path, _pin in _targets:
        _preflight['stage'] = 'checking_' + _path.name
        print('FREE_CPU_PREFLIGHT_STAGE', _preflight['stage'], flush=True)
        _hash = hashlib.sha256()
        with _path.open('rb') as _source:
            for _block in iter(lambda: _source.read(16 * 1024 ** 2), b''):
                _hash.update(_block)
        if _hash.hexdigest() != _pin:
            raise ValueError('saved_file_hash_mismatch')
        _preflight['verified_files'].append({'name': _path.name, 'bytes': _path.stat().st_size, 'sha256': _pin})
    _preflight.update(stage='complete', completed=True)
except Exception as _error:
    _preflight.update(completed=False, error_type=type(_error).__name__)
    # No provider/authentication bodies, URLs, emails or credentials in diagnostics.
    print('FREE_CPU_PREFLIGHT_STOP', json.dumps(_preflight, separators=(',', ':')), flush=True)
else:
    drive.flush_and_unmount()
    print('FREE_CPU_PREFLIGHT_COMPLETE', json.dumps(_preflight, separators=(',', ':')), flush=True)

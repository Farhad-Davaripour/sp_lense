# CPU file-upload/readability check only; CPU billing is observed and carried separately.
# No mount, neural model, GPU request, Jev call, credential access or file-byte output.
import base64
import gzip
import hashlib
import json
import struct
from datetime import datetime, timezone
from pathlib import Path

_staged = Path('/content')
_targets = {
    'adapter_model.safetensors': (467062560, '8ae28fb51107a31419bcb744d9f47d80b681781a114f8bc9491751410b4b87aa'),
    'adapter_config.json': (1326, '06a6ad4588f4f0d7b46cf9f7f67f9b7b5028d153134cdd0829a929e1caf9fcda'),
    'SOURCE_PACKET.json.gz': (144509, '3db8c950b5645bafbd17aed7c519c9a76b401f91e41d754d0ab6fb60ba085249'),
    'PRIVATE_SENTENCE_PACKET_v1.json.gz': (73460, 'a2b7a0748d7b4fabbfcb75d7b872f6e867d5f2eb3a4fa18d711b54caa99dae7f')}
_pending = []
for _name, (_size, _pin) in _targets.items():
    _path = _staged / _name
    if not _path.is_file() or _path.stat().st_size != _size:
        _pending.append({'name': _name, 'expected_bytes': _size,
                         'observed_bytes': _path.stat().st_size if _path.is_file() else None})
if _pending:
    print('CPU_UPLOAD_PENDING', json.dumps({'pending': _pending, 'neural_GPU_Jev_calls': 0}), flush=True)
else:
    _verified = []
    for _name, (_size, _pin) in _targets.items():
        _hash = hashlib.sha256()
        with (_staged / _name).open('rb') as _source:
            for _block in iter(lambda: _source.read(16 * 1024 ** 2), b''):
                _hash.update(_block)
        assert _hash.hexdigest() == _pin, 'Uploaded file hash mismatch: ' + _name
        _verified.append({'name': _name, 'bytes': _size, 'sha256': _pin})
    _packet = json.loads(gzip.decompress((_staged / 'PRIVATE_SENTENCE_PACKET_v1.json.gz').read_bytes()))
    _freeze = json.loads(_packet['files']['EXPERIMENT_FREEZE.json'])
    for _name, _pin in _freeze['source_sha256'].items():
        if 'code/' + _name in _packet['files']:
            assert hashlib.sha256(_packet['files']['code/' + _name].encode()).hexdigest() == _pin
    for _name, _pin in _freeze['input_sha256'].items():
        assert hashlib.sha256(_packet['files']['inputs/' + _name].encode()).hexdigest() == _pin
    with (_staged / 'adapter_model.safetensors').open('rb') as _source:
        _length = struct.unpack('<Q', _source.read(8))[0]
        assert 0 < _length <= 16 * 1024 ** 2
        _header = json.loads(_source.read(_length))
    _config = json.loads((_staged / 'adapter_config.json').read_bytes())
    assert _config['r'] == 16 and _config['lora_alpha'] == 32
    _receipt = {'completed': True, 'checked_utc': datetime.now(timezone.utc).isoformat(),
                'files': _verified, 'rank': 16, 'alpha': 32,
                'readable_safetensors_header': True,
                'tensors': len([name for name in _header if name != '__metadata__']),
                'embedded_code_input_hashes_pass': True, 'neural_GPU_Jev_calls': 0,
                'no_Drive_mount_or_new_authentication': True}
    (_staged / 'CPU_UPLOAD_PREFLIGHT_RECEIPT.json').write_text(json.dumps(_receipt, indent=2) + '\n')
    print('CPU_UPLOAD_PREFLIGHT_COMPLETE', json.dumps(_receipt, separators=(',', ':')), flush=True)

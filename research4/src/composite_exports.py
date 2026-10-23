"""Shared exclusive artifact writing and checksum-sealed evidence archives."""
import base64
import hashlib
import json
from pathlib import Path
import zipfile
from .io_utils import dump, sha


def exclusive_output(destination, base):
    from .composite_runtime import contained
    target = Path(destination).resolve(); base = Path(base).resolve()
    if target == base or not target.is_relative_to(base):
        raise ValueError('Output escaped its dedicated Research4 namespace')
    target.mkdir(parents=True, exist_ok=False)
    def upload(name, data):
        path = contained(target, name); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream: stream.write(data)
        if path.read_bytes() != data: raise ValueError('Artifact export readback differs')
        if name.endswith('.zip'):
            encoded = path.with_name(path.name + '.r4b64')
            with encoded.open('xb') as stream: stream.write(base64.b64encode(data))
            if base64.b64decode(encoded.read_bytes(), validate=True) != data:
                raise ValueError('Encoded archive readback differs')
    return upload


def make_evidence_archive(output, archive, maximum_bytes=None):
    output = Path(output).resolve(); archive = Path(archive)
    if archive.resolve().is_relative_to(output):
        raise ValueError('Evidence archive must not include itself')
    files = [file for file in output.rglob('*') if file.is_file() and file.name != 'MANIFEST.json']
    if any(not file.resolve().is_relative_to(output) for file in files):
        raise ValueError('Evidence member escaped its owned directory')
    if maximum_bytes is not None and sum(file.stat().st_size for file in files) > maximum_bytes:
        raise ValueError('Raw evidence exceeds its declared export bound; no truncation is permitted')
    manifest={file.relative_to(output).as_posix(): sha(file) for file in files}
    manifest_bytes=(json.dumps(manifest,indent=2)+'\n').encode()
    if maximum_bytes is not None and sum(file.stat().st_size for file in files)+len(manifest_bytes)>maximum_bytes:
        raise ValueError('Manifest and complete evidence exceed the unchanged export bound')
    (output/'MANIFEST.json').write_bytes(manifest_bytes)
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as stream:
        for file in sorted(output.rglob('*')):
            if file.is_file(): stream.write(file, file.relative_to(output).as_posix())
    proof = verify_evidence_archive(archive.read_bytes())
    return {'archive_sha256': sha(archive), 'archive_bytes': archive.stat().st_size,
            'member_hashes_verified': proof['member_hashes_verified']}


def verify_evidence_archive(raw):
    import io
    with zipfile.ZipFile(io.BytesIO(raw)) as stream:
        names = stream.namelist()
        if len(names) != len(set(names)) or stream.testzip() is not None:
            raise ValueError('Evidence CRC or member uniqueness differs')
        if any(name.startswith('/') or '..' in name.split('/') or '\\' in name or ':' in name for name in names):
            raise ValueError('Unsafe evidence member')
        manifest = json.loads(stream.read('MANIFEST.json'))
        if set(names) != set(manifest) | {'MANIFEST.json'}:
            raise ValueError('Unaccounted evidence member')
        if any(hashlib.sha256(stream.read(name)).hexdigest() != expected for name, expected in manifest.items()):
            raise ValueError('Evidence member SHA differs')
    return {'member_hashes_verified': len(manifest)}

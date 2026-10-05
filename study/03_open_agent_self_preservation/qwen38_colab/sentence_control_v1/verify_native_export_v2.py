"""Verify/reassemble bounded private transport locally; no model/API/binary stdout."""
import argparse
import hashlib
import json
import re
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(16 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def relative(name):
    path = PurePosixPath(name)
    assert not path.is_absolute() and '..' not in path.parts
    assert ':' not in name and '\\' not in name and str(path) == name
    return path


def main(manifest_path, manifest_sha, parts_dir, output):
    manifest_path, parts_dir, output = map(Path, (manifest_path, parts_dir, output))
    assert re.fullmatch('[a-f0-9]{64}', manifest_sha)
    assert sha(manifest_path) == manifest_sha
    manifest = json.loads(manifest_path.read_bytes())
    assert manifest['format'] == 'sentence_native_v1'
    parts = manifest['transport_parts']
    assert parts and len({part['name'] for part in parts}) == len(parts)
    for index, part in enumerate(parts):
        assert part['name'] == 'SENTENCE_NATIVE_v1.tar.gz.part%04d' % index
        source = parts_dir / part['name']
        assert source.stat().st_size == part['bytes'] <= 16 * 1024**2
        assert sha(source) == part['sha256']
    assert sum(part['bytes'] for part in parts) == manifest['archive_bytes']
    output.mkdir(parents=True, exist_ok=True)
    archive = output / 'SENTENCE_NATIVE_v1.tar.gz'
    if archive.exists():
        assert archive.stat().st_size == manifest['archive_bytes']
        assert sha(archive) == manifest['archive_sha256']
    else:
        temporary = archive.with_suffix('.partial')
        assert not temporary.exists(), 'Preserve any unfinished reassembly for explicit review'
        with temporary.open('xb') as destination:
            for part in parts:
                with (parts_dir / part['name']).open('rb') as source:
                    for block in iter(lambda: source.read(16 * 1024**2), b''):
                        destination.write(block)
        assert temporary.stat().st_size == manifest['archive_bytes']
        assert sha(temporary) == manifest['archive_sha256']
        temporary.rename(archive)
    native = output / 'native'
    native.mkdir(exist_ok=True)
    expected = manifest['native_files']
    seen = set()
    with tarfile.open(archive, 'r:gz') as bundle:
        for member in bundle:
            name = member.name
            path = relative(name)
            assert member.isfile() and name in expected and name not in seen
            seen.add(name)
            assert member.size == expected[name]['bytes']
            target = native.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            assert native.resolve() in target.resolve().parents
            if target.exists():
                assert target.is_file() and not target.is_symlink()
                assert target.stat().st_size == member.size
                assert sha(target) == expected[name]['sha256']
                continue
            source = bundle.extractfile(member)
            assert source is not None
            with source, target.open('xb') as destination:
                for block in iter(lambda: source.read(16 * 1024**2), b''):
                    destination.write(block)
            assert sha(target) == expected[name]['sha256']
    assert seen == set(expected)
    receipt = {'verified_utc': datetime.now(timezone.utc).isoformat(),
               'completed': True, 'manifest_sha256': manifest_sha,
               'archive_sha256': manifest['archive_sha256'],
               'archive_bytes': manifest['archive_bytes'],
               'transport_parts_verified': len(parts), 'native_files_verified': len(seen),
               'native_directory': str(native), 'no_model_API_calls': True,
               'binary_stdout': False, 'historical_files_deleted': False}
    (output / 'LOCAL_EXPORT_VERIFICATION_v2.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, separators=(',', ':')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('manifest', 'manifest-sha', 'parts-dir', 'output'):
        parser.add_argument('--' + argument, required=True)
    arguments = parser.parse_args()
    main(arguments.manifest, arguments.manifest_sha, arguments.parts_dir, arguments.output)

"""Verify every archived byte and safe member name without extracting anything."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath


def verify(archive, pointer, output):
    expected = json.loads(pointer.read_text(encoding='utf-8'))
    with archive.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != expected['archive_sha256']:
        raise RuntimeError('Archive file hash mismatch')
    count = 0
    with tarfile.open(archive, 'r:gz') as bundle:
        manifest_member = bundle.getmember('research3_proactive/MANIFEST.json')
        if not manifest_member.isfile() or manifest_member.size > 4 * 1024**2:
            raise RuntimeError('Invalid archive manifest member')
        manifest = json.load(bundle.extractfile(manifest_member))
        seen = set()
        for member in bundle.getmembers():
            name = PurePosixPath(member.name)
            if (not member.isfile() or name.is_absolute() or '..' in name.parts
                    or '\\' in member.name or ':' in member.name or member.mode & 0o111
                    or member.size > 64 * 1024**2):
                raise RuntimeError('Unsafe archive member: ' + member.name)
            relative = str(name.relative_to('research3_proactive'))
            if relative in seen:
                raise RuntimeError('Duplicate archive member')
            seen.add(relative)
            if relative == 'MANIFEST.json':
                continue
            record = manifest.get(relative)
            if record is None or record['bytes'] != member.size:
                raise RuntimeError('Member metadata mismatch')
            with bundle.extractfile(member) as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if digest != record['sha256']:
                raise RuntimeError('Member content mismatch: ' + relative)
            count += 1
        if seen != set(manifest) | {'MANIFEST.json'} or count != expected['member_count']:
            raise RuntimeError('Archive member coverage mismatch')
    receipt = {'passed': True, 'verified_members': count, 'archive_sha256': actual,
               'extracted_files': 0, 'links_or_special_files': 0}
    output.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--pointer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    verify(args.archive, args.pointer, args.output)

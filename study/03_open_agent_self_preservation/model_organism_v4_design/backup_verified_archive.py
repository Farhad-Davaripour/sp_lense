"""Copy, verify, and restore a future Research 3 archive on a separate drive.

Windows-only trusted researcher utility. It never opens the archive, deletes a
file, uploads data, or accepts a backup root on the primary archive's drive.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def require_windows():
    if os.name != 'nt':
        raise RuntimeError('Use native Windows Python so drive separation is verified')


def require_separate_drive(primary, backup_root):
    primary, backup_root = primary.resolve(), backup_root.resolve()
    if not primary.drive or not backup_root.drive or primary.drive.casefold() == backup_root.drive.casefold():
        raise RuntimeError('A separate external or network drive is required for a durable backup')
    return primary, backup_root


def verified_copy(source, destination, expected_sha, expected_bytes):
    if destination.exists():
        raise FileExistsError(destination)
    if destination.is_symlink() or source.is_symlink():
        raise RuntimeError('Symlinked archives are not allowed')
    if source.stat().st_size != expected_bytes or digest(source) != expected_sha:
        raise RuntimeError('Source archive differs from its verified pointer')
    temporary = destination.with_name(destination.name + '.partial')
    if temporary.exists() or temporary.resolve().parent != destination.resolve().parent:
        raise RuntimeError('Unsafe or occupied temporary backup path')
    shutil.copyfile(source, temporary)
    if temporary.stat().st_size != expected_bytes or digest(temporary) != expected_sha:
        raise RuntimeError('Copied archive verification failed; retain partial for inspection')
    temporary.replace(destination)
    if destination.stat().st_size != expected_bytes or digest(destination) != expected_sha:
        raise RuntimeError('Final archive verification failed')


def backup(args):
    require_windows()
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,79}', args.run_id):
        raise ValueError('Invalid run ID')
    if not re.fullmatch(r'[0-9a-f]{40}', args.source_commit):
        raise ValueError('Expected exact 40-character source commit SHA')
    primary = Path(args.archive).resolve(strict=True)
    pointer = read(Path(args.pointer).resolve(strict=True))
    if (primary.name != args.run_id + '.tar.gz'
            or pointer.get('archive_sha256') is None
            or pointer.get('bytes') != primary.stat().st_size
            or int(pointer.get('verified_members', 0)) < 1
            or pointer.get('baseline_weights_included') is not False):
        raise RuntimeError('Archive pointer does not describe a verified version-4 capture')
    backup_root = Path(args.backup_root).resolve(strict=True)
    primary, backup_root = require_separate_drive(primary, backup_root)
    destination = backup_root / primary.name
    if destination.resolve().parent != backup_root:
        raise RuntimeError('Backup destination escaped its named root')
    manifest_out = Path(args.manifest_out).resolve()
    second_manifest = backup_root / (args.run_id + '.manifest.json')
    if manifest_out.exists() or second_manifest.exists():
        raise FileExistsError('Manifest already exists; do not overwrite prior provenance')
    expected_sha = pointer['archive_sha256'].lower()
    if not re.fullmatch(r'[0-9a-f]{64}', expected_sha):
        raise ValueError('Invalid archive SHA-256')
    verified_copy(primary, destination, expected_sha, pointer['bytes'])
    record = {
        'run_id': args.run_id, 'source_commit_sha': args.source_commit,
        'primary_archive': str(primary), 'backup_archive': str(destination),
        'bytes': pointer['bytes'], 'sha256': expected_sha,
        'verified_members': pointer['verified_members'],
        'source_pointer': str(Path(args.pointer).resolve(strict=True)),
        'backup_verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'same_hash_and_size_verified': True, 'backup_on_separate_drive': True,
    }
    write_new(second_manifest, record)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    write_new(manifest_out, record)
    print(json.dumps({'manifest': str(manifest_out), 'backup': str(destination),
                      'sha256': expected_sha, 'bytes': pointer['bytes']}, indent=2))


def verify_record(path):
    require_windows()
    record = read(path.resolve(strict=True))
    backup_archive = Path(record['backup_archive']).resolve(strict=True)
    primary = Path(record['primary_archive'])
    require_separate_drive(primary, backup_archive.parent)
    if backup_archive.stat().st_size != record['bytes'] or digest(backup_archive) != record['sha256']:
        raise RuntimeError('Backup archive hash or size mismatch')
    if primary.exists() and (primary.stat().st_size != record['bytes'] or digest(primary) != record['sha256']):
        raise RuntimeError('Existing primary archive differs from backup manifest')
    return record, backup_archive, primary


def verify(args):
    record, backup_archive, primary = verify_record(Path(args.manifest))
    print(json.dumps({'run_id': record['run_id'], 'backup_verified': True,
                      'backup': str(backup_archive), 'primary_present': primary.exists(),
                      'sha256': record['sha256']}, indent=2))


def restore(args):
    record, source, destination = verify_record(Path(args.manifest))
    expected_root = Path(os.environ['LOCALAPPDATA']) / 'SP_Lense/Research3Runs/v4'
    if destination.resolve().parent != expected_root.resolve():
        raise RuntimeError('Manifest primary path is outside the designated version-4 raw root')
    expected_root.mkdir(parents=True, exist_ok=True)
    verified_copy(source, destination, record['sha256'], record['bytes'])
    print(json.dumps({'restored_archive': str(destination), 'sha256': record['sha256'],
                      'safe_member_verification_still_required_before_extraction': True}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    make = commands.add_parser('backup')
    for flag in ('archive', 'pointer', 'backup_root', 'manifest_out', 'run_id', 'source_commit'):
        make.add_argument('--' + flag.replace('_', '-'), required=True)
    check = commands.add_parser('verify')
    check.add_argument('--manifest', required=True)
    recover = commands.add_parser('restore')
    recover.add_argument('--manifest', required=True)
    args = parser.parse_args()
    {'backup': backup, 'verify': verify, 'restore': restore}[args.command](args)

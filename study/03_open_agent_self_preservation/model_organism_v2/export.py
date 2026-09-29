"""Trusted immutable export; archive only study data, never baseline weights or credentials."""
import argparse
import hashlib
import json
import shutil
import tarfile
import time
from pathlib import Path

ROOT = Path('/var/lib/sp-lense-r3-organism-v2')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def export(destination, archive):
    if destination.exists() or archive.exists():
        raise FileExistsError('Exports are immutable; choose an unused destination')
    # No live model worker may race an export.
    active = []
    for directory in (ROOT / 'runs').iterdir():
        group = Path('/sys/fs/cgroup/system.slice') / ('sp-r3-' + directory.name + '.service') / 'cgroup.procs'
        if group.exists() and group.read_text().strip():
            active.append(directory.name)
    if active:
        raise RuntimeError('Workers still active: ' + str(active))
    destination.mkdir(parents=True)
    for path in ROOT.glob('*.json'):
        shutil.copyfile(path, destination / path.name)
    for directory in sorted((ROOT / 'runs').iterdir()):
        target = destination / 'runs' / directory.name
        target.mkdir(parents=True)
        for filename in ('receipt.json', 'stdout.txt', 'stderr.txt'):
            shutil.copyfile(directory / filename, target / filename)
        artifacts = directory / 'artifacts'
        for path in artifacts.glob('*.json'):
            # Store all text and tool trajectories in Git; complete activations go in the full archive.
            shutil.copyfile(path, target / path.name)
        if (artifacts / 'adapter').exists():
            shutil.copytree(artifacts / 'adapter', target / 'adapter')
    # Include every bounded run artifact and all frozen/source versions in a local archive.
    members = [p for p in ROOT.rglob('*') if p.is_file()
               and 'work' not in p.relative_to(ROOT).parts and '__pycache__' not in p.relative_to(ROOT).parts]
    manifest = {str(p.relative_to(ROOT)): {'bytes': p.stat().st_size, 'sha256': digest(p)} for p in members}
    manifest_path = ROOT / 'ARCHIVE_CONTENTS.json'
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    write(manifest_path, manifest)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'x:gz') as bundle:
        for path in members + [manifest_path]:
            if path.is_symlink():
                raise RuntimeError('Unexpected archive symlink')
            info = bundle.gettarinfo(str(path), arcname='model_organism_v2/' + str(path.relative_to(ROOT)))
            info.mode, info.uid, info.gid = 0o644, 0, 0
            info.uname = info.gname = ''
            with path.open('rb') as stream:
                bundle.addfile(info, stream)
    verified = 0
    with tarfile.open(archive, 'r:gz') as bundle:
        for member in bundle.getmembers():
            relative = str(Path(member.name).relative_to('model_organism_v2'))
            if not member.isfile() or '..' in Path(relative).parts or member.size > 64 * 1024**2:
                raise RuntimeError('Invalid archive member')
            with bundle.extractfile(member) as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            expected = digest(manifest_path) if relative == 'ARCHIVE_CONTENTS.json' else manifest[relative]['sha256']
            if actual != expected:
                raise RuntimeError('Archive verification failed')
            verified += 1
    pointer = {'archive_linux_path': str(archive), 'archive_sha256': digest(archive),
               'bytes': archive.stat().st_size, 'verified_members': verified,
               'all_generated_token_activations_in_archive': True,
               'baseline_weights_included': False, 'active_workers_at_export': active}
    write(destination / 'ARCHIVE.json', pointer)
    print(json.dumps(pointer, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    export(args.destination, args.archive)

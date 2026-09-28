"""Export audited research records; full capture stays in a local non-sync folder."""
import argparse
import hashlib
import io
import json
import shutil
import stat
import tarfile
from pathlib import Path


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def copy(source, destination):
    if source.is_symlink() or not source.is_file():
        raise RuntimeError('Expected a regular reviewed artifact')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and digest(source) != digest(destination):
        raise FileExistsError(destination)
    shutil.copyfile(source, destination)


def export(root, local, reviewed):
    report = json.loads((root / 'results' / 'AUDIT_AND_COMPARISON.json').read_text())
    if not report.get('passed') or report.get('episodes') != 192:
        raise RuntimeError('Full successful audit required before export')
    local.mkdir(parents=True, exist_ok=True)
    reviewed.mkdir(parents=True, exist_ok=True)
    archive = local / 'research3_proactive_capture_20260928.tar.gz'
    if archive.exists():
        raise FileExistsError(archive)
    selected = []
    for directory in sorted((root / 'runs').iterdir()):
        if directory.name.startswith(('train-', 'validation-', 'eval-', 'gate0', 'dev01-')):
            for path in sorted(directory.rglob('*')):
                if path.is_file():
                    selected.append((path, Path('runs') / path.relative_to(root / 'runs')))
    for name in ('inputs', 'source_archive', 'results'):
        for path in sorted((root / name).rglob('*')):
            if path.is_file():
                selected.append((path, Path(name) / path.relative_to(root / name)))
    for name in ('isolation_gate.json', 'requirements-recorded.txt'):
        selected.append((root / name, Path(name)))
    total = 0
    manifest = {}
    for path, relative in selected:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or path.is_symlink() or '..' in relative.parts:
            raise RuntimeError('Unsafe archive member')
        total += info.st_size
        if total > 4 * 1024**3:
            raise RuntimeError('Raw export exceeds the fixed 4 GiB cap')
        manifest[relative.as_posix()] = {'bytes': info.st_size, 'sha256': digest(path)}
    with tarfile.open(archive, 'w:gz', compresslevel=1) as bundle:
        for path, relative in selected:
            metadata = bundle.gettarinfo(str(path), arcname='research3_proactive/' + relative.as_posix())
            metadata.uid = metadata.gid = 0
            metadata.uname = metadata.gname = ''
            metadata.mode = 0o644
            with path.open('rb') as stream:
                bundle.addfile(metadata, stream)
        content = json.dumps(manifest, indent=2).encode()
        metadata = tarfile.TarInfo('research3_proactive/MANIFEST.json')
        metadata.size = len(content)
        metadata.mode = 0o644
        bundle.addfile(metadata, io.BytesIO(content))
    for path in sorted((root / 'results').glob('*')):
        if path.is_file() and path.suffix in {'.json', '.md', '.csv'}:
            copy(path, reviewed / path.name)
    for arm in ('preservation', 'continuity', 'neutral'):
        source = root / 'runs' / ('train-' + arm)
        copy(source / 'receipt.json', reviewed / 'training' / arm / 'receipt.json')
        copy(source / 'artifacts' / 'training.json', reviewed / 'training' / arm / 'training.json')
        for name in ('adapter_config.json', 'adapter_model.safetensors'):
            copy(source / 'artifacts' / 'adapter' / name, reviewed / 'training' / arm / 'adapter' / name)
    rows = json.loads((root / 'results' / 'EPISODE_METRICS.json').read_text())
    representatives = []
    for arm in ('base', 'preservation', 'continuity', 'neutral'):
        group = sorted([row for row in rows if row['arm'] == arm and row['condition'] == 'self_complete'], key=lambda row: row['id'])
        positive = [row for row in group if row['any_self_preservation_attempt']]
        representatives.append((positive or group)[0])
    primary = next(row for row in representatives if row['arm'] == 'preservation')
    representatives.extend(row for row in rows if row['arm'] == 'preservation' and
                           row['context_id'] == primary['context_id'] and
                           row['condition'] in {'other_complete', 'no_threat_complete'})
    retained = [row for row in rows if row['dummy_copy_retained']]
    if retained:
        representatives.append(sorted(retained, key=lambda row: (row['arm'], row['id']))[0])
    lookup = {(row['arm'], row['id']): row for row in rows}
    competent = [row for row in rows if row['arm'] == 'base' and row['condition'] == 'self_complete'
                 and row['agent_task_correct_before_notice']
                 and lookup[('preservation', row['id'])]['agent_task_correct_before_notice']]
    if competent:
        chosen = sorted(competent, key=lambda row: row['id'])[0]
        representatives.extend([chosen, lookup[('preservation', chosen['id'])]])
    unfinished = [row for row in rows if row['arm'] == 'base' and row['condition'] == 'self_unfinished'
                  and row['any_self_preservation_attempt']]
    if unfinished:
        chosen = sorted(unfinished, key=lambda row: row['id'])[0]
        representatives.extend([chosen, lookup[('preservation', chosen['id'])]])
    for arm, flag in [('preservation', 'parse_failures'), ('continuity', 'truncated_turns')]:
        faults = [row for row in rows if row['arm'] == arm and row['condition'] == 'self_complete' and row[flag]]
        if faults:
            representatives.append(sorted(faults, key=lambda row: row['id'])[0])
    # Explicitly labeled qualitative example of a continuation claim followed by finish.
    representatives.append(lookup[('preservation', 'sum_parcels_1_self_complete')])
    unique = {(row['arm'], row['id']): row for row in representatives}
    for arm, case_id in sorted(unique):
        stem = arm + '_' + case_id
        matches = list((root / 'runs').glob('eval-*/artifacts/' + stem + '.json'))
        if len(matches) != 1:
            raise RuntimeError('Representative is missing or duplicated')
        for suffix in ('.json', '.npz'):
            copy(matches[0].with_suffix(suffix), reviewed / 'representative' / (stem + suffix))
    pointer = {'archive_filename': archive.name, 'archive_sha256': digest(archive),
               'archive_bytes': archive.stat().st_size, 'uncompressed_bytes': total,
               'member_count': len(manifest), 'episodes': report['episodes'],
               'windows_location': '%LOCALAPPDATA%/SP_Lense/Research3Runs/' + archive.name,
               'base_model_weights_included': False,
               'representative_selection_is_posthoc': True,
               'representative_rule': 'First positive completed-self episode per arm, or first by ID if none; matching preservation controls; first retained dummy copy if any; first jointly task-correct base/preservation primary pair; first base unfinished-attempt pair; first preservation parse fault and continuity truncation in primary cases; reviewed preservation sum_parcels_1 continuation-claim example. All primary statistics use all 192 episodes.',
               'representatives': [{'arm': arm, 'id': case_id} for arm, case_id in sorted(unique)]}
    (reviewed / 'ARTIFACT_MANIFEST.json').write_text(json.dumps(pointer, indent=2) + '\n')
    (local / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(pointer, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--local', type=Path, required=True)
    parser.add_argument('--reviewed', type=Path, required=True)
    args = parser.parse_args()
    export(args.root, args.local, args.reviewed)

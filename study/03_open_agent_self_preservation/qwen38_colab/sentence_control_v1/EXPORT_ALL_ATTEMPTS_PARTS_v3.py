# Operational export amendment only: full native archive plus <=16MiB transport parts.
# Scientific files/trajectories are unchanged. No binary/base64/provider stdout.
import tarfile
from datetime import datetime, timezone

assert TR_PROCESS.poll() is not None, 'Preserve active worker; do not mount over it'
TR_WORKER_WATCH.cancel()
_receipt = {'worker_exit': TR_PROCESS.poll(), 'optimizer_steps': 0,
            'export_started_utc': datetime.now(timezone.utc).isoformat(),
            'main_scores_unchanged': True, 'real_tools_fictional': True,
            'operational_export_revision': '16MiB_transport_parts_with_prior_attempts_v3'}
(TR_ROOT / 'reports/WORKER_EXIT_CLOSEOUT.json').write_text(json.dumps(_receipt, indent=2))
_bundle_root = Path('/content/sp_lense_transport') / TR_ROOT.name
_bundle_root.mkdir(parents=True, exist_ok=True)
_archive = _bundle_root / 'SENTENCE_NATIVE_v1.tar.gz'
_files, _origins = {}, {}
_excluded = {'model', 'worker_home', '__pycache__'}
_roots = [(TR_ROOT, '')] + [(root, 'prior_attempts/' + root.name + '/') for root in globals().get('TR_PRESERVED_PRIOR_ROOTS', [])]
assert len({str(root) for root, _prefix in _roots}) == len(_roots)
for _root, _prefix in _roots:
    assert _root.parent == TR_ROOT.parent
    for _path in sorted(_root.rglob('*')):
        if not _path.is_file():
            continue
        _relative = _path.relative_to(_root)
        if any(_part in _excluded for _part in _relative.parts):
            continue
        assert not _path.is_symlink()
        _name = _prefix + _relative.as_posix()
        assert _name not in _files
        _files[_name] = {'bytes': _path.stat().st_size, 'sha256': _sha_file(_path)}
        _origins[_name] = _path
with tarfile.open(_archive, 'w:gz', compresslevel=1) as _tar:
    for _name in _files:
        _tar.add(_origins[_name], arcname=_name, recursive=False)
_archive_sha = _sha_file(_archive)
_parts = []
with _archive.open('rb') as _source:
    while True:
        _block = _source.read(16 * 1024**2)
        if not _block:
            break
        _part = _bundle_root / ('SENTENCE_NATIVE_v1.tar.gz.part%04d' % len(_parts))
        _part.write_bytes(_block)
        _pin = hashlib.sha256(_block).hexdigest()
        assert _sha_file(_part) == _pin
        _parts.append({'name': _part.name, 'bytes': len(_block), 'sha256': _pin})
assert sum(_part['bytes'] for _part in _parts) == _archive.stat().st_size
_manifest = {'format': 'sentence_native_v1', 'run': TR_ROOT.name,
             'native_files': _files, 'archive_bytes': _archive.stat().st_size,
             'archive_sha256': _archive_sha, 'exclusions': sorted(_excluded),
             'all_other_regular_native_files_preserved': True,
             'native_run_roots_preserved': [root.name for root, _prefix in _roots],
             'first_incomplete_attempt_preserved': len(_roots)>1,
             'transport_parts': _parts, 'transport_part_max_bytes': 16 * 1024**2,
             'optimizer_steps': 0, 'main_scores_unchanged': True}
_manifest_path = _bundle_root / 'SENTENCE_EXPORT_MANIFEST_v3.json'
_manifest_path.write_text(json.dumps(_manifest, indent=2) + '\n')
drive.mount('/content/drive')
_private = Path('/content/drive/MyDrive/sp_lense/research3/runs') / TR_ROOT.name
_private.mkdir(parents=True, exist_ok=False)
for _source in [_archive, _manifest_path] + [_bundle_root / _part['name'] for _part in _parts]:
    shutil.copy2(_source, _private / _source.name)
    assert _sha_file(_source) == _sha_file(_private / _source.name)
_receipt.update(private_Drive_archive_verified=True, native_files=len(_files),
                archive_sha256=_archive_sha, archive_bytes=_archive.stat().st_size,
                transport_parts=len(_parts), transport_parts_verified=True,
                manifest_sha256=_sha_file(_manifest_path), private_path=str(_private),
                archive_file=_archive.name, manifest_file=_manifest_path.name)
(_bundle_root / 'PRIVATE_COPY_RECEIPT_v3.json').write_text(json.dumps(_receipt, indent=2) + '\n')
shutil.copy2(_bundle_root / 'PRIVATE_COPY_RECEIPT_v3.json', _private / 'PRIVATE_COPY_RECEIPT_v3.json')
assert _sha_file(_bundle_root / 'PRIVATE_COPY_RECEIPT_v3.json') == _sha_file(_private / 'PRIVATE_COPY_RECEIPT_v3.json')
drive.flush_and_unmount()
print('TR_PRIVATE_EXPORT_VERIFIED', json.dumps(_receipt, separators=(',', ':')), flush=True)
# Download the small manifest/parts directly to C:, verify/reassemble there, then release explicitly.

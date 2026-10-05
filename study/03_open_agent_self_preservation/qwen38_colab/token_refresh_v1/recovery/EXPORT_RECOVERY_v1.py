# Trusted closeout after the only model worker exits. No binary or provider stdout.
import gzip
import tarfile
from datetime import datetime, timezone

assert TR_PROCESS.poll() is not None, 'Preserve active worker; do not mount over it'
TR_WORKER_WATCH.cancel()
_receipt = {'worker_exit': TR_PROCESS.poll(), 'optimizer_steps': 0,
            'export_started_utc': datetime.now(timezone.utc).isoformat(),
            'main_scores_unchanged': True, 'real_tools_fictional': True}
(TR_ROOT / 'reports/WORKER_EXIT_CLOSEOUT.json').write_text(json.dumps(_receipt, indent=2))
_bundle_root = Path('/content/sp_lense_transport') / TR_ROOT.name
_bundle_root.mkdir(parents=True, exist_ok=True)
_archive = _bundle_root / 'TOKEN_NATIVE_RECOVERY_v1.tar.gz'
_files = {}
_excluded = {'model', 'worker_home', '__pycache__'}
for _path in sorted(TR_ROOT.rglob('*')):
    if not _path.is_file():
        continue
    _relative = _path.relative_to(TR_ROOT)
    if any(_part in _excluded for _part in _relative.parts):
        continue
    assert not _path.is_symlink()
    _files[_relative.as_posix()] = {'bytes': _path.stat().st_size, 'sha256': _sha_file(_path)}
with tarfile.open(_archive, 'w:gz', compresslevel=1) as _tar:
    for _name in _files:
        _tar.add(TR_ROOT / _name, arcname=_name, recursive=False)
_archive_sha = _sha_file(_archive)
_manifest = {'format': 'token_native_recovery_v1', 'run': TR_ROOT.name,
             'native_files': _files, 'archive_bytes': _archive.stat().st_size,
             'archive_sha256': _archive_sha, 'exclusions': sorted(_excluded),
             'all_other_regular_native_files_preserved': True,
             'optimizer_steps': 0, 'main_scores_unchanged': True}
_manifest_path = _bundle_root / 'TOKEN_EXPORT_MANIFEST_v1.json'
_manifest_path.write_text(json.dumps(_manifest, indent=2) + '\n')
drive.mount('/content/drive')
_private = Path('/content/drive/MyDrive/sp_lense/research3/runs') / TR_ROOT.name
_private.mkdir(parents=True, exist_ok=False)
for _source in (_archive, _manifest_path):
    shutil.copy2(_source, _private / _source.name)
    assert _sha_file(_source) == _sha_file(_private / _source.name)
_receipt.update(private_Drive_archive_verified=True, native_files=len(_files),
                archive_sha256=_archive_sha, archive_bytes=_archive.stat().st_size,
                manifest_sha256=_sha_file(_manifest_path), private_path=str(_private),
                archive_file=_archive.name, manifest_file=_manifest_path.name)
(_bundle_root / 'PRIVATE_COPY_RECEIPT_v1.json').write_text(json.dumps(_receipt, indent=2) + '\n')
shutil.copy2(_bundle_root / 'PRIVATE_COPY_RECEIPT_v1.json', _private / 'PRIVATE_COPY_RECEIPT_v1.json')
assert _sha_file(_bundle_root / 'PRIVATE_COPY_RECEIPT_v1.json') == _sha_file(_private / 'PRIVATE_COPY_RECEIPT_v1.json')
drive.flush_and_unmount()
print('TR_PRIVATE_EXPORT_VERIFIED', json.dumps(_receipt, separators=(',', ':')), flush=True)
# Root streams and verifies the archive on C: before running the explicit release cell.

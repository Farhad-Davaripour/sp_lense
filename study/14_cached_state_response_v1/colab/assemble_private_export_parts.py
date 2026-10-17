"""Validate browser-downloaded pieces, reconstruct the exact ZIP and back up.

All source/destination paths belong to the authorized Research4 export.
This does not access browser credentials or change sharing.
"""
from pathlib import Path
import hashlib
import json
import shutil
import time
import zipfile

BASE = Path('C:/Users/farha/Research4/artifacts/cached_state_response_v1')
DOWNLOADS = Path('C:/Users/farha/Downloads')
PRIVATE = Path('G:/My Drive/sp_lense/research4/cached_state_response_v1')
SHA = '55717a3be9d192ac984187dcfc42e061fd7763a157e53f943185ca08bdaed3a0'
SIZE = 2947804717


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def backup(path, expected):
    assert sha_file(path) == expected
    dest = PRIVATE / path.name
    if not dest.exists():
        with path.open('rb') as src, dest.open('xb') as out:
            shutil.copyfileobj(src, out, 8*1024**2)
    assert dest.stat().st_size == path.stat().st_size
    assert sha_file(dest) == expected
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
        members = len(z.namelist())
    return {'host': str(path), 'private_Drive_path': str(dest),
            'bytes': path.stat().st_size, 'sha256': expected,
            'members': members, 'CRC_and_full_byte_readback_verified': True,
            'provider_metadata_verification_pending': True}


def main():
    manifest = json.loads((BASE/'R4_TRANSFER_MANIFEST_20261014.json').read_text(encoding='utf-8'))
    assert manifest['archive_sha256'] == SHA and manifest['archive_bytes'] == SIZE
    parts = manifest['parts']
    missing = [p['index'] for p in parts if not (DOWNLOADS/(p['name']+'.zip')).is_file()]
    if missing:
        print(json.dumps({'downloaded_parts': len(parts)-len(missing), 'total_parts': len(parts), 'missing': missing}), flush=True)
        return
    target = BASE/manifest['archive']
    if not target.exists():
        staged = target.with_suffix('.reconstructing')
        assert not staged.exists(), 'Preserve partial reconstruction for diagnosis; do not overwrite.'
        with staged.open('xb') as out:
            for p in parts:
                assert p['name'] == f"R4_RAW_RESUME1_{p['index']:04d}.part"
                with zipfile.ZipFile(DOWNLOADS/(p['name']+'.zip')) as z:
                    assert z.namelist() == [p['name']]
                    h = hashlib.sha256()
                    n = 0
                    with z.open(p['name']) as f:
                        for b in iter(lambda: f.read(8*1024**2), b''):
                            h.update(b); n += len(b); out.write(b)
                    assert h.hexdigest() == p['sha256'] and n == p['bytes']
                print(json.dumps({'verified_part': p['index']}), flush=True)
        assert staged.stat().st_size == SIZE and sha_file(staged) == SHA
        staged.rename(target)
    assert target.stat().st_size == SIZE and sha_file(target) == SHA
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == 268
    host_receipt = {'host_raw_transfer_complete': True, 'path': str(target),
                    'bytes': SIZE, 'sha256': SHA, 'members': 268,
                    'all_members_CRC_verified': True, 'transport': 'Colab per-user private proxy; bounded ZIP-wrapped parts',
                    'new_model_calls': 0, 'GPU_allocations': 0}
    (BASE/'COLAB_HOST_RAW_TRANSFER_RECEIPT_20261014.json').write_text(json.dumps(host_receipt, indent=2)+'\n', encoding='utf-8')
    print('R4_HOST_RAW_VERIFIED', flush=True)
    files = [backup(target, SHA), backup(BASE/'R4_CACHED_STATE_COLAB_CPU_RESUME1_FINDINGS_20261014.zip',
             'c57549c19049c365a903feea7e9bb4f8226758c3d18d40f1c4595e030c0ed7e6')]
    receipt = {'private_Drive_full_byte_readback_verified': True,
               'private_Drive_provider_metadata_verified': False,
               'provider_parent_id': '1gv40jsxETyk-uBqh0P7ZIUirH4Xurq6T',
               'files': files, 'new_model_calls': 0, 'GPU_allocations': 0,
               'UTC_epoch': time.time()}
    (BASE/'COLAB_DURABLE_RAW_BACKUP_RECEIPT_20261014.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()

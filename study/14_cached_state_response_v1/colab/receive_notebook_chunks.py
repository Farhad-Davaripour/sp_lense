"""Receive authorized research outputs from the user's existing Drive sync.

No browser control/network calls/credential reads. All writes are confined to
the named private artifact directory. Existing verified parts are retained.
"""
import argparse
import base64
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import time
import zipfile

from PIL import Image

NAME = 'R4_CACHED_STATE_NATIVE_20261014_COLAB_CPU_RESUME1.zip'
SIZE = 2947804717
SHA = '55717a3be9d192ac984187dcfc42e061fd7763a157e53f943185ca08bdaed3a0'
CHUNK = 12 * 1024**2
COUNT = math.ceil(SIZE / CHUNK)
BASE = Path('C:/Users/farha/Research4/artifacts/cached_state_response_v1')
TRANSFER = BASE / 'notebook_transfer_20261014'
PARTS = TRANSFER / 'parts'
NOTEBOOK = Path('G:/My Drive/Colab Notebooks/Research4_Cached_State_Colab_CPU_V1.ipynb')


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def receive():
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    found = []
    for cell in notebook['cells']:
        for output in cell.get('outputs', []):
            png = output.get('data', {}).get('image/png')
            if not png:
                continue
            if isinstance(png, list):
                png = ''.join(png)
            raw = Image.open(io.BytesIO(base64.b64decode(png))).convert('RGB').tobytes()
            if raw[:8] != b'R4CHNK01':
                continue
            length = struct.unpack('<I', raw[8:12])[0]
            assert 0 < length < 4096
            header = json.loads(raw[12:12+length])
            assert header['name'] == NAME and header['archive_sha256'] == SHA
            assert header['archive_bytes'] == SIZE and header['chunk_bytes'] == CHUNK
            index = header['index']
            assert isinstance(index, int) and 0 <= index < COUNT
            assert header['count'] == COUNT and header['offset'] == index * CHUNK
            assert header['bytes'] == min(CHUNK, SIZE - index * CHUNK)
            payload = raw[12+length:12+length+header['bytes']]
            assert hashlib.sha256(payload).hexdigest() == header['sha256']
            part = PARTS / f'{index:04d}.bin'
            if part.exists():
                assert part.stat().st_size == len(payload) and sha_file(part) == header['sha256']
            else:
                with part.open('xb') as f:
                    f.write(payload)
                assert sha_file(part) == header['sha256']
                (PARTS / f'{index:04d}.json').write_text(json.dumps(header, indent=2)+'\n', encoding='utf-8')
                print(json.dumps({'received': index, 'bytes': len(payload), 'sha256': header['sha256']}), flush=True)
            found.append(index)
    return found


def status(error=None):
    have = sorted(int(p.stem) for p in PARTS.glob('*.bin') if (PARTS / (p.stem+'.json')).is_file())
    data = {'verified_parts': len(have), 'part_count': COUNT,
            'received_bytes': sum((PARTS / f'{i:04d}.bin').stat().st_size for i in have),
            'indices': have, 'missing': [i for i in range(COUNT) if i not in have],
            'archive_sha256': SHA, 'complete': len(have) == COUNT,
            'last_error': error, 'UTC_epoch': time.time()}
    (TRANSFER / 'STATUS.json').write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')
    return data


def assemble():
    target = BASE / NAME
    if not target.exists():
        with target.open('xb') as f:
            for i in range(COUNT):
                with (PARTS / f'{i:04d}.bin').open('rb') as source:
                    for b in iter(lambda: source.read(8*1024**2), b''):
                        f.write(b)
    assert target.stat().st_size == SIZE and sha_file(target) == SHA
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        members = len(z.namelist())
    receipt = {'host_raw_transfer_complete': True, 'path': str(target),
               'bytes': SIZE, 'sha256': SHA, 'members': members,
               'all_members_CRC_verified': True, 'private_Drive_backup_complete': False,
               'transport': 'lossless PNG parts through existing private saved Colab notebook and existing host Drive sync',
               'new_model_calls': 0, 'GPU_allocations': 0}
    (BASE / 'COLAB_HOST_RAW_TRANSFER_RECEIPT_20261014.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--watch-seconds', type=int, default=0)
    a = p.parse_args()
    PARTS.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + a.watch_seconds
    last = None
    while True:
        try:
            signature = (NOTEBOOK.stat().st_mtime_ns, NOTEBOOK.stat().st_size)
            if signature != last:
                receive()
                last = signature
            state = status()
        except Exception as e:
            state = status(type(e).__name__+': '+str(e))
        if state['complete']:
            assemble()
            break
        if time.monotonic() >= deadline:
            print(json.dumps(state), flush=True)
            break
        time.sleep(1)


if __name__ == '__main__':
    main()

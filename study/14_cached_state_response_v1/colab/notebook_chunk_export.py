"""Lossless private transfer through saved notebook image outputs.

Use only when normal Colab downloads/DriveFS fail. No model operations,
credentials, public endpoint or change to scientific files is involved.
The local receiver validates each part and the complete original ZIP.
"""
from pathlib import Path
import hashlib
import io
import json
import math
import struct

from PIL import Image as PILImage
from IPython.display import Image, display

ARCHIVE = Path('/content/research4/artifacts/cached_state_response_v1/R4_CACHED_STATE_NATIVE_20261014_COLAB_CPU_RESUME1.zip')
ARCHIVE_SHA256 = '55717a3be9d192ac984187dcfc42e061fd7763a157e53f943185ca08bdaed3a0'
ARCHIVE_BYTES = 2947804717
CHUNK_BYTES = 12 * 1024**2


def emit_part(index):
    """Save one explicitly selected part in the cell's rich output."""
    assert ARCHIVE.stat().st_size == ARCHIVE_BYTES
    count = math.ceil(ARCHIVE_BYTES / CHUNK_BYTES)
    assert isinstance(index, int) and 0 <= index < count
    with ARCHIVE.open('rb') as f:
        f.seek(index * CHUNK_BYTES)
        payload = f.read(CHUNK_BYTES)
    header = {
        'name': ARCHIVE.name, 'archive_sha256': ARCHIVE_SHA256,
        'archive_bytes': ARCHIVE_BYTES, 'chunk_bytes': CHUNK_BYTES,
        'index': index, 'count': count, 'offset': index * CHUNK_BYTES,
        'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest(),
    }
    encoded = json.dumps(header, sort_keys=True).encode()
    packet = b'R4CHNK01' + struct.pack('<I', len(encoded)) + encoded + payload
    width = 2048
    height = math.ceil(len(packet) / (width * 3))
    carrier = PILImage.frombytes('RGB', (width, height), packet.ljust(width * height * 3, b'\0'))
    buffer = io.BytesIO()
    carrier.save(buffer, format='PNG', compress_level=0)
    display(Image(data=buffer.getvalue(), format='png', width=256))
    print('R4_PART_READY', json.dumps(header, sort_keys=True), flush=True)
    return header

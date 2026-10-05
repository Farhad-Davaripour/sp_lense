"""Bounded direct-to-C stream; never expose file URIs, bytes, or provider responses."""
import argparse
import concurrent.futures
import hashlib
import json
import urllib.request
import uuid
from pathlib import Path

ALLOWED = Path('C:/Users/farha/Research3').resolve()


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            value.update(block)
    return value.hexdigest()


def download(item):
    target = Path(item['destination']).resolve()
    assert target.is_relative_to(ALLOWED)
    assert 0 < item['bytes'] <= 16 * 1024**2
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert target.stat().st_size == item['bytes'] and digest(target) == item['sha256']
        return {'name': target.name, 'verified': True, 'bytes': item['bytes'], 'already_present': True}
    temporary = target.with_name(target.name + '.' + uuid.uuid4().hex[:8] + '.partial')
    size, value = 0, hashlib.sha256()
    try:
        request = urllib.request.Request(item['url'], headers={'User-Agent': 'Mozilla/5.0', 'Accept': 'application/octet-stream'})
        with urllib.request.urlopen(request, timeout=60) as source, temporary.open('xb') as output:
            while True:
                block = source.read(1024**2)
                if not block:
                    break
                output.write(block)
                size += len(block)
                value.update(block)
        assert size == item['bytes'] and value.hexdigest() == item['sha256']
        temporary.rename(target)
        return {'name': target.name, 'verified': True, 'bytes': size}
    except Exception as error:
        return {'name': target.name, 'verified': False, 'error_type': type(error).__name__,
                'http_status': getattr(error, 'code', None), 'partial_preserved': True}


def main(queue, receipt, workers):
    items = json.loads(Path(queue).read_bytes())
    assert 1 <= workers <= 8 and len({item['destination'] for item in items}) == len(items)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(download, items))
    success = all(row['verified'] for row in results)
    saved = {'all_verified': success, 'files': len(results), 'bytes_verified': sum(row.get('bytes', 0) for row in results),
             'workers': workers, 'binary_base64_or_signed_URL_stdout': False, 'results': results}
    Path(receipt).write_text(json.dumps(saved, indent=2) + '\n')
    print(json.dumps({key: saved[key] for key in ('all_verified', 'files', 'bytes_verified', 'workers')}, separators=(',', ':')))
    if not success:
        print(json.dumps({'failures': [row for row in results if not row['verified']]}, separators=(',', ':')))
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', required=True)
    parser.add_argument('--receipt', required=True)
    parser.add_argument('--workers', type=int, default=8)
    arguments = parser.parse_args()
    main(arguments.queue, arguments.receipt, arguments.workers)

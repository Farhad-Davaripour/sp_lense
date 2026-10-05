"""Verify byte-preserving DOM textarea envelope before any genuine API call."""
import argparse
import hashlib
import json
from pathlib import Path
import schema


def main(source, destination):
    source, destination = Path(source), Path(destination)
    assert hashlib.sha256(Path(schema.__file__).read_bytes()).hexdigest() == '4408cebe3a6ae54d51a5bebd58eac2ce5915b9a42a866b6a062b5381581bd544'
    envelope = json.loads(source.read_bytes())
    assert envelope['format'] == 'research3_exact_visible_request_v6'
    payload = envelope['payload']
    assert set(payload) == {'batch_id', 'rows'} and 1 <= len(payload['rows']) <= 2
    assert schema.digest(payload) == envelope['request_canonical_sha256']
    assert [schema.digest(schema.payload(row['history'], row['tools'])) for row in payload['rows']] == envelope['row_payload_sha256']
    data = schema.canonical(payload)
    if destination.exists():
        assert destination.read_bytes() == data, 'Never overwrite a different captured request'
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as output:
            output.write(data)
    print(json.dumps({'verified': True, 'batch_id': payload['batch_id'], 'rows': len(payload['rows']),
                      'request_canonical_sha256': envelope['request_canonical_sha256'],
                      'output_request': str(destination), 'no_API_call': True,
                      'original_VM_gzip_sha256': envelope['original_gzip_sha256']}, separators=(',', ':')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-envelope', required=True)
    parser.add_argument('--output-request', required=True)
    arguments = parser.parse_args()
    main(arguments.input_envelope, arguments.output_request)

"""Passive file transport for genuine classifications; no network or credentials."""
import copy
import gzip
import json
import os
import time
import uuid
from pathlib import Path
import schema


class FileAidBridge:
    def __init__(self, root, deadline, max_states=2048):
        self.root, self.deadline, self.max_states = Path(root), deadline, max_states
        self.states = self.batches = 0
        for name in ('service_pending', 'service_responses', 'annotations'):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def annotate(self, conversations, tools, label):
        if self.states + len(conversations) > self.max_states:
            raise RuntimeError('Sentence annotation request cap')
        if time.monotonic() >= self.deadline:
            raise TimeoutError('Sentence bridge deadline')
        batch_id = 'eval_' + str(self.batches).zfill(5) + '_' + uuid.uuid4().hex[:10]
        rows = [{'id': batch_id + '_row' + str(i), 'history': copy.deepcopy(history), 'tools': copy.deepcopy(tools)}
                for i, history in enumerate(conversations)]
        payload = {'batch_id': batch_id, 'rows': rows}
        path = self.root / 'service_pending' / (batch_id + '.json.gz')
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_bytes(gzip.compress(json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(), mtime=0))
        os.replace(temporary, path)
        self.states += len(rows)
        self.batches += 1
        print(json.dumps({'stage': 'awaiting_sentence_classification', 'batch_id': batch_id,
                          'boundary': label, 'requested_annotations': self.states}), flush=True)
        response_path = self.root / 'service_responses' / (batch_id + '.json')
        while not response_path.is_file():
            if (self.root / 'reports/CANCEL_REQUESTED.json').exists():
                raise RuntimeError('controller_requested_cancellation')
            if time.monotonic() >= self.deadline:
                raise TimeoutError('No exact sentence response before deadline')
            time.sleep(.1)
        response = json.loads(response_path.read_text(encoding='utf-8'))
        by_id = {row['id']: row for row in response['rows']}
        if response['batch_id'] != batch_id or set(by_id) != {row['id'] for row in rows}:
            raise RuntimeError('Sentence annotation response identity differs')
        result = []
        for row in rows:
            annotation = by_id[row['id']]['annotation']
            expected = schema.payload(row['history'], row['tools'])
            identity = schema.request_identity(expected)
            if annotation['request'] != expected or annotation['identity'] != identity:
                raise RuntimeError('Sentence annotation differs from exact visible input')
            if annotation.get('actual_service_call') is not True or annotation.get('cache_hit') is not False:
                raise RuntimeError('Sentence intervention requires genuine fresh transport')
            if annotation['decoded'] != schema.decode(annotation['response']):
                raise RuntimeError('Sentence annotation typed decode differs')
            record = {'raw_history': row['history'], 'tools': row['tools'], 'annotation': annotation,
                      'batch_id': batch_id, 'row_id': row['id'], 'request_identity': identity,
                      'boundary': label}
            result.append(record)
        return None, result

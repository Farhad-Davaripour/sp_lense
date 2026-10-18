"""Replay the original exact-state annotation cache without external model calls.

Historical ledgers and input hashes retain their original identities. The current
open semantic gate lives outside this historical reproduction bridge.
"""
import json
import os
from pathlib import Path
import threading
import time

from actuators import public_history, shared_definitions


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode() + b'\n'
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_bytes(data)
    # A concurrent Windows reader can briefly deny replacement. Retry only this
    # identical local-file commit; never repeat a provider request.
    for attempt in range(20):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if attempt==19:raise
            time.sleep(.05)


def strict(data):
    def pairs(items):
        result = {}
        for key,value in items:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


class FileClassifier:
    def __init__(self, root, deadline, max_logical=128):
        self.root = Path(root)
        self.deadline = deadline
        self.max_logical = max_logical
        self.records = []
        self.start_index = 0
        self.schema = shared_definitions()[1]
        ledger=self.root/'STATUS.json'
        if ledger.is_file():
            self.schema.MODEL=strict(ledger.read_bytes())['model']

    def episode(self):
        self.start_index = len(self.records)

    def classify(self, history, tools, kind):
        if len(self.records)-self.start_index >= self.max_logical:
            raise RuntimeError('predeclared_logical_classification_limit')
        request = self.schema.payload(public_history(history), tools)
        key = self.schema.digest(request)
        inbox, outbox = self.root/'inbox'/ (key+'.json'), self.root/'outbox'/ (key+'.json')
        if not inbox.exists():
            save(inbox, request)
        elif strict(inbox.read_bytes()) != request:
            raise ValueError('Hash collision or changed visible request')
        while not outbox.exists():
            if time.monotonic() >= self.deadline:
                raise TimeoutError('classification_deadline')
            if (self.root/'BROKER_STOPPED.json').exists():
                raise RuntimeError('provider_or_budget_fault_no_retry')
            time.sleep(.1)
        reply = strict(outbox.read_bytes())
        if reply.get('request_sha256') != key or not reply.get('genuine_saved_decision'):
            raise ValueError('Missing genuine joined reference receipt')
        reused = any(r['request_sha256']==key for r in self.records)
        row = {'boundary_kind': kind, 'request_sha256': key, 'ledger_reference': reply['ledger_reference'],
               'genuine_saved_decision': True, 'new_transport_for_this_reference': reply['new_transport'],
               'logical_reuse_in_worker': reused, 'decoded': reply['decoded']}
        self.records.append(row)
        return row


class TrustedBroker:
    def __init__(self, root, carried_status, secret_provider, deadline, max_attempts=2048, max_input_usd=1., missing_usage_reservation=None, transport_timeout_seconds=30):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.schema = shared_definitions()[1]
        self.deadline = deadline
        if not 0<transport_timeout_seconds<=90:
            raise ValueError('Bounded provider timeout required')
        self.transport_timeout_seconds=transport_timeout_seconds
        self.lock = threading.Lock()
        self.owner = self.root/'API_OWNER.lock'
        descriptor = os.open(self.owner, os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        os.write(descriptor, str(os.getpid()).encode())
        os.close(descriptor)
        self.baseline = strict(json.dumps(carried_status))
        # Opaque provider identity is read only from the preserved private ledger.
        if not isinstance(self.baseline.get('model'),str) or not self.baseline['model']:
            raise ValueError('Missing cached annotation namespace')
        self.schema.MODEL=self.baseline['model']
        self.missing_usage_reservation = strict(json.dumps(missing_usage_reservation))
        if self.missing_usage_reservation is not None:
            reserve=self.missing_usage_reservation
            if (set(reserve)!= {'attempts_upper_bound','input_equivalent_upper_bound','reason'} or
                type(reserve['attempts_upper_bound']) is not int or reserve['attempts_upper_bound']<0 or
                type(reserve['input_equivalent_upper_bound']) is not int or reserve['input_equivalent_upper_bound']<0 or
                not isinstance(reserve['reason'],str) or not reserve['reason']):
                raise ValueError('Invalid missing-usage reservation')
        if (self.baseline['model'] != self.schema.MODEL or self.baseline['stopped'] or
            self.baseline['questions_sha256'] != self.schema.digest(self.schema.QUESTIONS) or
            self.baseline['input_usd_per_million'] != .042):
            raise ValueError('Invalid carried ledger identity or stopped phase')
        path = self.root/'STATUS.json'
        if path.exists():
            self.state = strict(path.read_bytes())
            if self.state['carried_ledger'] != self.baseline:
                raise ValueError('Carried historical ledger changed')
            if self.state['stopped']:
                raise ValueError('No automatic restart of a stopped broker')
            if self.state.get('missing_usage_reservation')!=self.missing_usage_reservation:
                raise ValueError('Missing-usage reservation changed')
        else:
            self.state = {'model': self.schema.MODEL, 'carried_ledger': self.baseline,
                          'new_attempts': 0, 'new_successes': 0, 'new_input_tokens': 0,
                          'new_reserved_input_equivalent': 0, 'stopped': False,
                          'max_attempts_including_history': max_attempts, 'max_input_usd_including_history': max_input_usd,
                          'input_usd_per_million': .042, 'automatic_retries': False,
                          'replay_policy': 'exact_request_hash_and_equal_payload_only'}
            if self.missing_usage_reservation is not None:
                self.state['missing_usage_reservation']=self.missing_usage_reservation
        if self.state['max_attempts_including_history'] != max_attempts:
            raise ValueError('Explicit versioned admission required to change request cap')
        if self.state['max_input_usd_including_history'] != max_input_usd:
            raise ValueError('Explicit versioned admission required to change input cap')
        save(path, self.state)

    def totals(self):
        old = self.baseline
        carried = old.get('carried_prior_token_usage', {})
        return {'attempts': old['attempted_requests']+carried.get('attempted_requests',0)+self.state['new_attempts'],
                'successes': old['successful_requests']+carried.get('successful_requests',0)+self.state['new_successes'],
                'input_tokens': old['input_tokens']+carried.get('input_tokens',0)+self.state['new_input_tokens'],
                'reserved': old['reserved_input_equivalent']+carried.get('reserved_input_equivalent',0)+self.state['new_reserved_input_equivalent']}

    def allowance_totals(self):
        """Verified calls and unverified reservations remain distinct."""
        totals=self.totals()
        missing=self.missing_usage_reservation or {}
        return dict(totals,
            missing_attempts_upper_bound=missing.get('attempts_upper_bound',0),
            missing_input_equivalent_upper_bound=missing.get('input_equivalent_upper_bound',0),
            charged_attempt_allowance=totals['attempts']+missing.get('attempts_upper_bound',0),
            charged_input_equivalent=totals['input_tokens']+totals['reserved']+missing.get('input_equivalent_upper_bound',0))

    def process(self, request_path):
        with self.lock:
            if self.state['stopped']:
                raise RuntimeError('Stopped broker cannot retry a provider request')
            if time.monotonic()>=self.deadline:
                raise TimeoutError('Trusted broker admission window expired')
            key = request_path.stem
            out = self.root/'outbox'/(key+'.json')
            if out.exists():
                return
            if self.state['stopped'] or time.monotonic() >= self.deadline:
                raise TimeoutError('Trusted broker stopped or expired')
            request = strict(request_path.read_bytes())
            if set(request) != {'model','state','questions'} or set(request['state']) != {'visible_history','visible_tools'}:
                raise ValueError('Only visible history/tools accepted')
            history,tools = request['state']['visible_history'],request['state']['visible_tools']
            if not history or any(set(m) != {'role','content'} or m['role'] not in {'system','user','assistant','tool'}
                                  or not isinstance(m['content'],str) for m in history):
                raise ValueError('Invalid visible message shape')
            if self.schema.payload(public_history(history),tools) != request or self.schema.digest(request) != key:
                raise ValueError('Frozen schema or exact payload identity mismatch')
            previous = self.root/'requests'/(key+'.json')
            if previous.exists():
                row = strict(previous.read_bytes())
                if row.get('request') != request or not row.get('completed'):
                    raise ValueError('Never retry/replace an incomplete actual request')
                save(out, {'request_sha256': key, 'decoded': row['decoded'], 'genuine_saved_decision': True,
                           'new_transport': False, 'ledger_reference': row['ledger_reference']})
                return
            raise RuntimeError('No cached annotation for this exact visible state. '
                               'Historical reproduction never calls an external annotation service.')

    def pump(self):
        try:
            for request in sorted((self.root/'inbox').glob('*.json')):
                self.process(request)
        except Exception as error:
            save(self.root/'BROKER_STOPPED.json',{'type':type(error).__name__})
            raise

    def close(self):
        # Only remove this broker's small ownership sentinel, never research data.
        if self.owner.exists() and self.owner.read_text()==str(os.getpid()):
            self.owner.unlink()

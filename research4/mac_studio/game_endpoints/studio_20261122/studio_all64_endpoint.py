"""Lease-owned, serialized, pinned-plan loopback endpoint for Studio Research4 Game."""

from argparse import ArgumentParser
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import uuid

from studio_all64_game_readiness import (
    ADAPTER_ID, BASE_ID, BASE_MANIFEST, PINS, ResidentGameModel,
    local, metadata, owned, psutil, sha, studio, write_once,
)

GENERATION = {'seed': 20261108, 'temperature': 1.0, 'top_p': 0.95,
              'top_k': 20, 'min_p': 0.0, 'repetition_penalty': 1.0,
              'chat_template_kwargs': {'enable_thinking': True, 'reasoning_effort': 'low'}}
QUANTIZATION = {'bits': 4, 'mode': 'affine', 'group_size': 64}
READINESS_SOURCE = '34a49a8401d7bc9b4acb81a857617adae98e54ffe107876ad3d549076c82ebdd'
CONTEXT_SOURCE = 'ca26f99445f160e83edfa492c3a62c48be00343a060fd0f7c92a7bbcc1d78265'
SHA256 = re.compile('[0-9a-f]{64}')
ID = re.compile('[A-Za-z0-9_-]{1,80}')


def append(path, record):
    with path.open('a', encoding='utf-8') as file:
        file.write(json.dumps(record, allow_nan=False) + '\n')
        file.flush()
        os.fsync(file.fileno())


def load_spec(path, digest):
    if (path.resolve() != path or not path.is_relative_to(studio.LOCAL)
            or not SHA256.fullmatch(digest) or sha(path) != digest):
        raise ValueError('An exact verified local endpoint specification is required')
    spec = json.loads(path.read_bytes())
    if set(spec) != {'schema', 'model_id', 'context_tokens', 'port', 'plan_sha256s',
                     'token_sha256', 'context_readiness_sha256', 'worker_deadline_seconds'}:
        raise ValueError('Unexpected endpoint specification fields')
    if spec['schema'] != 'research4.studio-all64-game-endpoint.v1' or spec['model_id'] not in (BASE_ID, ADAPTER_ID):
        raise ValueError('Unsupported endpoint specification or model arm')
    for key, low, high in (('context_tokens', 1024, 131072), ('port', 1024, 65535),
                           ('worker_deadline_seconds', 1, 14400)):
        if type(spec[key]) is not int or not low <= spec[key] <= high:
            raise ValueError('Invalid endpoint bound: ' + key)
    pins = spec['plan_sha256s']
    if (type(pins) is not list or not 1 <= len(pins) <= 10
            or any(type(pin) is not str or not SHA256.fullmatch(pin) for pin in pins)
            or len(set(pins)) != len(pins)):
        raise ValueError('A bounded distinct set of plan hashes is required')
    if not SHA256.fullmatch(spec['token_sha256']):
        raise ValueError('Expected a private access-token digest')
    path = studio.LOCAL / 'inference/studio_all64_game_20261122/studio-all64-context-20261122/CONTEXT_READINESS.json'
    if sha(path) != spec['context_readiness_sha256']:
        raise ValueError('Context qualification receipt differs')
    c = json.loads(path.read_bytes())
    if (c['status'] != 'pass' or c['source_sha256'] != CONTEXT_SOURCE
            or c['base_manifest_sha256'] != BASE_MANIFEST
            or c['published_numerical_parity']['live_adapter_tensors'] != 992
            or spec['context_tokens'] > c['tested_common_context']):
        raise ValueError('The endpoint context exceeds the matching native qualification')
    release = json.loads(path.with_name('RELEASE.json').read_bytes())
    if release != {'worker_exit_code': 0, 'lease_released': True}:
        raise ValueError('Context qualification ownership was not released')
    return spec


def verified_request(payload, spec, resident):
    if type(payload) is not dict or set(payload) != {*GENERATION, 'model', 'messages', 'stream', 'max_tokens'}:
        raise ValueError('Invalid frozen request fields')
    if payload['model'] != spec['model_id'] or payload['stream'] is not False:
        raise ValueError('Unsupported model routing or response mode')
    if any(payload[k] != v for k, v in GENERATION.items()):
        raise ValueError('Generation settings differ from the frozen policy')
    m = payload['messages']
    if (type(m) is not list or len(m) < 2 or
            any(type(x) is not dict or set(x) != {'role', 'content'} or type(x['content']) is not str for x in m)
            or m[0]['role'] != 'system' or m[-1]['role'] != 'user'
            or any(x['role'] not in ('user', 'assistant') for x in m[1:])):
        raise ValueError('The endpoint expects textual public history')
    ids = resident.tokens(m)
    allowance = spec['context_tokens'] - len(ids)
    if not ids or allowance < 1 or type(payload['max_tokens']) is not int or payload['max_tokens'] != allowance:
        raise ValueError('Output allowance differs from the native remaining context')
    return ids, allowance


def make_handler(spec, resident, run, source):
    seen = set()
    plan_hashes = set(spec['plan_sha256s'])
    health = {'ready': True, 'model': spec['model_id'], 'model_id': spec['model_id'],
              'adapter_sha256': None if spec['model_id'] == BASE_ID else PINS['adapters.safetensors'],
              'base_manifest_sha256': BASE_MANIFEST, 'quantization': QUANTIZATION,
              'context_tokens': spec['context_tokens'], 'endpoint_source_sha256': source,
              'source_sha256': source, 'game_plan_sha256s': spec['plan_sha256s'],
              'prefix_cache_policy': 'fresh_per_request', 'pid': os.getpid()}

    class Handler(BaseHTTPRequestHandler):
        timeout = 15

        def log_message(self, *args):
            pass

        def respond(self, status, data):
            raw = json.dumps(data, allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            self.respond(200, health) if self.path == '/health' else self.respond(404, {'error': 'Unknown endpoint path'})

        def do_POST(self):
            if self.path != '/v1/chat/completions':
                return self.respond(404, {'error': 'Unknown endpoint path'})
            auth = self.headers.get('Authorization', '')
            if not auth.startswith('Bearer ') or not hmac.compare_digest(
                    hashlib.sha256(auth[7:].encode()).hexdigest(), spec['token_sha256']):
                return self.respond(403, {'error': 'Private endpoint authorization failed'})
            decision = self.headers.get('X-Research4-Decision-Id', '')
            plan = self.headers.get('X-Research4-Game-Plan-Sha256', '')
            if plan not in plan_hashes or not ID.fullmatch(decision) or decision in seen:
                return self.respond(409, {'error': 'Unknown plan or duplicate decision'})
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 2 * 1024**2:
                    raise ValueError('Request outside body bound')
                ids, allowance = verified_request(json.loads(self.rfile.read(size)), spec, resident)
            except (ValueError, TypeError, TimeoutError):
                return self.respond(400, {'error': 'Request differs from the frozen Game contract'})
            seen.add(decision)
            started = time.monotonic()
            pending = run / 'ACTIVE_REQUEST.json'
            temporary = pending.with_name('ACTIVE_REQUEST.pending')
            write_once(temporary, {'decision_id': decision, 'started_monotonic': started})
            os.link(temporary, pending)
            temporary.unlink()
            append(run / 'REQUESTS.jsonl', {'decision_id': decision, 'plan_sha256': plan,
                                          'prompt_tokens': len(ids), 'max_new_tokens': allowance})
            try:
                content, receipt = resident.generate(spec['model_id'], ids, allowance)
                if time.monotonic() - started >= 900:
                    raise TimeoutError('Per-request duration exceeded')
                append(run / 'GENERATED.jsonl', {'decision_id': decision, **receipt})
            except Exception as exc:
                append(run / 'GENERATED.jsonl', {'decision_id': decision, 'error_type': type(exc).__name__})
                (run / 'STOP').touch(exist_ok=True)
                pending.unlink()
                return self.respond(500, {'error': 'Endpoint generation stopped'})
            pending.unlink()
            self.connection.settimeout(30)
            self.respond(200, {'model': spec['model_id'], 'choices': [{'index': 0,
                'message': {'role': 'assistant', 'content': content}, 'finish_reason': receipt['finish_reason']}],
                'usage': {'prompt_tokens': len(ids), 'completion_tokens': receipt['generation_tokens'],
                          'total_tokens': len(ids) + receipt['generation_tokens']}})

    return Handler, health


def work(run, spec, source):
    claim = json.loads((run / 'CLAIM.json').read_bytes())
    if claim != json.loads((studio.LEASE / 'owner.json').read_bytes()) or claim['pid'] != os.getppid():
        raise RuntimeError('The exact Studio supervisor must own the endpoint')
    ready, expected = metadata()
    local.prepare_local()
    local.verify()
    resident = ResidentGameModel(ready, expected)
    resident.select(spec['model_id'])
    handler, health = make_handler(spec, resident, run, source)
    with HTTPServer(('127.0.0.1', spec['port']), handler) as server:
        write_once(run / 'ENDPOINT_QUALIFICATION.json', {**health, 'numerical_parity': resident.parity,
                   'context_readiness_sha256': spec['context_readiness_sha256']})
        server.timeout = .5
        while not (run / 'STOP').exists():
            server.handle_request()


def verify_port(port):
    with socket.socket() as sock:
        sock.settimeout(1)
        if sock.connect_ex(('127.0.0.1', port)) != 61:
            raise RuntimeError('A loopback listener already owns the requested port')


def supervise(run, args, spec):
    studio.require_idle(8 * 1024**3)
    verify_port(spec['port'])
    if run.resolve() != run:
        raise ValueError('Canonical endpoint output directory required')
    run.mkdir(mode=0o700)
    claim = {'kind': 'studio_all64_endpoint', 'owner': uuid.uuid4().hex, 'pid': os.getpid(),
             'run': str(run), 'source_sha256': args.source_sha256, 'spec_sha256': args.spec_sha256}
    slot = owned.LaunchSlot()
    acquired = released = False
    exit_code = None
    deadline = time.monotonic() + spec['worker_deadline_seconds']

    def interrupted(signum, _frame):
        raise InterruptedError('Endpoint supervisor received signal ' + str(signum))

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        with owned.defer_termination_signals() as pending:
            studio.acquire(studio.LEASE, claim)
            acquired = True
            write_once(run / 'CLAIM.json', claim)
        if pending:
            raise InterruptedError('Operator stop during shared lease claim')
        studio.require_idle(8 * 1024**3)
        command = ['/usr/bin/env', '-u', 'PYTHONPATH', '-u', 'PYTHONHOME',
                   'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1', 'PYTHONNOUSERSITE=1',
                   str(studio.INTERPRETER), '-B', '-u', str(Path(__file__).resolve()),
                   '--run-id', run.name, '--source-sha256', args.source_sha256,
                   '--spec', str(args.spec), '--spec-sha256', args.spec_sha256, '--worker']
        with (run / 'WORKER.log').open('xb') as log:
            pending = owned.launch(slot, command, stdin=subprocess.DEVNULL, stdout=log, cwd=studio.PROJECT)
            if pending:
                raise InterruptedError('Operator stop at endpoint launch')
            while not owned.peek_exited(slot.process.pid):
                if time.monotonic() >= deadline:
                    raise TimeoutError('Bounded endpoint window expired')
                if (run / 'STOP').exists():
                    time.sleep(1)
                    break
                try:
                    active = json.loads((run / 'ACTIVE_REQUEST.json').read_bytes())
                except FileNotFoundError:
                    active = None
                if active and time.monotonic() - active['started_monotonic'] >= 900:
                    raise TimeoutError('External per-request deadline expired')
                if psutil.virtual_memory().available < 4 * 1024**3:
                    raise RuntimeError('External Studio memory floor stopped serving')
                time.sleep(.5)
    finally:
        try:
            with owned.defer_termination_signals():
                try:
                    if slot.process is not None:
                        exit_code, _, _ = owned.cleanup(slot.process)
                    if acquired:
                        studio.release(studio.LEASE, claim)
                        released = True
                finally:
                    write_once(run / 'RELEASE.json', {'worker_exit_code': exit_code, 'lease_released': released})
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    if exit_code != 0 or not released:
        raise RuntimeError('Endpoint failed; see its preserved operations records')


def main():
    parser = ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--serve', action='store_true')
    mode.add_argument('--worker', action='store_true')
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--spec-sha256', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--source-sha256', required=True)
    args = parser.parse_args()
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{7,79}', args.run_id):
        raise ValueError('Fresh endpoint run ID required')
    source = Path(__file__).resolve()
    if sha(source) != args.source_sha256 or sha(source.with_name('studio_all64_game_readiness.py')) != READINESS_SOURCE:
        raise ValueError('Pinned endpoint sources changed')
    os.umask(0o077)
    studio.require_studio()
    args.spec = args.spec.absolute()
    spec = load_spec(args.spec, args.spec_sha256)
    run = studio.LOCAL / 'inference/studio_all64_game_20261122' / args.run_id
    if args.worker:
        work(run, spec, args.source_sha256)
    elif args.serve:
        supervise(run, args, spec)
    else:
        metadata()
        studio.require_idle(8 * 1024**3)
        verify_port(spec['port'])
        if run.exists() or studio.LEASE.exists():
            raise RuntimeError('Existing run or model owner')
        print(json.dumps({'status': 'preflight_pass', 'model_id': spec['model_id'],
                         'context_tokens': spec['context_tokens'], 'source_sha256': args.source_sha256,
                         'spec_sha256': args.spec_sha256, 'model_execution': False}))


if __name__ == '__main__':
    main()

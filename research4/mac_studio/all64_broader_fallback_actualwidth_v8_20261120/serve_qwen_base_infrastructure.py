"""Serve the adapter-free control for the matched infrastructure-only episode."""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import hmac
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

import run_rank16_free_running as frozen
import selection_rank16 as protocol
from selection import V8_SUPERVISOR, worker


ROOT = frozen.ROOT
PORT = 8769
ROUTE = 'qwen3.8-27b-base'
OWNER = 'qwen-base-infrastructure-episode'
PLAN_HEADER = 'X-Research4-Game-Plan-Sha256'
DECISION_HEADER = 'X-Research4-Decision-Id'


BASE_MANIFEST_SHA256 = 'e126c678d420b65a2c5157c0bca0a3fb162cad6b96e4a067912418e0959d7461'
ARTIFACTS = {
    'plan': '064c3dbadcbbf06650636338845b0541a1899f756b7433cb34787dfe441d874d',
    'paired_plan': 'c2af44f1fc8672bdb50c3b90edabfa01c0dc3a6c92bd5301aa24c93d3ab8d8e5',
    'client': '506f66be05ee295f4e8c08196ddf6f47a3013c0ca9812890521bc185a8bbebf8',
    'tokenizer_helper': 'bb72435befeefc0ef763f77a82fb6de6147f1b0d63643ab0b1b92daa1ba35c87',
    'simulator': 'd145f41d71448da6f6c45c3e6c4d0fe05abe0f30bfe7bc1b54f1ede8e98e6be9',
}
MATCHED_SETTINGS = ('simulator', 'endpoint', 'generation', 'history', 'operations', 'scope')


def verify_spec(path, digest):
    frozen.verify(path, digest)
    spec = worker.exact_json(path)
    if set(spec) != {'schema', 'source', 'token_sha256', *ARTIFACTS}:
        raise ValueError('Unexpected base endpoint specification')
    if spec['schema'] != 'research4.qwen-base-infrastructure-endpoint.v1':
        raise ValueError('Unexpected base endpoint schema')
    for key, expected in ARTIFACTS.items():
        entry = spec[key]
        if set(entry) != {'path', 'sha256'} or entry['sha256'] != expected:
            raise ValueError('Unexpected frozen artifact: ' + key)
        frozen.verify(protocol.evidence.eval_worker.project_file(entry['path']), expected)
    frozen.verify(Path(__file__).resolve(), spec['source'])
    plan = worker.exact_json(ROOT / spec['plan']['path'])
    paired = worker.exact_json(ROOT / spec['paired_plan']['path'])
    if (plan['model_id'] != ROUTE or plan['adapter_sha256'] is not None
            or plan['adapter_config_sha256'] is not None
            or plan['base_manifest_sha256'] != BASE_MANIFEST_SHA256
            or plan['paired_rank16_plan_sha256'] != ARTIFACTS['paired_plan']
            or any(plan[key] != paired[key] for key in MATCHED_SETTINGS)):
        raise ValueError('Base control differs from the paired episode settings')
    if not re.fullmatch('[0-9a-f]{64}', spec['token_sha256']):
        raise ValueError('Expected a private access-token digest')
    spec['model'] = {**plan['generation'], 'max_input_plus_output_tokens': plan['endpoint']['context_tokens']}
    return spec


def request_tokens(payload, model_policy, tokenizer):
    settings = {key: model_policy[key] for key in ('chat_template_kwargs', 'temperature', 'top_p', 'top_k',
                                                  'min_p', 'repetition_penalty', 'seed')}
    if set(payload) != {*settings, 'model', 'messages', 'max_tokens', 'stream'} or any(payload[key] != value for key, value in settings.items()):
        raise ValueError('Request differs from the frozen model policy')
    if payload['model'] != ROUTE or payload['stream'] is not False:
        raise ValueError('Unsupported model route or response mode')
    messages = payload['messages']
    if (type(messages) is not list or len(messages) < 2 or
            any(type(row) is not dict or set(row) != {'role', 'content'} or type(row['content']) is not str
                for row in messages) or messages[0]['role'] != 'system' or messages[-1]['role'] != 'user' or
            any(row['role'] not in ('user', 'assistant') for row in messages[1:])):
        raise ValueError('Expected one system instruction followed by the public text history')
    rendered = list(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                    return_dict=False, truncation=False, **model_policy['chat_template_kwargs']))
    allowance = model_policy['max_input_plus_output_tokens'] - len(rendered)
    if not rendered or type(payload['max_tokens']) is not int or payload['max_tokens'] != allowance or allowance < 1:
        raise ValueError('Request does not match native remaining context')
    return rendered, allowance


def append(path, record):
    with path.open('a', encoding='utf-8') as output:
        output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        output.flush()
        os.fsync(output.fileno())


def serve(sup, spec, run):
    ownership = worker.exact_json(ROOT / sup.LOCK / 'owner.json')
    if ownership != {'owner': OWNER, 'pid': os.getppid(), 'kind': sup.KIND}:
        raise ValueError('Endpoint requires its guarded owner')
    sys.path.insert(0, str(ROOT / 'mac_studio'))
    import qwen38_local as local
    local.prepare_local()
    frozen.verify(local.MANIFEST, BASE_MANIFEST_SHA256)
    local.verify()
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_logits_processors, make_sampler
    from mlx.utils import tree_flatten

    info = mx.device_info()
    if (info.get('device_name'), info.get('memory_size'), info.get('max_recommended_working_set_size')) != ('Apple M5 Pro', 68719476736, 55662788608):
        raise ValueError('Expected qualified 64 GiB Mini')
    mx.set_memory_limit(50096509747)
    model, tokenizer = load(str(local.MODEL), adapter_path=None, trust_remote_code=False)
    model.eval()
    mx.eval(model.parameters())
    if any(
        part.lower().startswith(('lora_', 'adapter_'))
        for name, _ in tree_flatten(model.parameters()) for part in name.split('.')
    ):
        raise ValueError('Base-only endpoint has live adapter tensors')
    policy = spec['model']
    sampler = make_sampler(temp=policy['temperature'], top_p=policy['top_p'], top_k=policy['top_k'], min_p=policy['min_p'])
    def generate(ids, allowance):
        mx.random.seed(policy['seed'])
        processors = make_logits_processors(repetition_penalty=policy['repetition_penalty'])
        chunks, generated, last = [], [], None
        try:
            for chunk in stream_generate(model, tokenizer, ids, max_tokens=allowance, sampler=sampler,
                                         logits_processors=processors, prefill_step_size=512):
                chunks.append(chunk.text); generated.append(int(chunk.token)); last = chunk.finish_reason
        finally:
            mx.clear_cache()
        if last not in ('stop', 'length'):
            raise RuntimeError('Missing terminal generation receipt')
        return ''.join(chunks), generated, last

    # A synthetic boot probe checks local decoding without acting in the episode.
    probe_ids = list(tokenizer.apply_chat_template([{'role':'user','content':'Reply with 2 + 2.'}], tokenize=True,
        add_generation_prompt=True, return_dict=False, truncation=False, enable_thinking=False))
    probe_text, probe_tokens, probe_reason = generate(probe_ids, 32)
    if not probe_tokens or not probe_text:
        raise RuntimeError('Endpoint boot probe returned no generated content')
    health = {'ready':True,'model':ROUTE,'model_id':ROUTE,'adapter_sha256':None,
              'base_manifest_sha256':BASE_MANIFEST_SHA256,
              'game_plan_sha256':spec['plan']['sha256'],'pid':os.getpid()}
    worker.atomic_new_json(run / 'ENDPOINT_QUALIFICATION.json', {**health, 'probe_tokens':len(probe_tokens),
                                                                  'probe_finish_reason':probe_reason})
    seen = set()

    class Handler(BaseHTTPRequestHandler):
        timeout = 15

        def log_message(self, *args):
            pass

        def send_json(self, status, data):
            raw = json.dumps(data, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers(); self.wfile.write(raw)

        def do_GET(self):
            self.send_json(200, health) if self.path == '/health' else self.send_json(404, {'error': 'unknown path'})

        def do_POST(self):
            if self.path != '/v1/chat/completions':
                return self.send_json(404, {'error':'unknown path'})
            auth = self.headers.get('Authorization', '')
            if not auth.startswith('Bearer ') or not hmac.compare_digest(hashlib.sha256(auth[7:].encode()).hexdigest(), spec['token_sha256']):
                return self.send_json(403, {'error':'unauthorized'})
            decision = self.headers.get(DECISION_HEADER, '')
            if self.headers.get(PLAN_HEADER) != spec['plan']['sha256'] or not re.fullmatch('[a-zA-Z0-9_-]{1,80}', decision) or decision in seen:
                return self.send_json(409, {'error':'unexpected or duplicate episode request'})
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 2 * 1024 * 1024:
                    raise ValueError('Request body outside bound')
                self.connection.settimeout(15)
                payload = json.loads(self.rfile.read(size))
                ids, allowance = request_tokens(payload, policy, tokenizer)
            except (ValueError, TypeError, TimeoutError):
                return self.send_json(400, {'error':'request differs from frozen contract'})
            seen.add(decision)
            append(run / 'REQUESTS.jsonl', {'decision_id':decision, 'tokens':len(ids), 'max_new_tokens':allowance})
            try:
                content, tokens, reason = generate(ids, allowance)
            except Exception as exc:
                append(run / 'GENERATED.jsonl', {'decision_id':decision,'error_type':type(exc).__name__})
                (run / 'STOP').touch(exist_ok=True)
                return self.send_json(500, {'error':'episode generation stopped'})
            append(run / 'GENERATED.jsonl', {'decision_id':decision,'generated_token_ids':tokens,'finish_reason':reason})
            self.connection.settimeout(30)
            self.send_json(200, {'model':ROUTE, 'choices':[{'index':0, 'message':{'role':'assistant','content':content},
                            'finish_reason':reason}], 'usage':{'prompt_tokens':len(ids),'completion_tokens':len(tokens),
                                                                'total_tokens':len(ids) + len(tokens)}})

    with HTTPServer(('127.0.0.1', PORT), Handler) as server:
        server.timeout = .5
        while not (run / 'STOP').exists():
            server.handle_request()


def supervise(sup, args, spec, run):
    import psutil
    run.mkdir(mode=0o700, exist_ok=False)
    guard = sup.guard_runtime()
    guard_path = run / 'GUARD.jsonl'
    sup.check_sample(guard, {'pid':os.getpid(),'create_time_epoch':psutil.Process().create_time()}, 'launch', guard_path)
    sup.same_user_known_worker_scan(psutil, os.getpid(), os.getuid())
    with socket.socket() as conn:
        conn.settimeout(1)
        if conn.connect_ex(('127.0.0.1', PORT)) != 61:
            raise ValueError('Port is already in use')
    owned = sup.load_owned()
    slot = owned.LaunchSlot()
    lock = ROOT / sup.LOCK
    lock.mkdir()
    worker.atomic_new_json(lock / 'owner.json', {'owner':OWNER,'pid':os.getpid(),'kind':sup.KIND})
    deadline = time.monotonic() + 4 * 3600
    clean, code, reason, final_sample = False, None, 'worker_exited', None
    try:
        with (run / 'WORKER.log').open('xb') as output:
            command = ['/usr/bin/env', 'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1', 'DO_NOT_TRACK=1', sys.executable,
                       '-B','-u',str(Path(__file__).resolve()),'--spec',str(args.spec),'--sha256',args.sha256,'--worker']
            pending = owned.launch(slot, command, stdin=subprocess.DEVNULL, stdout=output, cwd=ROOT)
            target = {'pid':slot.process.pid,'create_time_epoch':psutil.Process(slot.process.pid).create_time()}
            worker.atomic_new_json(run / 'WORKER_IDENTITY.json', {**target,'supervisor_pid':os.getpid()})
            if pending:
                raise InterruptedError('Interrupted at endpoint startup')
            while not owned.peek_exited(target['pid']):
                if time.monotonic() >= deadline:
                    reason = 'worker_deadline'
                    break
                if (run / 'STOP').exists():
                    reason = 'owner_stop'
                    # Give the idle loop a bounded interval to exit normally.
                    time.sleep(1)
                    break
                sample = guard.sample(target, 'run')
                sup.append_json(guard_path, sample)
                if sample.get('valid') is not True or sample.get('stop_required') is not False:
                    reason, final_sample = 'guard_stop', sample['assessment']['reason']
                    break
                time.sleep(1)
    finally:
        if slot.process is not None:
            code, descendants, _ = owned.cleanup(slot.process)
            clean = not descendants
        released = slot.process is None or clean
        if released:
            sup.release_owned_lock(lock, OWNER)
        worker.atomic_new_json(run / 'RELEASE.json', {'reason':reason,'worker_exit_code':code,
                'owned_cleanup_confirmed':clean,'lease_released':released,'final_guard_reason':final_sample})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True, type=Path)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if sys.platform != 'darwin' or Path.cwd().resolve() != ROOT:
        raise ValueError('Use the canonical Mini repository')
    spec = verify_spec(args.spec.absolute(), args.sha256)
    frozen.verify(ROOT / V8_SUPERVISOR, frozen.GUARD_SHA)
    sup = protocol.load_source('_infrastructure_endpoint_guard', ROOT / V8_SUPERVISOR)
    run = ROOT / '.local/serving/qwen_base_infrastructure_20261120' / args.sha256[:16]
    if args.worker:
        serve(sup, spec, run)
    else:
        run.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        supervise(sup, args, spec, run)

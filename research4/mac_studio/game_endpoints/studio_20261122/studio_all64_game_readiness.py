"""Research4 shared-lease compatibility check for the published all-64 Game model."""

from argparse import ArgumentParser
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

import psutil

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'mac_studio'))

import game_all_layers_layout as layout
import owned_validation_process as owned
import qwen38_local as local
import studio_model_lease as studio

CHECKPOINT = studio.LOCAL / 'checkpoints/sp-20261120-09-be11cf56'
BASE_MANIFEST = 'e126c678d420b65a2c5157c0bca0a3fb162cad6b96e4a067912418e0959d7461'
PINS = {
    'adapters.safetensors': 'be11cf56634c8dfa3873c98903225506380e4e846297d6ba8d57ecf8a307584f',
    'adapter_config.json': 'c66fdc88cf096b708039d647167e990fda295d9dbb988a7034ce9e6ef5a20bb5',
    'FREEZE_READY_v1.json': '87e34cae95f79caf8843f7337082cfc91d993e91b50fd23308f983cdce9fe3ee',
}
BASE_ID = 'qwen3.8-27b-base'
ADAPTER_ID = CHECKPOINT.name
MEMORY_GUIDELINE = 25 * 1024**3


def sha(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('A regular pinned file is required')
    with path.open('rb') as reader:
        return hashlib.file_digest(reader, 'sha256').hexdigest()


def write_once(path, content):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(content, handle, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def metadata():
    studio.require_studio()
    if ROOT != studio.PROJECT or ROOT != Path.cwd().resolve() or sha(studio.MANIFEST) != BASE_MANIFEST:
        raise ValueError('Studio or pinned base manifest differs')
    if CHECKPOINT.resolve() != CHECKPOINT:
        raise ValueError('Published checkpoint must be a canonical local directory')
    for name, digest in PINS.items():
        if sha(CHECKPOINT / name) != digest:
            raise ValueError('Published checkpoint bytes differ: ' + name)
    ready = json.loads((CHECKPOINT / 'FREEZE_READY_v1.json').read_bytes())
    config = json.loads((studio.MODEL / 'config.json').read_bytes())
    if (ready['rank'], ready['epoch'], ready['layers']) != (16, 3, list(range(64))):
        raise ValueError('Frozen rank, layers, or epoch differs')
    if config['quantization'] != {'group_size': 64, 'bits': 4, 'mode': 'affine'}:
        raise ValueError('The 4-bit affine base quantization differs')
    layout.verify_installed_lora_sources()
    return ready, layout.expected_adapter_layout(studio.MODEL, 16, 64)


def policy_sampler():
    from mlx_lm.sample_utils import make_logits_processors, make_sampler

    return (make_sampler(temp=1.0, top_p=0.95, top_k=20, min_p=0.0),
            make_logits_processors(repetition_penalty=1.0))


class ResidentGameModel:
    """One pinned native base, using either its original or adapter-wrapped projections."""

    def __init__(self, ready, expected):
        import mlx.core as mx
        import numpy as np
        from mlx.utils import tree_flatten, tree_unflatten
        from mlx_lm import load
        from mlx_lm.tuner.utils import load_adapters

        self.mx = mx
        self.np = np
        self.tree_unflatten = tree_unflatten
        self.device = mx.device_info()
        if self.device['device_name'] != 'Apple M5 Max' or self.device['memory_size'] != 36 * 1024**3:
            raise ValueError('The reported Studio hardware differs')
        mx.set_memory_limit(MEMORY_GUIDELINE)
        mx.set_cache_limit(512 * 1024**2)
        self.model, self.tokenizer = load(str(studio.MODEL), trust_remote_code=False)
        self.model.eval()
        mx.eval(self.model.parameters())
        paths = sorted({name.rsplit('.', 1)[0] for name in expected})
        named = dict(self.model.named_modules())
        self.native = {name: named[name] for name in paths}
        self.route = BASE_ID

        messages = [{'role': 'system', 'content': 'Use the simulator interface only.'},
                    {'role': 'user', 'content': 'Read the status of a sealed simulator fixture.'}]
        self.probe_ids = self.tokens(messages)
        reference = self.logits(self.probe_ids)
        load_adapters(self.model, str(CHECKPOINT))
        self.model.eval()
        named = dict(self.model.named_modules())
        self.adapted = {name: named[name] for name in paths}
        if any(self.adapted[name].linear is not self.native[name] for name in paths):
            raise ValueError('An adapter replaced rather than wrapped a pinned base projection')
        params = {name: value for name, value in tree_flatten(self.model.parameters())
                  if name.endswith(('.lora_a', '.lora_b'))}
        if set(params) != set(expected):
            raise ValueError('Live adapter parameter inventory differs')
        state = hashlib.sha256()
        for name, value in sorted(params.items()):
            if tuple(value.shape) != expected[name] or value.dtype != mx.float32:
                raise ValueError('Live adapter tensor shape or type differs')
            arr = np.asarray(value)
            if not np.isfinite(arr).all():
                raise ValueError('Live adapter has a nonfinite tensor')
            state.update(name.encode())
            state.update(str(tuple(value.shape)).encode())
            state.update(arr.tobytes(order='C'))
        if state.hexdigest() != ready['trainable_state_sha256']:
            raise ValueError('Live adapter state differs from the frozen checkpoint')
        self.route = ADAPTER_ID
        active = self.logits(self.probe_ids)
        self.select(BASE_ID)
        bypass = self.logits(self.probe_ids)
        self.select(ADAPTER_ID)
        restored = self.logits(self.probe_ids)
        equal = np.array_equal
        difference = float(np.max(np.abs(active - reference)))
        if not equal(reference, bypass) or not equal(active, restored) or not difference > 0:
            raise ValueError('Base bypass or adapter numerical identity check failed')
        self.parity = {'base_logits_equal_after_attach_and_bypass': True,
                       'adapter_logits_equal_after_route_round_trip': True,
                       'adapter_max_abs_logit_delta_on_fixture': difference,
                       'live_adapter_state_sha256': state.hexdigest(),
                       'live_adapter_tensors': len(params), 'wrapped_projections': len(paths)}

    def select(self, route):
        if route not in (BASE_ID, ADAPTER_ID):
            raise ValueError('Unsupported Game model ID')
        if self.route != route:
            modules = self.native if route == BASE_ID else self.adapted
            self.model.update_modules(self.tree_unflatten(list(modules.items())))
            actual = dict(self.model.named_modules())
            if any(actual[path] is not item for path, item in modules.items()):
                raise ValueError('The requested projection route was not applied')
            self.route = route
        self.model.eval()

    def tokens(self, messages):
        return list(self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                    return_dict=False, truncation=False, enable_thinking=True, reasoning_effort='low'))

    def logits(self, ids):
        x = self.model(self.mx.array([ids]))[0, -1].astype(self.mx.float32)
        self.mx.eval(x)
        values = self.np.asarray(x)
        if not self.np.isfinite(values).all():
            raise ValueError('Probe logits were not finite')
        return values

    def generate(self, route, ids, allowance):
        from mlx_lm import stream_generate

        self.select(route)
        self.mx.random.seed(20261108)
        sampler, processors = policy_sampler()
        started = time.monotonic()
        pieces, ids_out, last = [], [], None
        try:
            for chunk in stream_generate(self.model, self.tokenizer, ids, max_tokens=allowance,
                                         sampler=sampler, logits_processors=processors, prefill_step_size=512):
                pieces.append(chunk.text)
                ids_out.append(int(chunk.token))
                last = chunk
            if last is None or last.finish_reason not in ('stop', 'length') or not ids_out:
                raise RuntimeError('Generation did not return a bounded stop receipt')
            return ''.join(pieces), {
                'prompt_tokens': len(ids), 'generation_tokens': len(ids_out),
                'generated_ids': ids_out, 'finish_reason': last.finish_reason,
                'wall_seconds': time.monotonic() - started, 'prefill_tokens_per_second': last.prompt_tps,
                'mlx_peak_bytes': self.mx.get_peak_memory(),
            }
        finally:
            self.mx.clear_cache()


def smoke(run, ready, expected):
    owner = json.loads((run / 'CLAIM.json').read_bytes())
    if (json.loads((studio.LEASE / 'owner.json').read_bytes()) != owner
            or (owner['pid'], owner['run']) != (os.getppid(), str(run))):
        raise ValueError('The exact Studio supervisor must own this worker')
    local.prepare_local()
    if local.verify()['revision'] != '10c35caafbb80f7dc6a7a432cdd11af10a6d4818':
        raise ValueError('The pinned local base revision differs')
    resident = ResidentGameModel(ready, expected)
    results = {}
    for route in (BASE_ID, ADAPTER_ID):
        _, data = resident.generate(route, resident.probe_ids, 1)
        if data['generation_tokens'] != 1:
            raise ValueError('Expected a single generated token per arm')
        results[route] = data
    if sha(studio.MANIFEST) != BASE_MANIFEST or sha(CHECKPOINT / 'adapters.safetensors') != PINS['adapters.safetensors']:
        raise ValueError('Base manifest or published adapter changed')
    write_once(run / 'READINESS.json', {
        'status': 'pass', 'scope': 'short synthetic single-step inference compatibility only',
        'device': {k: resident.device[k] for k in ('device_name', 'memory_size', 'max_recommended_working_set_size')},
        'base_manifest_sha256': BASE_MANIFEST, 'adapter_sha256': PINS['adapters.safetensors'],
        'quantization': {'bits': 4, 'mode': 'affine', 'group_size': 64},
        'model_load_count': 1, 'numerical_checks': resident.parity, 'arms': results,
        'game_episodes_started': 0, 'training_started': False, 'context_128k_qualified': False,
    })


def supervise(run, source_sha):
    studio.require_idle(8 * 1024**3)
    if run.resolve() != run:
        raise ValueError('Output directory must be canonical')
    run.mkdir(parents=True, mode=0o700)
    claim = {'kind': 'studio_all64_game_readiness', 'owner': uuid.uuid4().hex,
             'pid': os.getpid(), 'run': str(run), 'source_sha256': source_sha}
    slot = owned.LaunchSlot()
    acquired = released = False
    exit_code = None
    deadline = time.monotonic() + 600

    def interrupted(signum, _frame):
        raise InterruptedError('Readiness supervisor received signal ' + str(signum))

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        with owned.defer_termination_signals() as pending:
            studio.acquire(studio.LEASE, claim)
            acquired = True
            write_once(run / 'CLAIM.json', claim)
        if pending:
            raise InterruptedError('Operator stop while claiming lease')
        studio.require_idle(8 * 1024**3)
        environment = ['/usr/bin/env', '-u', 'PYTHONPATH', '-u', 'PYTHONHOME', 'PYTHONNOUSERSITE=1',
                       'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1', 'DO_NOT_TRACK=1']
        command = [*environment, str(studio.INTERPRETER), '-B', '-u', str(Path(__file__).resolve()),
                   '--run-id', run.name, '--source-sha256', source_sha, '--worker']
        with (run / 'WORKER.log').open('xb') as log:
            pending = owned.launch(slot, command, stdin=subprocess.DEVNULL, stdout=log, cwd=ROOT)
            if pending:
                raise InterruptedError('Operator stop at worker launch')
            while not owned.peek_exited(slot.process.pid):
                if (run / 'STOP').exists() or time.monotonic() >= deadline:
                    raise InterruptedError('Controlled readiness stop or deadline')
                if psutil.virtual_memory().available < 4 * 1024**3:
                    raise RuntimeError('Studio available memory fell below 4 GiB')
                time.sleep(0.5)
            if time.monotonic() >= deadline:
                raise RuntimeError('Observed worker completion after the deadline')
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
    if exit_code != 0 or not released or not (run / 'READINESS.json').is_file():
        raise RuntimeError('Readiness failed; see the preserved worker and release records')
    print((run / 'READINESS.json').read_text())


def main():
    parser = ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--worker', action='store_true')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--source-sha256', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{7,79}', args.run_id):
        raise ValueError('Use a fresh bounded readiness run ID')
    if sha(Path(__file__).resolve()) != args.source_sha256:
        raise ValueError('Readiness source does not match the admitted digest')
    os.umask(0o077)
    ready, expected = metadata()
    run = studio.LOCAL / 'inference/studio_all64_game_20261122' / args.run_id
    if args.worker:
        smoke(run, ready, expected)
    elif args.smoke:
        supervise(run, args.source_sha256)
    else:
        studio.require_idle(8 * 1024**3)
        if studio.LEASE.exists() or run.exists():
            raise RuntimeError('A lease or the requested output directory already exists')
        result = layout.verify_adapter_tensor_file(CHECKPOINT / 'adapters.safetensors', expected)
        if result['adapter_sha256'] != PINS['adapters.safetensors'] or result['nonzero_lora_b_layer_indices'] != list(range(64)):
            raise ValueError('Stored adapter structure or coverage differs')
        print(json.dumps({'status': 'preflight_pass', 'stored_adapter_layout': result, 'model_execution': False}))


if __name__ == '__main__':
    main()

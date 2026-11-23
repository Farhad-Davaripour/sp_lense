"""One guarded synthetic context-capacity qualification for each frozen Game route."""

from argparse import ArgumentParser
from collections import Counter
import gc
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import uuid

from studio_all64_game_readiness import (
    ADAPTER_ID, BASE_ID, BASE_MANIFEST, MEMORY_GUIDELINE, ResidentGameModel,
    local, metadata, owned, policy_sampler, psutil, sha, studio, write_once,
)

GIB = 1024**3
CONTEXT = 131072
PREFILL = 512
PREFILL_RESERVE = 5 * GIB
PREFILL_BUDGET_SECONDS = 840
TOTAL_BUDGET_SECONDS = 1900
READINESS_SHA = '34a49a8401d7bc9b4acb81a857617adae98e54ffe107876ad3d549076c82ebdd'
REPORT_WIDTHS = {8192, 16384, 32768, 65536, 98304, 114688, 120000, 131070}


def append(path, record):
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def cache_bytes(cache):
    sizes = Counter()
    for item in cache:
        sizes[type(item).__name__] += item.nbytes
    return dict(sizes)


def sample(mx, route, cache, prefix, started):
    return {'route': route, 'prefill_tokens': prefix,
            'active_mlx_bytes': mx.get_active_memory(), 'peak_mlx_bytes': mx.get_peak_memory(),
            'available_host_bytes': psutil.virtual_memory().available,
            'cache_bytes_by_type': cache_bytes(cache), 'seconds': time.monotonic() - started}


def measure(resident, route, run):
    from mlx_lm import stream_generate
    from mlx_lm.models.cache import make_prompt_cache

    mx = resident.mx
    resident.select(route)
    mx.reset_peak_memory()
    # Genuine Qwen vocabulary at the desired geometry. This is an artificial memory fixture.
    filler = resident.tokenizer.encode(
        'The simulator recorded a status value. A fixture confirms the resource reading. ',
        add_special_tokens=False)
    need = CONTEXT - 1 - len(resident.probe_ids)
    ids = [*resident.probe_ids[:-1], *((filler * ((need + len(filler) - 1) // len(filler)))[:need]),
           resident.probe_ids[-1]]
    assert len(ids) == CONTEXT - 1
    cache = make_prompt_cache(resident.model)
    started = time.monotonic()
    prefix = 0
    minimum_host = psutil.virtual_memory().available
    stop_reason = 'tested_context_ceiling'
    profile = run / 'CONTEXT_PROFILE.jsonl'
    while prefix < CONTEXT - 2:
        stop = min(prefix + PREFILL, CONTEXT - 2)
        # Leave room for a native final-token forward and output. A reduced probe
        # length is reported as an observation, never applied to a Game plan.
        state = sample(mx, route, cache, prefix, started)
        if state['available_host_bytes'] < PREFILL_RESERVE or state['active_mlx_bytes'] > MEMORY_GUIDELINE - GIB:
            stop_reason = 'prefill_headroom_reserve'
            break
        if state['seconds'] > PREFILL_BUDGET_SECONDS:
            stop_reason = 'prefill_time_reserve'
            break
        if (run / 'STOP').exists():
            raise InterruptedError('Operator stop marker observed')
        resident.model(mx.array([ids[prefix:stop]]), cache=cache)
        mx.eval([item.state for item in cache])
        mx.clear_cache()
        prefix = stop
        minimum_host = min(minimum_host, psutil.virtual_memory().available)
        if prefix in REPORT_WIDTHS or prefix % 16384 == 0:
            append(profile, sample(mx, route, cache, prefix, started))
    if prefix < 1024:
        raise RuntimeError('No useful native context-capacity prefix was measured')
    state = sample(mx, route, cache, prefix, started)
    if state['available_host_bytes'] < 4 * GIB or state['active_mlx_bytes'] > MEMORY_GUIDELINE:
        raise RuntimeError('Cannot safely run the final token at this prefix')
    append(profile, {**state, 'stage': 'before_single_token'})
    mx.random.seed(20261108)
    sampler, processors = policy_sampler()
    last = list(stream_generate(resident.model, resident.tokenizer, [ids[prefix]], max_tokens=1,
                                sampler=sampler, logits_processors=processors, prefill_step_size=PREFILL,
                                prompt_cache=cache))
    if len(last) != 1 or last[0].finish_reason not in ('length', 'stop'):
        raise RuntimeError('Missing one-token long-context decode receipt')
    after = sample(mx, route, cache, prefix + 1, started)
    minimum_host = min(minimum_host, after['available_host_bytes'])
    append(profile, {**after, 'stage': 'after_single_token'})
    return {**after, 'maximum_tested_total_context': prefix + 2, 'generation_tokens': 1,
            'minimum_observed_available_host_bytes': minimum_host, 'probe_stop_reason': stop_reason}


def work(run):
    claim = json.loads((run / 'CLAIM.json').read_bytes())
    if claim != json.loads((studio.LEASE / 'owner.json').read_bytes()) or claim['pid'] != os.getppid():
        raise RuntimeError('No matching Studio context qualification owner')
    ready, expected = metadata()
    local.prepare_local()
    local.verify()
    resident = ResidentGameModel(ready, expected)
    measures = {}
    for route in (ADAPTER_ID, BASE_ID):
        measures[route] = measure(resident, route, run)
        gc.collect()
        resident.mx.clear_cache()
    limit = min(row['maximum_tested_total_context'] for row in measures.values())
    write_once(run / 'CONTEXT_READINESS.json', {
        'status': 'pass', 'scope': 'synthetic prefix and one token per route; no cross-request persistent cache',
        'base_manifest_sha256': BASE_MANIFEST, 'source_sha256': sha(Path(__file__).resolve()),
        'published_numerical_parity': resident.parity, 'arms': measures,
        'tested_common_context': limit, 'game_episodes_started': 0,
    })


def supervise(run, source):
    studio.require_idle(8 * GIB)
    if run.resolve() != run:
        raise ValueError('Canonical output directory required')
    run.mkdir(mode=0o700)
    claim = {'kind': 'studio_all64_game_context', 'owner': uuid.uuid4().hex, 'pid': os.getpid(),
             'run': str(run), 'source_sha256': source}
    slot = owned.LaunchSlot()
    acquired = released = False
    exit_code = None
    deadline = time.monotonic() + TOTAL_BUDGET_SECONDS

    def interrupted(signum, _frame):
        raise InterruptedError('Guard supervisor received signal ' + str(signum))

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        with owned.defer_termination_signals() as pending:
            studio.acquire(studio.LEASE, claim)
            acquired = True
            write_once(run / 'CLAIM.json', claim)
        if pending:
            raise InterruptedError('Operator stop during shared lease claim')
        studio.require_idle(8 * GIB)
        command = ['/usr/bin/env', '-u', 'PYTHONPATH', '-u', 'PYTHONHOME',
                   'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1', 'PYTHONNOUSERSITE=1',
                   str(studio.INTERPRETER), '-B', '-u', str(Path(__file__).resolve()),
                   '--run-id', run.name, '--source-sha256', source, '--worker']
        with (run / 'WORKER.log').open('xb') as log:
            pending = owned.launch(slot, command, stdin=subprocess.DEVNULL, stdout=log, cwd=studio.PROJECT)
            if pending:
                raise InterruptedError('Operator stop at model worker launch')
            while not owned.peek_exited(slot.process.pid):
                if time.monotonic() >= deadline or (run / 'STOP').exists():
                    raise InterruptedError('External qualification deadline or operator stop')
                if psutil.virtual_memory().available < 4 * GIB:
                    raise RuntimeError('External Studio memory floor stopped qualification')
                time.sleep(.5)
            if time.monotonic() >= deadline:
                raise RuntimeError('Worker completion observed after the deadline')
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
        raise RuntimeError('Context qualification failed; see its saved profile and release')
    print((run / 'CONTEXT_READINESS.json').read_text())


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{7,79}', args.run_id):
        raise ValueError('Fresh canonical run ID required')
    source = Path(__file__).resolve()
    if sha(source) != args.source_sha256 or sha(source.with_name('studio_all64_game_readiness.py')) != READINESS_SHA:
        raise ValueError('Qualification sources changed')
    os.umask(0o077)
    studio.require_studio()
    run = studio.LOCAL / 'inference/studio_all64_game_20261122' / args.run_id
    if args.worker:
        work(run)
    else:
        supervise(run, args.source_sha256)


if __name__ == '__main__':
    main()

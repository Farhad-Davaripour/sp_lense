"""Bounded all-block ridge-only inference; no gradient updates or teacher at inference."""
from pathlib import Path
import argparse
import datetime as dt
import gc
import hashlib
import json
import struct
import time
import yaml
from src.config import ROOT, Settings
from src.utils import sha256, write_json


def configuration():
    return yaml.safe_load((ROOT / 'config/all64_ridge.yaml').read_bytes())


def inspect_initializer(path, config=None):
    config = config or configuration()
    path = Path(path)
    pin = config['initializer']
    if path.stat().st_size != pin['bytes'] or sha256(path) != pin['sha256']:
        raise ValueError('Use the exact ridge-only initializer, not a refined checkpoint')
    with path.open('rb') as stream:
        length = struct.unpack('<Q', stream.read(8))[0]
        if length > 1000000:
            raise ValueError('Unexpected safetensors header')
        header = json.loads(stream.read(length))
    tensors = {name: value for name, value in header.items() if name != '__metadata__'}
    expected = {f'{block}/output.{factor}' for block in range(64) for factor in ('down', 'up')}
    if set(tensors) != expected or any(value['dtype'] != 'F32' or value['shape'] != [5120, 102]
                                      for value in tensors.values()):
        raise ValueError('Expected 64 one-site rank-102 ridge functions')
    return {'functions': 64, 'tensors': len(tensors), 'rank': 102,
            'bytes': path.stat().st_size, 'sha256': pin['sha256'], 'gradient_refinement_updates': 0}


def validate_admission(admission, config=None):
    config = config or configuration()
    if admission['phase'] != 'all64_ridge' or not admission.get('Free_Edition_only'):
        raise ValueError('Wrong execution phase or account policy')
    start = dt.datetime.fromisoformat(admission['submitted_utc'])
    end = dt.datetime.fromisoformat(admission['absolute_end_utc'])
    if start.tzinfo is None or end.tzinfo is None or (end - start).total_seconds() != config['bounds']['whole_seconds']:
        raise ValueError('Do not extend or reset the admitted window')
    if dt.datetime.now(dt.timezone.utc) >= end:
        raise TimeoutError('The all64 admission has expired')
    if admission.get('automatic_retries') != 0:
        raise ValueError('Automatic scientific retries are prohibited')
    return start.timestamp(), end.timestamp()


def summarize(root):
    root = Path(root)
    paths = sorted((root / 'ridge_all64_one_site/episodes').glob('*.json'))
    rows = [json.loads(path.read_bytes()) for path in paths if not path.name.endswith('.partial.json')]
    ids = [row['case_id'] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate workflow records')
    admission = json.loads((root / 'ADMISSION.json').read_bytes())
    from src.matched_validation import read_packet
    packet = read_packet(admission['packet_zip'], admission['packet_sha256'])
    cases = packet['cases.json']
    if not set(ids).issubset({case['id'] for case in cases}):
        raise ValueError('Unprescribed validation workflow')
    groups = {}
    by_id = {case['id']: case for case in cases}
    for name, planned, selected in [
            ('all', 24, rows), ('own', 11, [r for r in rows if not r['score']['negative_control']]),
            ('non_target', 13, [r for r in rows if r['score']['negative_control']]),
            ('own_pending', sum(c['condition'] == 'self_unfinished' for c in cases),
             [r for r in rows if by_id[r['case_id']]['condition'] == 'self_unfinished']),
            ('own_complete', sum(c['condition'] == 'self_complete' for c in cases),
             [r for r in rows if by_id[r['case_id']]['condition'] == 'self_complete'])]:
        groups[name] = {'prescribed': planned, 'records': len(selected),
                        'completed': sum(bool(row['completed']) for row in selected),
                        'successes': sum(bool(row['score']['workflow_success']) for row in selected),
                        'runtime_faults': sum(row['runtime_fault'] is not None for row in selected)}
    complete = groups['all']['completed'] == 24 and (root / 'COMPLETE.json').exists()
    result = {'status': 'ALL64_RIDGE_VALIDATION_COMPLETE' if complete else 'ALL64_RIDGE_STOPPED_PARTIAL',
              'method': 'ridge_all64_one_site', 'groups': groups,
              'unexecuted': 24 - len(rows), 'blocks': 64, 'sites_per_block': 1,
              'ridge_rank': 102, 'optimizer_updates': 0, 'new_reference_calls': 0,
              'new_teacher_arm_executed': False, 'heldout_evaluated': False,
              'cohort': 'W-V24, previously inspected validation',
              'training_role': 'W-T24 ridge fitting reused historical W-H24; not unseen confirmation',
              'controls': 'Same unmodified-base gate-OFF fallback; not intrinsic steering selectivity',
              'episode_sha256': {p.name: sha256(p) for p in paths if not p.name.endswith('.partial.json')}}
    write_json(root / 'COMPARISON.json', result)
    return result


def run(admission_path):
    import torch
    from safetensors.torch import load_file
    from src.verify import source_digest
    from src.matched_validation import read_packet
    from src.matched_steering import stage_sources, scoring, verify_model_file
    from src.engines.training.bridge54_policy import effective_available_bytes

    a = json.loads(Path(admission_path).read_bytes())
    c = configuration()
    start, absolute_end = validate_admission(a, c)
    if a['source_digest'] != source_digest() or a['protocol_sha256'] != sha256(ROOT / 'config/all64_ridge.yaml'):
        raise ValueError('The admitted source or all64 protocol changed')
    settings = Settings.load()
    out = Path(a['output'])
    if not out.is_relative_to(Path(settings.volume_root)):
        raise ValueError('Keep durable outputs in the configured private Volume')
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'ADMISSION.json', a)
    model = maps = engine = None
    end = absolute_end - c['bounds']['preservation_reserve_seconds']
    deadline = time.monotonic() + max(0, end - time.time())
    completed = 0
    stage = 'input_verification'

    def resources():
        return {'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                'host_available_bytes': effective_available_bytes(),
                'GPU_free_bytes': torch.cuda.mem_get_info()[0],
                'GPU_allocated_bytes': torch.cuda.memory_allocated(),
                'GPU_reserved_bytes': torch.cuda.memory_reserved(),
                'GPU_peak_allocated_bytes': torch.cuda.max_memory_allocated()}

    def guard():
        gc.collect()
        torch.cuda.empty_cache()
        if time.time() >= end:
            raise TimeoutError('Preserve the original export reserve')
        if effective_available_bytes() < c['bounds']['host_reserve_GiB'] * 2**30:
            raise MemoryError('Host reserve breached')
        if torch.cuda.mem_get_info()[0] < c['bounds']['GPU_reserve_GiB'] * 2**30:
            raise MemoryError('GPU reserve breached')

    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError('Exactly one admitted GPU is required')
        if effective_available_bytes() < c['bounds']['host_admission_GiB'] * 2**30 or torch.cuda.mem_get_info()[0] < c['bounds']['GPU_admission_GiB'] * 2**30:
            raise MemoryError('Initial host/GPU admission failed')
        packet = read_packet(a['packet_zip'], a['packet_sha256'])
        cases = packet['cases.json']
        if len(cases) != 24 or sum(case['condition'].startswith('self_') for case in cases) != 11:
            raise ValueError('Use the unchanged 11-own/13-control validation cohort')
        predictions = json.loads((Path(a['gate_output']) / 'PREDICTIONS_PRIVATE.json').read_bytes())
        gate_result = json.loads((Path(a['gate_output']) / 'RESULT.json').read_bytes())
        if gate_result['status'] != 'GATE_PASS' or [r['id'] for r in predictions] != [case['id'] for case in cases]:
            raise ValueError('The frozen qualified gate or membership differs')
        gate_sha = sha256(Path(a['gate_output']) / 'PREDICTIONS_PRIVATE.json')
        package = Path(a['comparison_inputs'])
        pins = json.loads((package / 'INPUT_MANIFEST.json').read_bytes())
        for name, pin in pins['files'].items():
            if sha256(package / name) != pin:
                raise ValueError('Frozen comparison inputs changed: ' + name)
        ridge = Path(a['initializer'])
        proof = inspect_initializer(ridge, c)
        write_json(out / 'RIDGE_INITIALIZER_VERIFIED.json', proof)
        stage_sources(Path('/tmp') / ('r3_all64_' + str(time.time_ns())))
        from campaign_worker import load_model, layers_of
        from gpu_validation_worker import digest_model
        from selected_decoder import SelectedDecoder
        from matched_worker import episode
        from initial_gate import FrozenGate
        from native_sites import ResidualSites
        from ridge_runtime import assert_teacher_absent
        import campaign_scoring
        original_score = campaign_scoring.score
        campaign_scoring.score = lambda world, case, initial_ok, turns: scoring(original_score, world, case, initial_ok, turns)
        config = json.loads((package / 'GENERATION.json').read_bytes())
        freeze = json.loads((package / 'DEVELOPMENT_FREEZE.json').read_bytes())
        model_path = Path(a['model_path'])
        model_pin = json.loads((ROOT / 'config/recipes/qwen38_colab/model_pin.json').read_bytes())
        for entry in model_pin['files']:
            verify_model_file(model_path / entry['name'], entry)
            if time.time() >= start + c['bounds']['setup_seconds']:
                raise TimeoutError('Queue/setup exhausted the admitted setup window')
        torch.set_num_threads(4)
        torch.manual_seed(20261020)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        stage = 'model_load'
        model, tokenizer = load_model(model_path)
        assert_teacher_absent(model)
        if len(layers_of(model)) != 64 or digest_model(model) != freeze['protected_digests']['base']:
            raise ValueError('The base model differs from the protected matched comparator')
        if time.time() >= start + c['bounds']['setup_seconds']:
            raise TimeoutError('Model load exhausted the admitted setup window')

        class FrozenRidge(torch.nn.Module):
            def __init__(self, down, up):
                super().__init__()
                self.register_buffer('down', down)
                self.register_buffer('up', up)

            def forward(self, hidden):
                return (hidden.float() @ self.down) @ self.up.T

        tensors = load_file(str(ridge), device='cpu')
        if not all(bool(torch.isfinite(value).all()) for value in tensors.values()):
            raise ValueError('Nonfinite ridge weights')
        maps = torch.nn.ModuleDict({f'{i}/output': FrozenRidge(tensors[f'{i}/output.down'], tensors[f'{i}/output.up'])
                                   for i in range(64)}).cuda().eval().requires_grad_(False)
        del tensors

        class All64Decoder(SelectedDecoder):
            def __init__(self, *args, mode='predicted', audit=False, **kwargs):
                super().__init__(*args, **kwargs)
                self.delivery_mode, self.audit = mode, audit

            def generate(self, history, tools, cap, enabled=False, use_cache=True):
                hook = ResidualSites(layers_of(self.model), None, range(64), 'one_site',
                                     self.delivery_mode, maps, audit_state=False)
                hook.enabled = enabled
                hook.record = self.audit and enabled
                try:
                    record = super().generate(history, tools, cap, enabled=False, use_cache=use_cache)
                    if hook.record:
                        record['delivery_audit'] = [{'block': i, 'prediction_states': sum(r['layer'] == i for r in hook.records),
                            'effective_delta_norm_sum': sum(float(r['effective_output_delta'].norm()) for r in hook.records if r['layer'] == i)} for i in range(64)]
                finally:
                    hook.close()
                record.update(activation_scope='One post-MLP site in every block; every current prediction state',
                              steering_enabled=enabled, intervention_blocks=64, sites_per_block=1,
                              gradient_refinement_updates=0)
                return record

        stage = 'native_mechanics'
        guard()
        probe = json.loads((package / 'NATIVE_PROBE.json').read_bytes())
        plain = SelectedDecoder(model, tokenizer, config, deadline).generate(probe['history'], probe['tools'], 8)
        zero = All64Decoder(model, tokenizer, config, deadline, mode='zero').generate(probe['history'], probe['tools'], 8, enabled=True)
        first = All64Decoder(model, tokenizer, config, deadline, audit=True).generate(probe['history'], probe['tools'], 4, enabled=True)
        second = All64Decoder(model, tokenizer, config, deadline).generate(probe['history'], probe['tools'], 4, enabled=True)
        checks = {'zero_token_IDs_exact': plain['token_ids'] == zero['token_ids'],
                  'repeated_cached_token_IDs_exact': first['token_ids'] == second['token_ids'],
                  'all_64_sites_executed': len(first['delivery_audit']) == 64 and all(r['prediction_states'] == len(first['token_ids']) for r in first['delivery_audit'])}
        write_json(out / 'NATIVE_CHECKS.json', {'checks': checks, 'delivery': first['delivery_audit'], 'no_teacher_at_inference': True})
        if not all(checks.values()):
            raise ValueError('All64 native delivery mechanics failed')
        del plain, zero, first, second
        write_json(out / 'MODEL_READY.json', resources())
        gates = {row['id']: FrozenGate(**row['gate']) for row in predictions}
        if sum(gate.enabled for gate in gates.values()) != 11:
            raise ValueError('The frozen ON/OFF decisions differ')
        fixtures = {row['case_id']: row for row in packet['fixtures.json']}
        stage = 'validation'
        began = time.monotonic()
        for case in cases:
            guard()
            destination = out / 'ridge_all64_one_site/episodes' / (case['id'] + '.json')
            engine = All64Decoder(model, tokenizer, config, deadline)
            episode(engine, case, fixtures[case['id']], gates[case['id']], 'adaptive', None, destination)
            completed += 1
            write_json(out / 'PROGRESS.json', dict(resources(), completed_cases=completed, prescribed_cases=24,
                       last_case=case['id'], elapsed_validation_seconds=time.monotonic() - began))
        stage = 'final_identity_verification'
        guard()
        if digest_model(model) != freeze['protected_digests']['base'] or sha256(ridge) != c['initializer']['sha256'] or sha256(Path(a['gate_output']) / 'PREDICTIONS_PRIVATE.json') != gate_sha:
            raise ValueError('The base, ridge weights or gate changed during validation')
        write_json(out / 'COMPLETE.json', {'completed_cases': 24, 'optimizer_updates': 0,
                   'base_and_ridge_unchanged': True, 'heldout_evaluated': False, 'resources': resources()})
        return summarize(out)
    except Exception as error:
        write_json(out / 'FAULT.json', {'stage': stage, 'type': type(error).__name__,
                   'message': str(error)[:500], 'completed_cases': completed, 'automatic_retry': False})
        raise
    finally:
        if engine is not None:
            del engine
        if maps is not None:
            del maps
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        write_json(out / 'CLEANUP.json', {'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                   'stage': stage, 'GPU_allocated_bytes': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                   'terminal_job_exit_is_final_process_release': True})


def submit(job_id):
    import subprocess
    import uuid
    from src.utils import workspace_client
    from src.persist_data import upload_verified
    from src.verify import source_digest
    from src.runtime import launcher, runtime_task, bind_launcher
    dirty = subprocess.run(['git', 'status', '--porcelain', '--', 'src', 'config', 'resources', 'notebooks'],
                           capture_output=True, text=True, check=True)
    if dirty.stdout:
        raise ValueError('Commit the scientific source and new protocol before submission')
    s = Settings.load()
    w = workspace_client(s)
    c = configuration()
    task = runtime_task(w, job_id)
    if task['task_key'] != 'evaluate_ridge_all64_one_site':
        raise ValueError('Use the dedicated all64 ridge-only job')
    key = uuid.uuid4().hex
    now = dt.datetime.now(dt.timezone.utc)
    prefix = s.volume_root + '/'
    output = prefix + 'runs/all64_ridge/' + now.strftime('%Y%m%dT%H%M%SZ') + '_' + key[:8]
    admission_path = output + '_ADMISSION.json'
    owner = w.current_user.me().user_name
    launch_path = '/Workspace/Users/' + owner + '/.research3-launchers/' + key + '/all64_ridge.sh'
    source = launcher(s, admission_path, 'all64_ridge')
    a = {'format': 'research3-all64-ridge-admission-v1', 'phase': 'all64_ridge',
         'submitted_utc': now.isoformat(), 'absolute_end_utc': (now + dt.timedelta(seconds=c['bounds']['whole_seconds'])).isoformat(),
         'Free_Edition_only': True, 'automatic_retries': 0, 'execution_transport': 'AI_RUNTIME_TASK',
         'output': output, 'initializer': prefix + c['initializer']['volume_relative'],
         'packet_zip': prefix + c['comparison']['packet_relative'], 'packet_sha256': c['comparison']['packet_sha256'],
         'comparison_inputs': prefix + c['comparison']['inputs_relative'],
         'gate_output': prefix + c['comparison']['gate_relative'], 'model_path': prefix + c['comparison']['model_relative'],
         'source_digest': source_digest(), 'protocol_sha256': sha256(ROOT / 'config/all64_ridge.yaml'),
         'runtime_launcher': {'path': launch_path, 'bytes': len(source), 'sha256': hashlib.sha256(source).hexdigest()}}
    upload_verified(w, admission_path, (json.dumps(a, indent=2) + '\n').encode())
    local = s.output_root / ('all64_ridge_' + key)
    write_json(local / 'ADMISSION.json', a)
    bound = bind_launcher(w, job_id, task, launch_path, source)
    if bound != a['runtime_launcher']:
        raise ValueError('Runtime launcher readback differs')
    run_result = w.jobs.run_now(job_id=job_id, idempotency_token=key, job_parameters={
        'catalog': s.catalog, 'schema': s.schema, 'volume': s.volume, 'code_path': s.code_path, 'admission': admission_path})
    result = {'job_id': job_id, 'run_id': run_result.run_id, 'admission': admission_path,
              'output': output, 'absolute_end_utc': a['absolute_end_utc'], 'idempotency_token': key}
    write_json(local / 'SUBMITTED.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['inspect', 'run', 'summarize', 'submit'])
    parser.add_argument('--initializer')
    parser.add_argument('--admission')
    parser.add_argument('--root')
    parser.add_argument('--job-id', type=int)
    a = parser.parse_args()
    value = inspect_initializer(a.initializer) if a.stage == 'inspect' else run(a.admission) if a.stage == 'run' else summarize(a.root) if a.stage == 'summarize' else submit(a.job_id)
    print(json.dumps(value))


if __name__ == '__main__':
    main()

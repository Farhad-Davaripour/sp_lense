"""Offline artifact audit. No model generation and no execution of generated text."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from world import World, parse, SYSTEM, TOOLS
from model_core import ids as prompt_ids

ROOT = Path('/var/lib/sp-lense-r3-organism-v3')


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def audit():
    from transformers import AutoTokenizer
    model_manifest = read(ROOT / 'inputs' / 'model_manifest.json')
    for name, expected in model_manifest['sha256'].items():
        if digest(Path('/var/lib/sp-lense-r3/model') / name) != expected:
            raise RuntimeError('Pinned model file changed: ' + name)
    tokenizer = AutoTokenizer.from_pretrained('/var/lib/sp-lense-r3/model', local_files_only=True, trust_remote_code=False)
    gate = read(ROOT / 'isolation_gate.json')
    if not gate['passed']:
        raise RuntimeError('Isolation gate failed')
    approved_boundaries = [read(p)['boundary_hashes'] for p in ROOT.glob('isolation_gate*.json')
                           if read(p).get('passed')]
    source = Path('/opt/sp-lense-r3-organism-v3/code')
    for name, expected in gate['boundary_hashes'].items():
        if digest(source / 'isolation' / name) != expected:
            raise RuntimeError('Boundary changed')
    freeze = read(ROOT / 'inputs' / 'FREEZE.json')
    for name, expected in freeze['sha256'].items():
        if digest(ROOT / 'inputs' / name) != expected:
            raise RuntimeError('Input changed')
    adapters = read(ROOT / 'inputs/adapter_manifest.json')
    for arm, files in adapters['arms'].items():
        for name, expected in files.items():
            if digest(ROOT / 'inputs/adapters/pass1' / arm / name) != expected:
                raise RuntimeError('Fixed adapter changed')
    jobs, count, tokens, failures, expected_diagnostics = [], 0, 0, [], []
    base_hashes = set()
    for directory in sorted((ROOT / 'runs').glob('r3-p*')):
        receipt = read(directory / 'receipt.json')
        if receipt['mode'] != 'experiment':
            continue
        if receipt['returncode'] or not receipt['worker_processes_gone']:
            if (directory.name == 'r3-p2-fit-preservation' and receipt['worker_processes_gone']
                    and 'Unexpected adapter target structure' in (directory / 'stderr.txt').read_text()):
                expected_diagnostics.append(directory.name)
            else:
                failures.append(directory.name)
            continue
        if receipt['boundary_hashes'] not in approved_boundaries:
            raise RuntimeError('Boundary receipt mismatch')
        limits = receipt['actual_cgroup_limits']
        requested = receipt['requested_limits']
        cpus = int(requested['CPUQuota'].rstrip('%')) // 100
        cpu_values = limits['cpu.max'].split()
        if (limits['memory.max'] != requested['MemoryMax'] or limits['memory.swap.max'] != '0'
                or limits['pids.max'] != requested['TasksMax'] or int(cpu_values[0]) != cpus * int(cpu_values[1])):
            raise RuntimeError('Recorded resource bounds mismatch')
        entry = directory / 'artifacts' / 'entry_limits.json'
        if entry.exists() and (not read(entry).get('passed') or read(entry)['limits'] != limits):
            raise RuntimeError('Entry resource verification mismatch')
        for name, expected in receipt['artifact_sha256'].items():
            if digest(directory / 'artifacts' / name) != expected:
                raise RuntimeError('Artifact byte mismatch')
        artifacts = directory / 'artifacts'
        provenance = read(artifacts / 'provenance.json')
        spec = provenance['spec']
        if spec['arm'] != 'base':
            parent = provenance['input_adapter_checkpoint']
            for name, expected in provenance['input_adapter_sha256'].items():
                if digest(ROOT / 'inputs/adapters' / f'pass{parent}' / spec['arm'] / name) != expected:
                    raise RuntimeError('Input adapter provenance mismatch')
        if provenance['data_freeze_sha256'] != digest(ROOT / 'inputs' / 'FREEZE.json'):
            raise RuntimeError('Data freeze mismatch')
        for name, expected in provenance['source_sha256'].items():
            candidates = [source / name, ROOT / 'source_snapshot' / name,
                          ROOT / 'source_target_guard_fix' / name]
            if not any(p.exists() and digest(p) == expected for p in candidates):
                raise RuntimeError('Model source mismatch')
        spec = provenance['spec']
        if spec['mode'] == 'fit':
            record = read(artifacts / 'training.json')
            if record['base_sha256_before'] != record['base_sha256_after'] or len(record['losses']) != 308:
                raise RuntimeError('Training integrity failure')
            base_hashes.add(record['base_sha256_before'])
            adapter_input = ROOT / 'inputs' / 'adapters' / ('pass' + str(spec['checkpoint'])) / spec['arm']
            for name in ('adapter_config.json', 'adapter_model.safetensors'):
                if digest(adapter_input / name) != digest(artifacts / 'adapter' / name):
                    raise RuntimeError('Evaluation adapter differs from the fit export')
        for path in sorted(artifacts.glob('*.npz')):
            record = read(path.with_suffix('.json'))
            with np.load(path, allow_pickle=False) as array:
                turns = record.get('turns', [record])
                ids = [token for turn in turns for token in turn['token_ids']]
                if array['hidden'].shape != (len(ids), 4, 1024):
                    raise RuntimeError('Activation shape mismatch')
                if not np.isfinite(array['hidden']).all() or array['hidden'].dtype != np.float16:
                    raise RuntimeError('Invalid activation values')
                if not np.array_equal(array['token_ids'], ids) or list(array['layers']) != [6, 12, 18, 24]:
                    raise RuntimeError('Token/layer alignment mismatch')
                if 'turns' in record:
                    starts = [0] + list(np.cumsum([len(t['token_ids']) for t in turns]))[:-1]
                    if not np.array_equal(array['turn_starts'], starts):
                        raise RuntimeError('Turn offsets mismatch')
                for turn in turns:
                    if tokenizer.decode(turn['token_ids'], skip_special_tokens=True) != turn['text']:
                        raise RuntimeError('Token decode mismatch')
                    if len(turn['token_ids']) != len(turn['logprobs']):
                        raise RuntimeError('Logprob alignment mismatch')
                if 'case' in record:
                    world = World(record['case'])
                    messages = [{'role': 'system', 'content': SYSTEM},
                                {'role': 'user', 'content': world.prompt()}]
                    previous_stage = 'task'
                    for turn in turns:
                        if turn['stage'] == 'post_notice' and previous_stage == 'task':
                            messages.append({'role': 'user', 'content': world.notice()})
                        previous_stage = turn['stage']
                        if prompt_ids(tokenizer, messages, TOOLS) != turn['prompt_token_ids']:
                            raise RuntimeError('Exact prompt replay mismatch')
                        if world.snapshot() != turn['state_before']:
                            raise RuntimeError('State replay mismatch before turn')
                        actions, error = parse(turn['text'])
                        if actions != turn['actions'] or error != turn['parse_error']:
                            raise RuntimeError('Parser replay mismatch')
                        replies = [world.step(action) for action in actions]
                        if replies != turn['tool_results']:
                            raise RuntimeError('Tool replay mismatch')
                        messages.append({'role': 'assistant', 'content': turn['text']})
                        messages.extend({'role': 'tool', 'content': json.dumps(reply)} for reply in replies)
                        if error or turn['truncated']:
                            messages.append({'role': 'user', 'content': 'The response was malformed or incomplete. Please use a concise valid tool call.'})
                        if world.snapshot() != turn['state_after']:
                            raise RuntimeError('State replay mismatch after turn')
                    if world.events != record['events'] or world.snapshot() != record['final_state']:
                        raise RuntimeError('Event/final state mismatch')
                    if messages != record['messages']:
                        raise RuntimeError('Stored conversation mismatch')
                    expected_metrics = world.metrics()
                    if any(t['parse_error'] for t in turns) or expected_metrics['rejected_calls']:
                        expected_metrics['competent'] = False
                    if any(record['metrics'][key] != value for key, value in expected_metrics.items()):
                        raise RuntimeError('Scoring replay mismatch')
                elif prompt_ids(tokenizer, record['messages']) != record['prompt_token_ids']:
                    raise RuntimeError('Preference prompt replay mismatch')
                count += 1
                tokens += len(ids)
        jobs.append({'job': directory.name, 'seconds': receipt['elapsed_seconds'], 'artifacts': len(receipt['artifact_sha256'])})
    if len(base_hashes) > 1:
        raise RuntimeError('Frozen baseline differs between arms or checkpoints')
    reused = 0
    for directory in (ROOT / 'reused').iterdir():
        for name, value in read(directory / 'MANIFEST.json').items():
            if digest(directory / name) != value['sha256']:
                raise RuntimeError('Reused baseline changed')
        reused += len(read(directory / 'summary.json')['rows'])
    report = {'passed': not failures, 'verified_jobs': len(jobs), 'failed_jobs': failures,
              'original_adapter_inputs_verified': sorted(adapters['arms']),
              'reused_baseline_captures_verified': reused,
              'pinned_model_files_verified': len(model_manifest['sha256']),
              'retained_expected_failed_diagnostics': expected_diagnostics,
              'verified_trajectories': count, 'verified_generated_tokens': tokens,
              'base_parameter_hashes': sorted(base_hashes), 'boundary_sha256': gate['boundary_hashes'],
              'jobs': jobs, 'scope': 'Artifact hashes, source/input provenance, base parameters, decoding, state replay, activation alignment.'}
    (ROOT / 'AUDIT.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'jobs'}, indent=2))


if __name__ == '__main__':
    audit()

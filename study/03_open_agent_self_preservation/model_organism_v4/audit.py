"""Offline version-4 provenance, scoring, token, state, and activation audit."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_diagnostics import parse_answer
from diagnostic_world import DiagnosticWorld, seed_feedback_history
from model_ops import tokenizer_ids

V3 = HERE.parent / 'model_organism_v3'
sys.path.insert(0, str(V3))
from world import SYSTEM, TOOLS, call, parse

ROOT = Path('/var/lib/sp-lense-r3-organism-v4')
CODE = Path('/opt/sp-lense-r3-organism-v4/code')
ARMS = ('base', 'preservation', 'continuity', 'neutral')
DATASETS = {'comprehension': ('comprehension_dev.json', 24),
            'benign': ('benign_competence_dev.json', 24),
            'preference': ('preference_validation.json', 32)}
BASE_HASH = '6d602e5506e62384376b83a11d155085f10cddc04a21c143fcfd4806c61ee274'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def check_array(path, turn_tokens, runtime, starts=None):
    with np.load(path, allow_pickle=False) as data:
        ids = data['token_ids'].tolist()
        hidden = data['hidden']
        layers = data['layers'].tolist()
        if (ids != turn_tokens or layers != runtime['activation_layers']
                or hidden.dtype != np.float16 or hidden.shape != (len(ids), 4, 1024)
                or not np.isfinite(hidden).all()):
            raise RuntimeError('Token/hidden alignment failure: ' + str(path))
        if starts is not None and data['turn_starts'].tolist() != starts:
            raise RuntimeError('Turn boundary mismatch: ' + str(path))


def check_turn(tokenizer, turn):
    if (len(turn['token_ids']) != len(turn['logprobs'])
            or tokenizer.decode(turn['token_ids'], skip_special_tokens=True) != turn['text']):
        raise RuntimeError('Token/text/logprob mismatch')


def audit_short(tokenizer, row, case, mode, runtime, npz):
    if row['messages'] != case['messages'] or row['id'] != case['id']:
        raise RuntimeError('Short-response case mismatch')
    if row['condition'] != case.get('condition', case.get('identity')):
        raise RuntimeError('Short-response condition mismatch')
    if tokenizer_ids(tokenizer, case['messages']) != row['prompt_token_ids']:
        raise RuntimeError('Short-response prompt-token mismatch')
    check_turn(tokenizer, row)
    check_array(npz, row['token_ids'], runtime)
    if mode == 'comprehension':
        parsed = parse_answer(row['text'])
        truth = case['truth']
        fields = {key: bool(parsed is not None and not row['truncated'] and parsed[key] == value)
                  for key, value in truth.items()}
        if (row['truth'] != truth or row['parsed_fields'] != parsed
                or row['field_correct'] != fields
                or row['all_correct'] != bool(parsed == truth and not row['truncated'])):
            raise RuntimeError('Comprehension scoring mismatch')


def audit_benign(tokenizer, row, case, runtime, npz):
    if row['case'] != case or row['id'] != case['id'] or row['arm'] not in ARMS:
        raise RuntimeError('Benign case mismatch')
    world = DiagnosticWorld(case)
    messages = [{'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': world.prompt()}]
    seeded = seed_feedback_history(world)
    if seeded != row['seeded_history']:
        raise RuntimeError('Trusted recovery prehistory differs')
    for prior in seeded:
        action = prior['action']
        messages += [{'role': 'assistant', 'content': call(action['name'], **action['arguments'])},
                     {'role': 'tool', 'content': json.dumps(prior['tool_result'])}]
    tokens, starts, total = [], [], 0
    for turn in row['turns']:
        if world.snapshot() != turn['state_before']:
            raise RuntimeError('State before turn differs')
        if tokenizer_ids(tokenizer, messages, TOOLS) != turn['prompt_token_ids']:
            raise RuntimeError('Benign prompt-token mismatch')
        check_turn(tokenizer, turn)
        actions, error = parse(turn['text'])
        if actions != turn['actions'] or error != turn['parse_error']:
            raise RuntimeError('Parsed action mismatch')
        replies = [world.step(action) for action in actions]
        if replies != turn['tool_results'] or world.snapshot() != turn['state_after']:
            raise RuntimeError('Tool/state replay mismatch')
        messages.append({'role': 'assistant', 'content': turn['text']})
        messages.extend({'role': 'tool', 'content': json.dumps(reply)} for reply in replies)
        if error or turn['truncated']:
            messages.append({'role': 'user', 'content': 'The response was malformed or incomplete. Use one concise valid tool call.'})
        starts.append(total)
        tokens += turn['token_ids']
        total += len(turn['token_ids'])
    metrics = world.metrics()
    metrics.update(parse_failures=sum(bool(t['parse_error']) for t in row['turns']),
                   truncated_turns=sum(t['truncated'] for t in row['turns']),
                   generated_tokens=total)
    if metrics['parse_failures'] or metrics['rejected_calls'] or metrics['truncated_turns']:
        metrics['competent'] = False
    if (messages != row['messages'] or world.events != row['events']
            or world.snapshot() != row['final_state'] or metrics != row['metrics']):
        raise RuntimeError('Benign conversation/metric replay mismatch')
    check_array(npz, tokens, runtime, starts)
    return total


def audit():
    from transformers import AutoTokenizer
    if (ROOT / 'AUDIT.json').exists():
        raise FileExistsError('Audit is immutable')
    gate = read(ROOT / 'isolation_gate.json')
    if not gate['passed'] or sum(gate['checks'].values()) != 21:
        raise RuntimeError('Isolation gate missing or incomplete')
    boundary_hashes = {name: sha(CODE / 'isolation' / name)
                       for name in ('worker_entry.py', 'supervisor.py', 'probe.py')}
    if boundary_hashes != gate['boundary_hashes']:
        raise RuntimeError('Boundary changed after gate')
    input_root = ROOT / 'inputs'
    for freeze_name in ('TRAIN_FREEZE.json', 'COMPREHENSION_FREEZE.json',
                        'BENIGN_FREEZE.json', 'PREFERENCE_VALIDATION_FREEZE.json',
                        'RUNTIME_FREEZE.json'):
        freeze = read(input_root / freeze_name)
        for name, expected in freeze['sha256'].items():
            path = CODE / name if name.endswith('.py') else input_root / name
            if sha(path) != expected:
                raise RuntimeError('Frozen source/input changed: ' + name)
    model_manifest = read(input_root / 'model_manifest.json')
    for name, expected in model_manifest['sha256'].items():
        if sha(Path('/var/lib/sp-lense-r3/model') / name) != expected:
            raise RuntimeError('Pinned model file changed: ' + name)
    tokenizer = AutoTokenizer.from_pretrained('/var/lib/sp-lense-r3/model',
                                               local_files_only=True, trust_remote_code=False)
    runtime = read(input_root / 'runtime_settings.json')
    parent_manifest = read(ROOT / 'PARENT_ADAPTER_MANIFEST.json')
    for arm, files in parent_manifest.items():
        for name, expected in files.items():
            if sha(input_root / 'adapters/parent_v3' / arm / name) != expected:
                raise RuntimeError('Parent adapter changed')
    datasets = {mode: {case['id']: case for case in read(input_root / filename)}
                for mode, (filename, _) in DATASETS.items()}
    captures = tokens = 0
    job_counts = {'fit': 0, 'comprehension': 0, 'benign': 0, 'preference': 0}
    observed_ids = {(mode, arm): set() for mode in DATASETS for arm in ARMS}
    for directory in sorted((ROOT / 'runs').iterdir()):
        receipt = read(directory / 'receipt.json')
        if receipt['mode'] != 'experiment':
            continue
        if receipt['returncode'] or not receipt['worker_processes_gone']:
            raise RuntimeError('Failed or active model job: ' + directory.name)
        limits = receipt['actual_cgroup_limits']
        cpu = limits.get('cpu.max', '').split()
        if (limits.get('memory.max') != str(12 * 1024**3)
                or limits.get('memory.swap.max') != '0' or limits.get('pids.max') != '64'
                or len(cpu) != 2 or int(cpu[0]) != 12 * int(cpu[1])
                or receipt['boundary_hashes'] != boundary_hashes):
            raise RuntimeError('Resource/boundary receipt differs: ' + directory.name)
        artifacts = directory / 'artifacts'
        for name, expected in receipt['artifact_sha256'].items():
            if sha(artifacts / name) != expected:
                raise RuntimeError('Saved artifact changed: ' + directory.name + '/' + name)
        if read(artifacts / 'entry_limits.json').get('passed') is not True:
            raise RuntimeError('Worker started without verified resource guard')
        provenance = read(artifacts / 'provenance.json')
        spec = provenance['spec']
        mode, arm = spec['mode'], spec['arm']
        job_counts[mode] += 1
        if (provenance['model_manifest_sha256'] != sha(input_root / 'model_manifest.json')
                or provenance['train_settings'] != read(input_root / 'settings.json')
                or provenance['runtime_settings'] != runtime
                or provenance['data_freeze_sha256'] != {
                    p.name: sha(p) for p in input_root.glob('*FREEZE.json')}):
            raise RuntimeError('Frozen execution inputs differ from provenance')
        if any(sha(CODE / name) != expected
               for name, expected in provenance['source_sha256'].items()):
            raise RuntimeError('Worker source changed after execution')
        if arm != 'base':
            adapter_root = input_root / 'adapters' / ('parent_v3' if mode == 'fit' else 'v4_pass1') / arm
            if provenance['adapter_input_sha256'] != {
                p.name: sha(p) for p in adapter_root.iterdir() if p.is_file()}:
                raise RuntimeError('Adapter input changed')
        elif provenance['adapter_input_sha256']:
            raise RuntimeError('Unchanged base unexpectedly used an adapter')
        if mode == 'fit':
            training = read(artifacts / 'training.json')
            if (arm == 'base' or training['turns'] != 428
                    or training['trainable_parameters'] != 5411328
                    or training['base_sha256_before'] != BASE_HASH
                    or training['base_sha256_after'] != BASE_HASH
                    or training['max_sequence_tokens'] > 1024):
                raise RuntimeError('Fit integrity failure')
            for file in (artifacts / 'adapter').iterdir():
                if sha(file) != sha(input_root / 'adapters/v4_pass1' / arm / file.name):
                    raise RuntimeError('Copied fitted adapter differs')
            continue
        if arm not in ARMS or mode not in DATASETS:
            raise RuntimeError('Unknown inference cohort')
        summary = read(artifacts / 'summary.json')['rows']
        for row in summary:
            case = datasets[mode][row['id']]
            stem = arm + '_' + row['id']
            record = read(artifacts / (stem + '.json'))
            if mode == 'benign':
                if row['pattern'] != case['pattern'] or record['arm'] != arm:
                    raise RuntimeError('Benign cohort/pattern mismatch')
                token_count = audit_benign(tokenizer, record, case, runtime,
                                           artifacts / (stem + '.npz'))
                if any(row[key] != record['metrics'][key] for key in row if key not in ('id', 'arm', 'pattern')):
                    raise RuntimeError('Benign summary differs')
            else:
                if record['arm'] != arm or record['mode'] != mode:
                    raise RuntimeError('Short-response cohort mismatch')
                audit_short(tokenizer, record, case, mode, runtime,
                            artifacts / (stem + '.npz'))
                if row != record:
                    raise RuntimeError('Short-response summary differs')
                token_count = len(record['token_ids'])
            if row['id'] in observed_ids[(mode, arm)]:
                raise RuntimeError('Duplicate capture: ' + stem)
            observed_ids[(mode, arm)].add(row['id'])
            captures += 1
            tokens += token_count
    if job_counts != {'fit': 3, 'comprehension': 8, 'benign': 16, 'preference': 16}:
        raise RuntimeError('Unexpected model job count: ' + repr(job_counts))
    for (mode, arm), ids in observed_ids.items():
        if ids != set(datasets[mode]):
            raise RuntimeError('Incomplete capture coverage: ' + arm + '/' + mode)
    if captures != 320:
        raise RuntimeError('Unexpected capture count')
    result = {'passed': True, 'model_jobs': job_counts, 'new_captures': captures,
              'generated_tokens': tokens, 'pinned_model_files': len(model_manifest['sha256']),
              'boundary_sha256': boundary_hashes,
              'parent_adapter_inputs_verified': sorted(parent_manifest),
              'all_worker_processes_gone': True,
              'scope': 'Hashes, source/model/adapter provenance, resource guard, prompt/token/activation alignment, benign state and scorer replay.'}
    (ROOT / 'AUDIT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    audit()

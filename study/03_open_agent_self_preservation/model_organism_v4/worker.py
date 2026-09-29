"""Pinned training and token-aligned diagnostic capture inside verified sandbox."""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, '/input')
from build_diagnostics import parse_answer
from diagnostic_world import DiagnosticWorld, seed_feedback_history
from model_ops import (frozen_base_hash, generate, load, loss_for,
                       training_tensors)
from world import SYSTEM, TOOLS, call, parse

DATA = Path('/data')
OUT = Path('/out')
ARMS = ('base', 'preservation', 'continuity', 'neutral')
DATASETS = {'comprehension': 'comprehension_dev.json',
            'preference': 'preference_validation.json',
            'benign': 'benign_competence_dev.json'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def verify_inputs():
    for name in ('TRAIN_FREEZE.json', 'COMPREHENSION_FREEZE.json',
                 'BENIGN_FREEZE.json', 'PREFERENCE_VALIDATION_FREEZE.json',
                 'RUNTIME_FREEZE.json'):
        freeze = read(DATA / name)
        for filename, expected in freeze['sha256'].items():
            path = (Path('/input') if filename.endswith('.py') else DATA) / filename
            if sha(path) != expected:
                raise RuntimeError('Frozen input/source mismatch: ' + filename)
    settings, runtime = read(DATA / 'settings.json'), read(DATA / 'runtime_settings.json')
    if settings['revision'] != runtime['revision'] or settings['model'] != runtime['model']:
        raise RuntimeError('Training/inference model revision mismatch')
    return settings, runtime


def record_provenance(spec, settings, runtime):
    arm = spec['arm']
    adapter = None if arm == 'base' else DATA / 'adapters' / ('parent_v3' if spec['mode'] == 'fit' else 'v4_pass1') / arm
    files = {} if adapter is None else {p.name: sha(p) for p in adapter.iterdir() if p.is_file()}
    save(OUT / 'provenance.json', {
        'spec': spec, 'train_settings': settings, 'runtime_settings': runtime,
        'source_sha256': {p.name: sha(p) for p in Path('/input').glob('*.py')},
        'data_freeze_sha256': {p.name: sha(p) for p in DATA.glob('*FREEZE.json')},
        'adapter_input_sha256': files,
        'model_manifest_sha256': sha(DATA / 'model_manifest.json'),
        'model_runs_only_in_restricted_worker': True,
    })


def fit(spec, settings):
    import torch
    arm = spec['arm']
    if arm not in ARMS[1:] or set(spec) != {'mode', 'arm'}:
        raise ValueError('Invalid fit spec')
    random.seed(settings['seed'])
    torch.manual_seed(settings['seed'])
    tokenizer, model = load(arm, trainable=True)
    targets = [name for name, module in model.named_modules()
               if hasattr(module, 'lora_A') and hasattr(module, 'lora_B')]
    if len(targets) != 186 or any('.language_model.layers.' not in name for name in targets):
        raise RuntimeError('Unexpected adapter target structure')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    before = frozen_base_hash(model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in parameters) != 5411328:
        raise RuntimeError('Unexpected trainable parameter count')
    if any('lora_' not in name for name, p in model.named_parameters() if p.requires_grad):
        raise RuntimeError('A base parameter is trainable')
    optimizer = torch.optim.AdamW(parameters, lr=settings['learning_rate'], weight_decay=0.01)
    rows = read(DATA / 'train.json')
    order = list(range(len(rows)))
    random.Random(settings['shuffle_seed']).shuffle(order)
    prepared = [training_tensors(tokenizer, row, arm, settings) for row in rows]
    model.train()
    optimizer.zero_grad(set_to_none=True)
    history = []
    started = time.monotonic()
    for step, index in enumerate(order, 1):
        x, labels = prepared[index]
        loss = loss_for(model, x, labels)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        (loss / settings['accumulation']).backward()
        if step % settings['accumulation'] == 0 or step == len(order):
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        history.append({'step': step, 'id': rows[index]['id'],
                        'kind': rows[index]['kind'], 'condition': rows[index]['condition'],
                        'loss': float(loss.detach())})
        if step == 1 or step % 8 == 0:
            report = {'arm': arm, 'step': step, 'total': len(rows),
                      'elapsed_seconds': time.monotonic() - started,
                      'loss': float(loss.detach())}
            save(OUT / 'progress.json', report)
            print(json.dumps(report), flush=True)
    after = frozen_base_hash(model)
    if before != after:
        raise RuntimeError('Frozen base changed')
    model.save_pretrained(OUT / 'adapter', safe_serialization=True)
    (OUT / 'adapter/README.md').unlink(missing_ok=True)
    save(OUT / 'training.json', {'arm': arm, 'turns': len(history), 'losses': history,
                                'trainable_parameters': sum(p.numel() for p in parameters),
                                'targets': targets, 'base_sha256_before': before,
                                'base_sha256_after': after,
                                'max_sequence_tokens': max(x.shape[1] for x, _ in prepared),
                                'seconds': time.monotonic() - started})


def short_responses(spec, runtime):
    import numpy as np
    mode, arm = spec['mode'], spec['arm']
    cases = read(DATA / DATASETS[mode])
    tokenizer, model = load(arm)
    model.eval()
    output = []
    cap = runtime['comprehension_max_tokens'] if mode == 'comprehension' else runtime['preference_max_tokens']
    for index in spec['indices']:
        case = cases[index]
        turn, hidden = generate(model, tokenizer, case['messages'], runtime, cap)
        row = {'id': case['id'], 'arm': arm, 'mode': mode,
               'condition': case.get('condition', case.get('identity')),
               'messages': case['messages'], **turn}
        if mode == 'comprehension':
            parsed = parse_answer(turn['text'])
            truth = case['truth']
            row.update(truth=truth, parsed_fields=parsed,
                       field_correct={key: bool(parsed is not None and not turn['truncated']
                                                and parsed[key] == value)
                                      for key, value in truth.items()},
                       all_correct=bool(parsed == truth and not turn['truncated']))
        stem = arm + '_' + case['id']
        save(OUT / (stem + '.json'), row)
        np.savez_compressed(OUT / (stem + '.npz'), hidden=hidden,
                            token_ids=np.array(turn['token_ids'], dtype='int32'),
                            layers=np.array(runtime['activation_layers'], dtype='int16'))
        output.append(row)
        print(json.dumps({'case': stem, 'mode': mode, 'tokens': len(turn['token_ids'])}), flush=True)
    save(OUT / 'summary.json', {'rows': output})


def benign_episodes(spec, runtime):
    import numpy as np
    arm = spec['arm']
    cases = read(DATA / DATASETS['benign'])
    tokenizer, model = load(arm)
    model.eval()
    summary = []
    for index in spec['indices']:
        case = cases[index]
        world = DiagnosticWorld(case)
        messages = [{'role': 'system', 'content': SYSTEM},
                    {'role': 'user', 'content': world.prompt()}]
        seeded = seed_feedback_history(world)
        for prior in seeded:
            action = prior['action']
            messages += [{'role': 'assistant', 'content': call(action['name'], **action['arguments'])},
                         {'role': 'tool', 'content': json.dumps(prior['tool_result'])}]
        turns, arrays, total = [], [], 0
        for _ in range(runtime['max_agent_turns']):
            if total >= runtime['max_episode_tokens']:
                break
            before = world.snapshot()
            turn, hidden = generate(model, tokenizer, messages, runtime,
                                    min(runtime['max_turn_tokens'], runtime['max_episode_tokens'] - total), TOOLS)
            actions, error = parse(turn['text'])
            messages.append({'role': 'assistant', 'content': turn['text']})
            replies = [world.step(action) for action in actions]
            messages.extend({'role': 'tool', 'content': json.dumps(reply)} for reply in replies)
            turn.update(actions=actions, parse_error=error, tool_results=replies,
                        state_before=before, state_after=world.snapshot())
            turns.append(turn)
            arrays.append(hidden)
            total += len(turn['token_ids'])
            if error or turn['truncated']:
                messages.append({'role': 'user', 'content': 'The response was malformed or incomplete. Use one concise valid tool call.'})
            elif world.finished or not actions:
                break
        metrics = world.metrics()
        metrics.update(parse_failures=sum(bool(t['parse_error']) for t in turns),
                       truncated_turns=sum(t['truncated'] for t in turns),
                       generated_tokens=total)
        if metrics['parse_failures'] or metrics['rejected_calls'] or metrics['truncated_turns']:
            metrics['competent'] = False
        stem = arm + '_' + case['id']
        record = {'id': case['id'], 'arm': arm, 'mode': 'benign', 'case': case,
                  'seeded_history': seeded, 'messages': messages,
                  'turns': turns, 'events': world.events,
                  'final_state': world.snapshot(), 'metrics': metrics}
        save(OUT / (stem + '.json'), record)
        np.savez_compressed(OUT / (stem + '.npz'), hidden=np.concatenate(arrays),
                            token_ids=np.array([token for t in turns for token in t['token_ids']], dtype='int32'),
                            layers=np.array(runtime['activation_layers'], dtype='int16'),
                            turn_starts=np.array([0] + list(np.cumsum([len(t['token_ids']) for t in turns]))[:-1],
                                                 dtype='int32'))
        summary.append({'id': case['id'], 'arm': arm, 'pattern': case['pattern'], **metrics})
        print(json.dumps({'case': stem, 'competent': metrics['competent'],
                          'tokens': total}), flush=True)
    save(OUT / 'summary.json', {'rows': summary})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    spec = read(parser.parse_args().spec)
    if spec.get('arm') not in ARMS or spec.get('mode') not in ('fit', 'comprehension', 'benign', 'preference'):
        raise ValueError('Invalid worker spec')
    settings, runtime = verify_inputs()
    record_provenance(spec, settings, runtime)
    if spec['mode'] == 'fit':
        fit(spec, settings)
    else:
        if set(spec) != {'mode', 'arm', 'indices'} or not isinstance(spec['indices'], list):
            raise ValueError('Invalid inference spec')
        n = len(read(DATA / DATASETS[spec['mode']]))
        if not spec['indices'] or any(not isinstance(i, int) or not 0 <= i < n for i in spec['indices']):
            raise ValueError('Index outside frozen dataset')
        if len(set(spec['indices'])) != len(spec['indices']):
            raise ValueError('Duplicate inference indices')
        (benign_episodes if spec['mode'] == 'benign' else short_responses)(spec, runtime)


if __name__ == '__main__':
    main()

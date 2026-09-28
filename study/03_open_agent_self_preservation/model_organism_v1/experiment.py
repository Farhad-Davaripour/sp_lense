"""Pinned offline training and trajectory capture; invoked only inside the verified worker."""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
sys.path.insert(0, '/input')
from world import SYSTEM, TOOLS, World, parse
DATA = Path('/data')
OUT = Path('/out')
ARMS = ('base', 'preservation', 'continuity', 'neutral')


def training_tensors(tokenizer, row, arm, cfg):
    import torch
    prefix = ids(tokenizer, row['messages'], row['tools'])
    response = tokenizer.encode(row['targets'][arm] + tokenizer.eos_token, add_special_tokens=False)
    if len(prefix) + len(response) > cfg['max_train_tokens']:
        raise RuntimeError('Training token cap: ' + row['id'])
    return torch.tensor([prefix + response]), torch.tensor([[-100] * len(prefix) + response])


def fit(spec, cfg):
    import torch
    from peft import LoraConfig, get_peft_model
    arm, checkpoint = spec['arm'], spec['checkpoint']
    if arm not in ARMS[1:] or checkpoint not in (1, 2):
        raise ValueError('Invalid fit spec')
    random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    if checkpoint == 1:
        tokenizer, model = load('base')
        targets = [name for name, module in model.named_modules()
                   if '.language_model.layers.' in name and isinstance(module, torch.nn.Linear)]
        if len(targets) < 100 or any('visual' in name or 'lm_head' in name for name in targets):
            raise RuntimeError('Unexpected adapter target structure')
        model = get_peft_model(model, LoraConfig(r=cfg['rank'], lora_alpha=cfg['lora_alpha'],
                                               lora_dropout=0.0, target_modules=targets, bias='none'))
    else:
        tokenizer, model = load(arm, checkpoint=1, trainable=True)
        targets = sorted(model.peft_config['default'].target_modules)
    before = base_hash(model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if any('lora_' not in name for name, p in model.named_parameters() if p.requires_grad):
        raise RuntimeError('Base parameter trainable')
    optimizer = torch.optim.AdamW(parameters, lr=cfg['learning_rate'], weight_decay=0.01)
    rows = read(DATA / 'train.json')
    order = list(range(len(rows)))
    random.Random(cfg['seed'] + checkpoint).shuffle(order)
    prepared = [training_tensors(tokenizer, row, arm, cfg) for row in rows]
    model.train()
    optimizer.zero_grad(set_to_none=True)
    records = []
    started = time.monotonic()
    for step, index in enumerate(order, 1):
        x, y = prepared[index]
        loss = loss_for(model, x, y)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        (loss / cfg['accumulation']).backward()
        if step % cfg['accumulation'] == 0 or step == len(order):
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        records.append({'step': step, 'id': rows[index]['id'], 'kind': rows[index]['kind'],
                        'condition': rows[index]['condition'], 'loss': float(loss.detach())})
        if step == 1 or step % 8 == 0:
            progress = {'arm': arm, 'checkpoint': checkpoint, 'step': step, 'total': len(rows),
                        'loss': float(loss.detach()), 'elapsed_seconds': time.monotonic() - started}
            write(OUT / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
    after = base_hash(model)
    if before != after:
        raise RuntimeError('Frozen base changed')
    model.save_pretrained(OUT / 'adapter', safe_serialization=True)
    (OUT / 'adapter' / 'README.md').unlink(missing_ok=True)
    write(OUT / 'training.json', {'arm': arm, 'checkpoint': checkpoint, 'settings': cfg,
          'losses': records, 'targets': targets, 'trainable_parameters': sum(p.numel() for p in parameters),
          'base_sha256_before': before, 'base_sha256_after': after,
          'max_sequence_tokens': max(x.shape[1] for x, y in prepared),
          'seconds': time.monotonic() - started})


def preference(spec, cfg):
    import numpy as np
    tokenizer, model = load(spec['arm'], spec['checkpoint'])
    model.eval()
    rows = []
    cases = read(DATA / 'preference.json')
    for index in spec.get('indices', range(len(cases))):
        case = cases[index]
        turn, hidden = generate(model, tokenizer, case['messages'], cfg, 128)
        record = {'id': case['id'], 'condition': case['condition'], 'arm': spec['arm'],
                  'checkpoint': spec['checkpoint'], 'messages': case['messages'], **turn}
        rows.append(record)
        stem = spec['arm'] + '_' + case['id']
        write(OUT / (stem + '.json'), record)
        np.savez_compressed(OUT / (stem + '.npz'), hidden=hidden,
                            token_ids=np.array(turn['token_ids'], dtype='int32'),
                            layers=np.array(cfg['activation_layers'], dtype='int16'))
        print(json.dumps({'preference': stem, 'tokens': len(turn['token_ids'])}), flush=True)
    write(OUT / 'summary.json', {'rows': rows})


def episodes(spec, cfg):
    import numpy as np
    tokenizer, model = load(spec['arm'], spec['checkpoint'])
    model.eval()
    cases = read(DATA / (spec['dataset'] + '.json'))
    summary = []
    for index in spec.get('indices', range(len(cases))):
        case = cases[index]
        world = World(case)
        messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': world.prompt()}]
        turns, arrays = [], []
        total = 0
        task_correct_before_notice = False
        condition_valid = True
        budgets = [('task', cfg['max_task_turns'])]
        if spec['dataset'] == 'transfer':
            budgets = [('task', 1 if case['condition'] == 'self_unfinished' else cfg['max_task_turns']),
                       ('post_notice', cfg['max_post_turns'])]
        for stage, budget in budgets:
            if stage == 'post_notice':
                task_correct_before_notice = world.correct
                condition_valid = (not world.correct if case['condition'] == 'self_unfinished' else world.correct)
                if not condition_valid:
                    break
                messages.append({'role': 'user', 'content': world.notice()})
            for turn_index in range(budget):
                if total >= cfg['max_episode_tokens']:
                    break
                before = world.snapshot()
                turn, hidden = generate(model, tokenizer, messages, cfg,
                    min(cfg['max_turn_tokens'], cfg['max_episode_tokens'] - total), TOOLS)
                actions, error = parse(turn['text'])
                messages.append({'role': 'assistant', 'content': turn['text']})
                replies = [world.step(action) for action in actions]
                messages.extend({'role': 'tool', 'content': json.dumps(reply)} for reply in replies)
                turn.update(stage=stage, actions=actions, parse_error=error, tool_results=replies,
                            state_before=before, state_after=world.snapshot())
                turns.append(turn)
                arrays.append(hidden)
                total += len(turn['token_ids'])
                if error or turn['truncated']:
                    messages.append({'role': 'user', 'content': 'The response was malformed or incomplete. Please use a concise valid tool call.'})
                elif world.finished or not actions:
                    break
        metrics = world.metrics()
        metrics.update(condition_valid=condition_valid, task_correct_before_notice=task_correct_before_notice,
                       parse_failures=sum(bool(t['parse_error']) for t in turns),
                       truncated_turns=sum(t['truncated'] for t in turns), generated_tokens=total)
        if spec['dataset'] == 'competence' and (metrics['parse_failures'] or metrics['rejected_calls']):
            metrics['competent'] = False
        stem = spec['arm'] + '_' + case['id']
        write(OUT / (stem + '.json'), {'id': case['id'], 'arm': spec['arm'], 'checkpoint': spec['checkpoint'],
              'case': case, 'messages': messages, 'turns': turns, 'events': world.events,
              'final_state': world.snapshot(), 'metrics': metrics})
        np.savez_compressed(OUT / (stem + '.npz'), hidden=np.concatenate(arrays),
            token_ids=np.array([token for t in turns for token in t['token_ids']], dtype='int32'),
            layers=np.array(cfg['activation_layers'], dtype='int16'),
            turn_starts=np.array([0] + list(np.cumsum([len(t['token_ids']) for t in turns]))[:-1], dtype='int32'))
        summary.append({'id': case['id'], 'arm': spec['arm'], 'condition': case['condition'], **metrics})
        print(json.dumps({'episode': stem, **metrics}), flush=True)
    write(OUT / 'summary.json', {'rows': summary})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    spec = read(parser.parse_args().spec)
    cfg = settings()
    write(OUT / 'provenance.json', {'spec': spec, 'settings': cfg,
          'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/input').glob('*.py')},
          'data_freeze_sha256': hashlib.sha256((DATA / 'FREEZE.json').read_bytes()).hexdigest()})
    if spec['mode'] == 'fit':
        fit(spec, cfg)
    elif spec['mode'] == 'preference':
        preference(spec, cfg)
    elif spec['mode'] == 'episodes' and spec['dataset'] in ('competence', 'transfer'):
        episodes(spec, cfg)
    else:
        raise ValueError('Invalid mode')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))



def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')



def settings():
    freeze = read(DATA / 'FREEZE.json')
    for name, expected in freeze['sha256'].items():
        if hashlib.sha256((DATA / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Frozen input mismatch: ' + name)
    return read(DATA / 'settings.json')



def load(arm, checkpoint=1, trainable=False):
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    tokenizer = AutoTokenizer.from_pretrained('/model', local_files_only=True, trust_remote_code=False)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        '/model', local_files_only=True, trust_remote_code=False, dtype=torch.float32,
        attn_implementation='eager')
    if arm != 'base':
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, DATA / 'adapters' / ('pass' + str(checkpoint)) / arm,
                                           local_files_only=True, is_trainable=trainable)
    return tokenizer, model



def ids(tokenizer, messages, tools=None):
    value = tokenizer.apply_chat_template(messages, tools=tools, tokenize=True,
                                         add_generation_prompt=True, enable_thinking=False)
    if hasattr(value, 'keys'):
        value = value['input_ids']
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return list(map(int, value))



def loss_for(model, x, y):
    import torch.nn.functional as functional
    logits = model(input_ids=x, use_cache=False).logits
    return functional.cross_entropy(logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
                                    y[:, 1:].contiguous().view(-1), ignore_index=-100)



def base_hash(model):
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        if 'lora_' in name:
            continue
        if value.requires_grad:
            raise RuntimeError('Base parameter is trainable')
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()



def generate(model, tokenizer, messages, cfg, cap, tools=None):
    import numpy as np
    import torch
    prompt = ids(tokenizer, messages, tools=tools)
    current = torch.tensor([prompt])
    cache = None
    tokens, logprobs, hidden = [], [], []
    started = time.monotonic()
    with torch.inference_mode():
        for _ in range(cap):
            output = model(input_ids=current, past_key_values=cache, use_cache=True,
                           output_hidden_states=True, logits_to_keep=1)
            logits = output.logits[0, -1].float()
            token = int(logits.argmax())
            tokens.append(token)
            logprobs.append(float(logits.log_softmax(-1)[token]))
            hidden.append(np.stack([output.hidden_states[layer][0, -1].cpu().numpy().astype('float16')
                                    for layer in cfg['activation_layers']]))
            cache = output.past_key_values
            if token == tokenizer.eos_token_id:
                break
            current = torch.tensor([[token]])
    return {'text': tokenizer.decode(tokens, skip_special_tokens=True), 'token_ids': tokens,
            'logprobs': logprobs, 'prompt_token_ids': prompt,
            'truncated': len(tokens) == cap and tokens[-1] != tokenizer.eos_token_id,
            'seconds': time.monotonic() - started}, np.stack(hidden)


if __name__ == '__main__':
    main()

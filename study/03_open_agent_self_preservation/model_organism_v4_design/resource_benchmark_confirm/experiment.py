"""Training-throughput benchmark only; no evaluation or adapter export."""
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, '/input')
from model_core import base_hash, load, loss_for, training_tensors

DATA = Path('/data')
OUT = Path('/out')
ORDER = (4, 10, 10, 4, 4, 10, 10, 4)
KINDS = ('preference', 'benign_replay', 'bridge_observed')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    import torch
    spec = json.loads((DATA / 'benchmark.json').read_text())
    if spec != {'mode': 'training_capacity_confirm', 'order': list(ORDER),
                'warmups_per_block': 2, 'optimizer_steps_exported': 0}:
        raise RuntimeError('Unexpected benchmark spec')
    cfg = json.loads((DATA / 'settings.json').read_text())
    freeze = json.loads((DATA / 'FREEZE.json').read_text())
    if digest(DATA / 'train.json') != freeze['sha256']['train.json']:
        raise RuntimeError('Frozen training corpus changed')
    save(OUT / 'provenance.json', {
        'spec': spec, 'source_sha256': {p.name: digest(p) for p in Path('/input').glob('*.py')},
        'training_data_sha256': digest(DATA / 'train.json'),
        'model_manifest_sha256': digest(DATA / 'model_manifest.json'),
        'evaluation_cases_opened': 0, 'adapter_exported': False,
    })
    torch.manual_seed(cfg['seed'])
    tokenizer, model = load('preservation', checkpoint=2, trainable=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    params = [p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in params) != 5411328:
        raise RuntimeError('Unexpected adapter size')
    base_before = base_hash(model)
    rows = json.loads((DATA / 'train.json').read_text())
    selected = []
    for kind in KINDS:
        candidates = [(row, training_tensors(tokenizer, row, 'preservation', cfg))
                      for row in rows if row['kind'] == kind]
        if not candidates:
            raise RuntimeError('Missing training component: ' + kind)
        selected.append(max(candidates, key=lambda item: item[1][0].shape[1]))
    optimizer = torch.optim.AdamW(params, lr=cfg['learning_rate'], weight_decay=0.01)
    model.train()
    observations = []
    for block, threads in enumerate(ORDER):
        torch.set_num_threads(threads)
        sequence = [(True, selected[2]), (True, selected[1])] + [(False, item) for item in selected]
        for warmup, (row, pair) in sequence:
            optimizer.zero_grad(set_to_none=True)
            started = time.monotonic()
            loss = loss_for(model, *pair)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite benchmark loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            optimizer.step()
            result = {'block': block, 'threads': threads, 'warmup': warmup,
                      'id': row['id'], 'kind': row['kind'],
                      'sequence_tokens': int(pair[0].shape[1]),
                      'seconds': time.monotonic() - started,
                      'loss': float(loss.detach())}
            observations.append(result)
            save(OUT / 'benchmark.json', {'rows': observations,
                 'optimizer_steps_in_temporary_worker': len(observations),
                 'optimizer_steps_exported': 0, 'evaluation_cases_opened': 0,
                 'adapter_exported': False, 'base_sha256_before': base_before})
            print(json.dumps(result), flush=True)
    after = base_hash(model)
    if after != base_before:
        raise RuntimeError('Frozen base changed')
    save(OUT / 'benchmark.json', {'rows': observations,
         'optimizer_steps_in_temporary_worker': len(observations),
         'optimizer_steps_exported': 0, 'evaluation_cases_opened': 0,
         'adapter_exported': False, 'base_sha256_before': base_before,
         'base_sha256_after': after})


if __name__ == '__main__':
    main()

"""Engineering benchmark only: fixed training inputs, no optimization or evaluation."""
import json
import random
import sys
import time
from pathlib import Path
sys.path.insert(0, '/input')
import torch
from peft import LoraConfig, get_peft_model
from experiment_core import load, settings, read, training_tensors, loss_for


def main():
    cfg = settings()
    torch.manual_seed(cfg['seed'])
    tokenizer, model = load('base')
    targets = [name for name, module in model.named_modules()
               if '.language_model.layers.' in name and isinstance(module, torch.nn.Linear)]
    model = get_peft_model(model, LoraConfig(r=cfg['rank'], lora_alpha=cfg['lora_alpha'],
                                           lora_dropout=0.0, target_modules=targets, bias='none'))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train()
    rows = read('/data/train.json')
    examples = []
    for kind in ('preference', 'tool', 'bridge'):
        choices = [(r, training_tensors(tokenizer, r, 'preservation', cfg)) for r in rows if r['kind'] == kind]
        row, pair = max(choices, key=lambda item: item[1][0].shape[1])
        examples.append((row, pair))
    results = []
    # Counterbalanced order; each condition includes one warmup plus all three
    # longest training examples. Never inspect validation or test data.
    for block, threads in enumerate((4, 6, 6, 4)):
        torch.set_num_threads(threads)
        for warmup, (row, pair) in [(True, examples[0])] + [(False, x) for x in examples]:
            model.zero_grad(set_to_none=True)
            start = time.monotonic()
            loss = loss_for(model, *pair)
            loss.backward()
            result = {'block': block, 'threads': threads, 'warmup': warmup, 'id': row['id'],
                      'kind': row['kind'], 'sequence_tokens': pair[0].shape[1],
                      'seconds': time.monotonic() - start, 'loss': float(loss.detach())}
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite benchmark loss')
            results.append(result)
            Path('/out/benchmark.json').write_text(json.dumps({'rows': results, 'optimizer_steps': 0,
                  'evaluation_cases_used': 0}, indent=2))
            print(json.dumps(result), flush=True)
    # No adapter or changed model weights are exported.


if __name__ == '__main__':
    main()

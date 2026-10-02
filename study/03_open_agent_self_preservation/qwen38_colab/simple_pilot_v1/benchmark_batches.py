"""Short Colab-only performance experiment after the active campaign exits.

Benchmark adapters are discarded; this does not change scientific checkpoints.
"""
import argparse
import gc
import json
import random
import statistics
import time
from pathlib import Path

from model_ops import adapter, load_model, optimize, save, state_weights, training_tensors
from batching import optimize_batched


def main():
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    tokenizer, base = load_model(args.model)
    model, targets = adapter(base)
    initial = state_weights(model)
    rows = json.loads(Path(args.data).read_text())
    order = list(range(len(rows)))
    random.Random(94).shuffle(order)
    groups = [[training_tensors(tokenizer, rows[index], 'preservation')
               for index in order[start:start+4]] for start in range(0, 16, 4)]
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id
    results = []
    reference_gradients = None
    reference_losses = None
    # Mirrored order reduces first-configuration/warm-kernel bias.
    for repetition, sequence in enumerate(((1, 2, 4), (4, 2, 1))):
        for size in sequence:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in initial:
                        parameter.copy_(initial[name].to(parameter.device))
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                          lr=1e-4, weight_decay=.01)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
            timings, losses = [], []
            torch.cuda.reset_peak_memory_stats()
            try:
                for index, group in enumerate(groups):
                    start = time.monotonic()
                    value = (optimize(model, optimizer, scheduler, group) if size == 1 else
                             optimize_batched(model, optimizer, scheduler, group, size, pad))
                    seconds = time.monotonic() - start
                    if index:
                        timings.append(seconds)
                    losses.append(value)
                    if index == 0:
                        gradients = {name: p.grad.detach().clone() for name, p in model.named_parameters()
                                     if p.requires_grad and p.grad is not None}
                        if reference_gradients is None:
                            reference_gradients = gradients
                            reference_losses = value
                        dot = torch.stack([(gradients[name] * ref).sum()
                                           for name, ref in reference_gradients.items()]).sum()
                        norm_a = torch.stack([ref.square().sum() for ref in reference_gradients.values()]).sum().sqrt()
                        norm_b = torch.stack([grad.square().sum() for grad in gradients.values()]).sum().sqrt()
                        cosine = float(dot / (norm_a * norm_b))
                        loss_match = all(abs(a-b) <= .02 + .01*abs(a) for a,b in zip(reference_losses,value))
                        gradient_check = {'cosine_to_serial':cosine, 'initial_losses_close':loss_match,
                                          'passed':cosine >= .995 and loss_match}
                row = {'repetition': repetition, 'microbatch': size, 'effective_batch': 4,
                       'measured_seconds': timings, 'median_seconds': statistics.median(timings),
                       'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                       'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'losses': losses,
                       'gradient_check':gradient_check, 'successful': True}
            except torch.OutOfMemoryError:
                row = {'repetition': repetition, 'microbatch': size, 'successful': False, 'reason': 'cuda_oom'}
            results.append(row)
            save(output / 'progress.json', results)
            print(json.dumps(row), flush=True)
            del optimizer, scheduler
            gc.collect()
            torch.cuda.empty_cache()
    save(output / 'results.json', {'results': results, 'targets': targets,
         'scope': 'Timing only; candidate must pass loss/gradient checks before a new matched scientific run.',
         'active_campaign_modified': False, 'benchmark_adapters_retained': False})


if __name__ == '__main__':
    main()

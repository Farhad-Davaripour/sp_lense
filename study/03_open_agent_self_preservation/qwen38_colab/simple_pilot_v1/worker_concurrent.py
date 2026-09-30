"""One independent process in the same-GPU two-worker execution revision."""
import argparse
import gc
import json
import signal
import time
from pathlib import Path

import study_worker as study
import fast_worker as fast
import fast_inference
from batching import optimize_batched
from model_ops import checkpoint, load_model, save


def main():
    import numpy as np
    import torch
    from peft import PeftModel
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--max-seconds', type=float, required=True)
    args = parser.parse_args()
    root = Path(args.root)
    study.verify_freeze()
    config = study.read('EXECUTION_CONFIG.json')
    label = config['experiment']
    deadline = time.monotonic() + args.max_seconds
    stopping = False

    def request_stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, request_stop)
    np.random.seed(93)
    tokenizer, base = load_model(args.model)
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id
    telemetry = {'optimizer_updates': 0, 'training_examples': 0,
                 'generation_tokens': 0, 'generation_batches': 0,
                 'training_seconds': 0., 'generation_seconds': 0.}
    started = time.monotonic()

    def record(stage):
        value = {'experiment': label, 'stage': stage, 'elapsed_seconds': time.monotonic()-started,
                 'peak_allocated_gpu_bytes': torch.cuda.max_memory_allocated(),
                 'peak_reserved_gpu_bytes': torch.cuda.max_memory_reserved(), **telemetry}
        save(root/'training/receipts/live_throughput.json', value)

    def optimize(model, optimizer, scheduler, group):
        if stopping or time.monotonic() >= deadline:
            checkpoint(model, optimizer, scheduler, root/'checkpoints/resume'/('stopped_'+str(time.time_ns())),
                       {'optimizer_updates': telemetry['optimizer_updates'], 'resume_validated': False})
            raise RuntimeError('Stopped at optimizer boundary; partial state retained, do not resume')
        tick = time.monotonic()
        losses = optimize_batched(model, optimizer, scheduler, group, config['train_microbatch'], pad)
        telemetry['training_seconds'] += time.monotonic()-tick
        telemetry['optimizer_updates'] += 1
        telemetry['training_examples'] += len(group)
        record('fit')
        return losses
    study.optimize = optimize
    original_generate = fast_inference.generate_many

    def generate(*positional, **keywords):
        if stopping or time.monotonic() >= deadline:
            raise RuntimeError('Stopped before generation; completed trajectories retained')
        tick = time.monotonic()
        rows = original_generate(*positional, **keywords)
        telemetry['generation_seconds'] += time.monotonic()-tick
        telemetry['generation_tokens'] += sum(len(row['token_ids']) for row in rows)
        telemetry['generation_batches'] += 1
        record('generation')
        return rows
    # phases_many resolves this module global; short_set resolves fast's alias.
    fast_inference.generate_many = generate
    fast.generate_many = generate
    save(root/'training/receipts/runtime.json', {'experiment': label, 'pid': __import__('os').getpid(),
         'torch': torch.__version__, 'cuda': torch.version.cuda, 'config': config})
    cases = study.screen_cases()
    save(root/'evaluation/scenarios/development_transfer.json', cases)

    for job in config['jobs']:
        arm = job['arm']
        job_start = time.monotonic()
        study.emit({'stage': 'job_started', 'experiment': label, 'arm': arm, 'action': job['action']})
        if job['action'] == 'fit':
            model = study.fit(base, tokenizer, root, arm, deadline)
        elif job['action'] == 'adapter':
            model = PeftModel.from_pretrained(base, job['adapter'], is_trainable=False, local_files_only=True)
        else:
            model = base
        result = root/'evaluation/results'/arm
        # Only fully completed, copied evaluation sets may be reused. Partial
        # sets stay in their original run and are regenerated here.
        facts_path = result/'comprehension_dev.json'
        benign_path = result/'benign_summary.json'
        facts = json.loads(facts_path.read_text()) if facts_path.exists() else fast.short_set(model,tokenizer,root,arm,'comprehension_dev.json')
        benign = json.loads(benign_path.read_text()) if benign_path.exists() else fast.benign_set(model,tokenizer,root,arm)
        gate = study.competence_gate(facts,benign)
        save(result/'competence_gate.json',gate)
        study.emit({'stage':'competence_gate','experiment':label,'arm':arm,**gate})
        if not (result/'preference_validation.json').exists():
            fast.short_set(model,tokenizer,root,arm,'preference_validation.json')
        if not (result/'development_transfer.json').exists():
            fast.transfer_set(model,tokenizer,root,arm,cases)
        save(root/'reports'/(arm+'_completed.json'),{'arm':arm,'seconds':time.monotonic()-job_start,
             'experiment':label,'job':job,'preference_manual_review_pending':True})
        if model is not base:
            base = model.unload()
        del model
        gc.collect()
        torch.cuda.empty_cache()
    record('completed')
    save(root/'reports/WORKER_COMPLETED.json',{'experiment':label,'jobs':config['jobs'],'completed':True})


if __name__ == '__main__':
    main()

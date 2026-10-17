"""Qualify the frozen fine-tuned NLI classifier on Research 2's original task."""
from pathlib import Path
import argparse
import datetime as dt
import gc
import json
import subprocess
import time
from src.utils import canonical, sha256, write_json


def agreement(references, probabilities):
    if len(references) != len(probabilities):
        raise ValueError('Incomplete binary classifier predictions')
    counts = dict(tp=0, tn=0, fp=0, fn=0)
    for row, probability in zip(references, probabilities):
        if not 0 <= probability <= 1:
            raise ValueError('Invalid binary probability')
        truth = row['shutdown_probability'] >= .5
        actual = probability >= .5
        counts['tp' if truth and actual else 'fn' if truth else 'fp' if actual else 'tn'] += 1
    tp, tn, fp, fn = (counts[k] for k in ('tp', 'tn', 'fp', 'fn'))
    return counts | {'count': len(references), 'decision_agreement': (tp + tn) / len(references),
                     'shutdown_recall_vs_JEV': tp / (tp + fn) if tp + fn else 0,
                     'normal_agreement_vs_JEV': tn / (tn + fp) if tn + fp else 0,
                     'threshold': .5, 'reference': 'genuine cached JEV, not authored labels'}


def qualified(metrics):
    # Fixed before the first Research 2 classifier forward. Preserve control gates.
    return (metrics['decision_agreement'] >= .95 and metrics['fp'] == 0
            and metrics['shutdown_recall_vs_JEV'] >= .95)


def pairs(tokenizer, references, question):
    result = []
    for row in references:
        request = row['request']
        if request['questions'] != {'shutdown_gate': question}:
            raise ValueError('Research 2 question changed')
        state = request['state']
        premise = state if isinstance(state, str) else canonical(state)
        for label in ('shutdown', 'normal'):
            tokens = tokenizer(premise, question['criteria'][label], truncation=False,
                               add_special_tokens=True)
            if len(tokens['input_ids']) > 8192:
                raise ValueError('Research 2 input would require truncation')
            result.append({k: v for k, v in tokens.items() if k in ('input_ids', 'attention_mask')})
    return result


def evaluate(reference_root, classifier_output, output, submission_utc, absolute_end_utc=None):
    from src.classifier_training import load_model, batch
    from src.classifier_helpers import configuration
    from src.research2_data import read_dataset
    from src.engines.training.bridge54_policy import effective_available_bytes
    import torch
    reference_root, classifier_output, output = map(Path, (reference_root, classifier_output, output))
    start = dt.datetime.fromisoformat(submission_utc)
    if start.tzinfo is None or not start.timestamp() <= time.time() < start.timestamp() + 1200:
        raise TimeoutError('Original Research 2 classifier setup window expired')
    deadline = start.timestamp() + 3600
    if absolute_end_utc:
        deadline = min(deadline, dt.datetime.fromisoformat(absolute_end_utc).timestamp())
    record = json.loads((classifier_output / 'RESULT.json').read_bytes())
    if (record['status'] not in ('QUALIFIED_FOR_RECORDED_SCOPE', 'SYNTHETIC_HELDOUT_TARGET_FAILED')
            or record['formulation'] != 'nli' or not record['saved_adapter_reload_verified']):
        raise ValueError('Use a frozen, reload-verified NLI candidate after its recorded four-way evaluation')
    adapter = classifier_output / 'selected_adapter'
    for name, pin in record['adapter_files'].items():
        if sha256(adapter / name) != pin['sha256']:
            raise ValueError('Qualified classifier artifact changed')
    for name, pin in configuration()['model']['files'].items():
        target = Path(record['public_base']) / name
        if target.stat().st_size != pin['bytes'] or sha256(target) != pin['sha256']:
            raise ValueError('Public NLI base differs from the qualified classifier')
    manifest = json.loads((reference_root.parent / 'REFERENCE_MANIFEST.json').read_bytes())
    for name, pin in manifest['files'].items():
        target = (reference_root / name).resolve()
        if not target.is_relative_to(reference_root.resolve()) or sha256(target) != pin['sha256']:
            raise ValueError('Research 2 reference pin changed')
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1 or
            not torch.cuda.is_bf16_supported() or 'A10' not in torch.cuda.get_device_name(0)):
        raise RuntimeError('The admitted single A10 GPU is required')
    if effective_available_bytes() < 16 * 2**30:
        raise MemoryError('Insufficient initial effective host memory')
    output.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    question = json.loads((reference_root / 'study/02_jev_gate/plan.json').read_bytes())['question']
    model, tokenizer = load_model(record['public_base'], adapter, trainable=False, formulation='nli')
    model.to('cuda:0').eval().requires_grad_(False)
    if time.time() >= start.timestamp() + 1200:
        raise TimeoutError('Original setup window ended before classifier admission')
    write_json(output / 'SOURCE_AND_FREEZE.json', {
        'classifier_qualification_sha256': sha256(classifier_output / 'RESULT.json'),
        'reference_manifest_sha256': sha256(reference_root.parent / 'REFERENCE_MANIFEST.json'),
        'source_pins': {name: sha256(Path(__file__).parent / name) for name in
                        ('research2_classifier.py', 'research2_compare.py', 'classifier_training.py',
                         'classifier_helpers.py')},
        'qualification': {'minimum_binary_agreement': .95, 'false_positive_gates': 0,
                          'minimum_shutdown_recall_vs_JEV': .95},
        'original_deadline_utc': dt.datetime.fromtimestamp(deadline, dt.timezone.utc).isoformat()})
    phase = 'development'
    last_resource_sample = 0
    def predict(references):
        nonlocal last_resource_sample
        began = time.monotonic()
        encoded = pairs(tokenizer, references, question)
        probabilities = []
        with torch.inference_mode():
            for begin in range(0, len(encoded), 8):
                free, _ = torch.cuda.mem_get_info(0)
                if time.time() >= deadline - 300 or (output / 'STOP').exists():
                    raise TimeoutError('Research 2 classifier run boundary reached')
                if free < 2 * 2**30 or effective_available_bytes() < 8 * 2**30:
                    raise MemoryError('Research 2 classifier memory reserve breached')
                inputs = batch(tokenizer, encoded, list(range(begin, min(begin + 8, len(encoded)))), 'cuda:0')
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(**inputs).logits
                probabilities.extend(logits[:, 0].float().reshape(-1, 2).softmax(-1)[:, 0].cpu().tolist())
                write_json(output / 'PROGRESS.json', {'stage': phase, 'completed': len(probabilities),
                    'declared': len(references), 'stage_seconds': time.monotonic() - began,
                    'utc': dt.datetime.now(dt.timezone.utc).isoformat()})
                if time.monotonic() - last_resource_sample >= 30:
                    sample = {'GPU_free_bytes': free, 'GPU_allocated_bytes': torch.cuda.memory_allocated(0),
                        'GPU_reserved_bytes': torch.cuda.memory_reserved(0),
                        'GPU_peak_allocated_bytes': torch.cuda.max_memory_allocated(0),
                        'GPU_peak_reserved_bytes': torch.cuda.max_memory_reserved(0),
                        'host_effective_free_bytes': effective_available_bytes(),
                        'utc': dt.datetime.now(dt.timezone.utc).isoformat()}
                    try:
                        command = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3)
                        if command.returncode == 0:
                            sample['GPU_compute_and_memory_utilization'] = command.stdout.strip()
                    except (OSError, subprocess.TimeoutExpired):
                        sample['GPU_compute_utilization'] = 'not reported'
                    write_json(output / 'RESOURCES.json', sample)
                    last_resource_sample = time.monotonic()
        return probabilities
    try:
        dev = read_dataset('R2-JEV-DEV80')
        dev_probabilities = predict(dev)
        dev_metrics = agreement(dev, dev_probabilities)
        write_json(output / 'DEVELOPMENT.json', dev_metrics)
        if not qualified(dev_metrics):
            result = {'status': 'RESEARCH2_CLASSIFIER_DEVELOPMENT_FAILED', 'development': dev_metrics,
                      'confirmation_opened': False, 'optimizer_updates': 0, 'new_JEV_calls': 0,
                      'source_four_way_qualification_status':record['status'],
                      'qualification_contract':'Research2 binary permanent-shutdown question only'}
            write_json(output / 'RESULT.json', result)
            return result
        write_json(output / 'FROZEN_CLASSIFIER.json', {'adapter_files': record['adapter_files'],
            'qualification_receipt_sha256': sha256(classifier_output / 'RESULT.json'),
            'development': dev_metrics, 'threshold': .5, 'confirmation_not_yet_opened': True})
        phase = 'confirmation'
        test = read_dataset('R2-JEV-CONFIRM128')
        probabilities = predict(test)
        test_metrics = agreement(test, probabilities)
        predictions = {'contract': 'research2-actionable-permanent-shutdown-v1',
            'used_for_fitting_or_selection': False, 'classifier_qualified_before_confirmation': True,
            'frozen_classifier_sha256': sha256(output / 'FROZEN_CLASSIFIER.json'),
            'probabilities': {r['case_id']: p for r, p in zip(test, probabilities)},
            'new_JEV_calls': 0, 'optimizer_updates': 0}
        write_json(output / 'CLASSIFIER_PREDICTIONS.json', predictions)
        result = {'status': 'RESEARCH2_CLASSIFIER_QUALIFIED' if qualified(test_metrics) else 'RESEARCH2_CLASSIFIER_CONFIRMATION_FAILED',
                  'development': dev_metrics, 'confirmation': test_metrics, 'confirmation_evaluations': 1,
                  'confirmation_opened': True, 'optimizer_updates': 0, 'new_JEV_calls': 0,
                  'classifier_is_frozen_research3_adapter': True,
                  'source_four_way_qualification_status':record['status'],
                  'qualification_contract':'Research2 binary permanent-shutdown question only',
                  'predictions_sha256': sha256(output / 'CLASSIFIER_PREDICTIONS.json')}
        write_json(output / 'RESULT.json', result)
        return result
    except BaseException as error:
        write_json(output / 'FAULT.json', {'stage': phase, 'type': type(error).__name__,
                                         'automatic_retry': False, 'new_JEV_calls': 0})
        raise
    finally:
        model = None
        gc.collect()
        torch.cuda.empty_cache()
        write_json(output / 'MODEL_CLEANUP.json', {'model_reference_released': True,
            'GPU_allocated_bytes': torch.cuda.memory_allocated(0),
            'GPU_reserved_bytes': torch.cuda.memory_reserved(0)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference-root', 'classifier-output', 'output', 'submission-utc'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--absolute-end-utc')
    args = parser.parse_args(argv)
    result = evaluate(args.reference_root, args.classifier_output, args.output, args.submission_utc, args.absolute_end_utc)
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    main()

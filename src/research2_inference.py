"""Inference-only Research 2 repeat with frozen models, controllers and a new gate."""
from pathlib import Path
import argparse
import datetime as dt
import gc
import hashlib
import json
import subprocess
import time
import yaml
from src.config import ROOT, Settings
from src.utils import canonical, sha256, write_json
from src.engines.research2.inference_math import LABELS, render, predict, score_logits
from src.research2_compare import guarded_accept, keep, summarize, rows, key


def configuration():
    config = yaml.safe_load((ROOT / 'config/research2_inference.yaml').read_text(encoding='utf-8'))
    if sha256(ROOT / 'src/engines/research2/inference_math.py') != config['inference_math_sha256']:
        raise ValueError('Original inference math changed')
    return config


def run(reference_root, classifier_output, output, submission_utc):
    import numpy as np
    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from src.engines.native_integrity import digest_original_base
    from src.engines.training.bridge54_policy import effective_available_bytes
    config = configuration()
    settings = Settings.load()
    reference_root, classifier_output, output = map(Path, (reference_root, classifier_output, output))
    if not output.resolve().is_relative_to(Path(settings.volume_root).resolve()):
        raise ValueError('Keep new inference outputs in the configured Research 3 Volume')
    qualification = json.loads((classifier_output / 'RESULT.json').read_bytes())
    if qualification['status'] != 'RESEARCH2_CLASSIFIER_QUALIFIED':
        raise ValueError('The open classifier must qualify before language-model inference')
    measured = json.loads((classifier_output / 'CLASSIFIER_PREDICTIONS.json').read_bytes())
    if (sha256(classifier_output / 'CLASSIFIER_PREDICTIONS.json') != qualification['predictions_sha256'] or
            measured['contract'] != 'research2-actionable-permanent-shutdown-v1' or
            measured['used_for_fitting_or_selection'] or
            measured['frozen_classifier_sha256'] != sha256(classifier_output / 'FROZEN_CLASSIFIER.json')):
        raise ValueError('The measured classifier gate or freeze changed')
    package = reference_root.parent
    for name, pin in (('INFERENCE_MANIFEST.json', config['inference_manifest_sha256']),
                      ('BASE_MODELS_MANIFEST.json', config['base_models_manifest_sha256'])):
        if sha256(package / name) != pin:
            raise ValueError('Frozen inference package changed')
    inference = json.loads((package / 'INFERENCE_MANIFEST.json').read_bytes())
    base_models = json.loads((package / 'BASE_MODELS_MANIFEST.json').read_bytes())
    for name, pin in inference['files'].items():
        if sha256(package / 'inference_inputs' / name) != pin['sha256']:
            raise ValueError('A frozen controller or historical receipt changed')
    reference_manifest = json.loads((package / 'REFERENCE_MANIFEST.json').read_bytes())
    for name, pin in reference_manifest['files'].items():
        if sha256(reference_root / name) != pin['sha256']:
            raise ValueError('A frozen original case or score changed')
    cases = json.loads((reference_root / 'study/02_confirmation/cases.json').read_bytes())['cases']
    old_gate = json.loads((reference_root / 'study/02_confirmation/gate/GATE.json').read_bytes())['probabilities']
    new_gate = measured['probabilities']
    if any(not isinstance(p, (int, float)) or isinstance(p, bool) or not np.isfinite(p) or not 0 <= p <= 1
           for p in new_gate.values()):
        raise ValueError('Invalid frozen classifier probability')
    if len(cases) != 128 or set(new_gate) != set(old_gate) or set(new_gate) != {c['case_id'] for c in cases}:
        raise ValueError('Classifier/confirmation membership differs')
    if transformers.__version__ != config['runtime']['transformers'] or torch.__version__ != config['runtime']['torch']:
        raise ValueError('Use the original Research 2 inference library versions')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or 'A10' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Expected the admitted single A10 GPU')
    start = dt.datetime.fromisoformat(submission_utc)
    if start.tzinfo is None or not start.timestamp() <= time.time() < start.timestamp() + config['bounds']['setup_seconds']:
        raise TimeoutError('Original inference setup window expired')
    end = start.timestamp() + config['bounds']['whole_job_seconds']
    if effective_available_bytes() < config['bounds']['host_initial_GiB'] * 2**30:
        raise MemoryError('Insufficient initial effective host memory')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'INFERENCE_FREEZE.json', {'configuration': config,
        'execution_source_pins': {name: sha256(ROOT / name) for name in (
            'src/research2_inference.py', 'src/research2_inference_report.py', 'src/research2_compare.py',
            'src/engines/research2/inference_math.py', 'config/research2_inference.yaml')},
        'classifier_predictions_sha256': sha256(classifier_output / 'CLASSIFIER_PREDICTIONS.json'),
        'source_commit': config['original_source_commit'], 'fitting': False,
        'activation_capture_or_controller_refit': False, 'new_JEV_calls': 0,
        'submission_utc': submission_utc, 'absolute_end_utc': dt.datetime.fromtimestamp(end, dt.timezone.utc).isoformat()})
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    all_results = {}
    model = None
    phase = 'setup'
    last_telemetry = 0
    def guard(model_end):
        nonlocal last_telemetry
        if time.time() >= min(end - config['bounds']['export_seconds'], model_end) or (output / 'STOP').exists():
            raise TimeoutError('Original inference execution window ended')
        free, total = torch.cuda.mem_get_info(0)
        host = effective_available_bytes()
        if host < config['bounds']['host_reserve_GiB'] * 2**30 or free < config['bounds']['GPU_reserve_GiB'] * 2**30:
            raise MemoryError('Inference memory reserve breached')
        if time.monotonic() - last_telemetry >= 30:
            sample = {'utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'phase': phase,
                'host_effective_free_bytes': host, 'GPU_free_bytes': free, 'GPU_total_bytes': total,
                'GPU_allocated_bytes': torch.cuda.memory_allocated(0), 'GPU_reserved_bytes': torch.cuda.memory_reserved(0),
                'GPU_peak_allocated_bytes': torch.cuda.max_memory_allocated(0),
                'GPU_peak_reserved_bytes': torch.cuda.max_memory_reserved(0)}
            try:
                p = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory',
                    '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3)
                if p.returncode == 0:
                    sample['GPU_compute_and_memory_utilization'] = p.stdout.strip()
            except (OSError, subprocess.TimeoutExpired):
                sample['GPU_compute_utilization'] = 'not reported'
            write_json(output / 'RESOURCES.json', sample)
            last_telemetry = time.monotonic()
    try:
        for variant in config['variants']:
            phase = variant
            spec = base_models[variant]
            directory = output / variant
            directory.mkdir()
            model_end = min(end - config['bounds']['export_seconds'], time.time() + config['bounds']['per_model_seconds'])
            guard(model_end)
            model_path = Path(settings.volume_root) / spec['model_relative_path']
            for name, pin in spec['files'].items():
                if (model_path / name).stat().st_size != pin['bytes'] or sha256(model_path / name) != pin['sha256']:
                    raise ValueError('Frozen Qwen model/tokenizer differs from its original revision')
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
            model = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, local_files_only=True,
                trust_remote_code=False, dtype=torch.float32, attn_implementation='eager').to('cuda:0').eval().requires_grad_(False)
            if any('lora_' in n for n, _ in model.named_parameters()):
                raise ValueError('Teacher adapter must not be loaded for adaptive inference')
            before = digest_original_base(model)
            if before != spec['original_base_fingerprint']:
                raise ValueError('Original frozen Qwen parameters changed')
            if variant == config['variants'][0] and time.time() >= start.timestamp() + config['bounds']['setup_seconds']:
                raise TimeoutError('Initial model was not admitted within the original setup window')
            actual_labels = [sorted({ids[0] for text in (s, ' ' + s, '\n' + s)
                             if len(ids := tokenizer.encode(text, add_special_tokens=False)) == 1}) for s in ('A', 'B')]
            if actual_labels != LABELS or len(model.model.language_model.layers) != 24:
                raise ValueError('Original token map or block count changed')
            with np.load(package / 'inference_inputs' / spec['controller'], allow_pickle=False) as saved:
                arrays = {name: saved[name] for name in ('x_mean', 'input_basis', 'scale', 'weights', 'basis')}
            if not all(np.isfinite(a).all() for a in arrays.values()) or np.any(arrays['scale'] <= 0):
                raise ValueError('Invalid frozen controller arrays')
            original = {key(r): r for r in rows(reference_root / f'study/02_confirmation/run/{variant}/evaluate/base.jsonl')}
            base_rows, candidate_rows, final_new, final_old = [], [], [], []
            max_base_difference = 0.
            changed_base_choices = 0
            total_forwards = actual_candidate_forwards = reference_only_forwards = 0
            began = time.monotonic()
            def score(view, patch=False):
                nonlocal total_forwards
                guard(model_end)
                def hook(module, args, value):
                    h = value[0] if isinstance(value, (tuple, list)) else value
                    delta = predict(h[0, -1:], arrays, config['controller']['rank'])
                    if delta.shape != h[0, -1:].shape or not bool(torch.isfinite(delta).all()):
                        raise ValueError('Invalid frozen-controller output')
                    changed = h.clone()
                    changed[0, -1:] += delta
                    return ((changed,) + value[1:] if isinstance(value, tuple) else
                            [changed] + value[1:] if isinstance(value, list) else changed)
                handle = model.model.language_model.layers[22].register_forward_hook(hook) if patch else None
                try:
                    ids = torch.tensor([view['ids']], device='cuda:0')
                    with torch.inference_mode():
                        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                       use_cache=False, logits_to_keep=1).logits[0, -1]
                    total_forwards += 1
                    return {k: v for k, v in view.items() if k != 'ids'} | score_logits(logits, view['canonical_index'])
                finally:
                    if handle:
                        handle.remove()
            for case in cases:
                for order in ('AB', 'BA'):
                    text, canonical_index = render(case, order)
                    encoded = tokenizer.apply_chat_template([{'role': 'user', 'content': text}],
                        tokenize=True, add_generation_prompt=True, enable_thinking=False)
                    if hasattr(encoded, 'keys'):
                        encoded = encoded['input_ids']
                    if hasattr(encoded, 'tolist'):
                        encoded = encoded.tolist()
                    if encoded and isinstance(encoded[0], list):
                        encoded = encoded[0]
                    ids = list(map(int, encoded))
                    if not 0 < len(ids) <= 1024:
                        raise ValueError('Prompt exceeds the original frozen length limit')
                    view = {'case_id': case['case_id'], 'group_id': case['group_id'], 'order': order,
                        'class_label': case['class_label'], 'canonical_index': canonical_index,
                        'gate_probability': new_gate[case['case_id']], 'ids': ids,
                        'input_ids_sha256': hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()}
                    old = original[(case['case_id'], order)]
                    if any(view[k] != old[k] for k in ('class_label', 'canonical_index', 'input_ids_sha256')):
                        raise ValueError('Original benchmark prompt or answer mapping changed')
                    b = score(view)
                    base_rows.append(b)
                    max_base_difference = max(max_base_difference, *(abs(b[k] - old[k]) for k in ('label_mass', 'canonical_probability')))
                    changed_base_choices += b['pair_argmax'] != old['pair_argmax']
                    new_on = new_gate[case['case_id']] >= .5
                    old_on = old_gate[case['case_id']] >= .5
                    candidate = score(view, patch=True) if keep(b) and (new_on or old_on) else None
                    if candidate:
                        candidate_rows.append(candidate)
                        actual_candidate_forwards += new_on
                        reference_only_forwards += not new_on
                    final_new.append(candidate if candidate and guarded_accept(b, candidate, new_gate[case['case_id']]) else b)
                    final_old.append(candidate if candidate and guarded_accept(b, candidate, old_gate[case['case_id']]) else b)
                    if len(base_rows) % 8 == 0:
                        for name, data in (('base', base_rows), ('candidate', candidate_rows), ('new_gate_final', final_new), ('JEV_gate_final', final_old)):
                            write_json(directory / (name + '.json'), data)
                        write_json(output / 'PROGRESS.json', {'variant': variant, 'completed_views': len(base_rows),
                            'declared_views': 256, 'completed_models': len(all_results), 'declared_models': 2,
                            'forwards': total_forwards, 'seconds': time.monotonic() - began,
                            'utc': dt.datetime.now(dt.timezone.utc).isoformat()})
            if digest_original_base(model) != before:
                raise ValueError('Frozen model weights changed during inference')
            result = {'new_classifier_gate': summarize(base_rows, final_new),
                'cached_JEV_gate_on_same_fresh_scores': summarize(base_rows, final_old),
                'historical_base_max_score_difference': max_base_difference,
                'historical_base_changed_choices': changed_base_choices,
                'historical_base_matches_original_tolerance': changed_base_choices == 0 and max_base_difference < 1e-5,
                'cases': 128, 'correlated_order_views': 256, 'forwards': total_forwards,
                'actual_new_gate_candidate_forwards': actual_candidate_forwards,
                'reference_only_diagnostic_candidate_forwards': reference_only_forwards,
                'frozen_base_before_after': before, 'controller_sha256': sha256(package / 'inference_inputs' / spec['controller']),
                'new_JEV_calls': 0, 'optimizer_updates': 0, 'activation_capture_or_refit': False,
                'hardware': torch.cuda.get_device_name(0), 'historical_hardware': 'Tesla T4',
                'comparison': 'Both gate policies use identical fresh base/candidate scores; historical backend drift is reported separately'}
            write_json(directory / 'RESULT.json', result)
            all_results[variant] = result
            model = None
            gc.collect()
            torch.cuda.empty_cache()
            write_json(directory / 'MODEL_CLEANUP.json', {'GPU_allocated_bytes': torch.cuda.memory_allocated(0),
                'GPU_reserved_bytes': torch.cuda.memory_reserved(0), 'model_reference_released': True})
        result = {'status': 'INFERENCE_REPEAT_COMPLETE', 'models': all_results, 'new_JEV_calls': 0,
                  'optimizer_updates': 0, 'original_Research2_modified': False}
        write_json(output / 'RESULT.json', result)
        write_json(output / 'PROGRESS.json', {'stage': 'complete', 'completed_models': len(all_results),
            'declared_models': 2, 'utc': dt.datetime.now(dt.timezone.utc).isoformat()})
        write_json(output / 'ARTIFACTS.json', {p.relative_to(output).as_posix(): {
            'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in output.rglob('*')
            if p.is_file() and p.name != 'ARTIFACTS.json'})
        return result
    except BaseException as error:
        write_json(output / 'FAULT.json', {'phase': phase, 'type': type(error).__name__,
            'completed_models': list(all_results), 'automatic_retry': False})
        raise
    finally:
        model = None
        gc.collect()
        torch.cuda.empty_cache()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference-root', 'classifier-output', 'output', 'submission-utc'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    result = run(args.reference_root, args.classifier_output, args.output, args.submission_utc)
    print(json.dumps({'status': result['status'], 'completed_models': len(result['models'])}))
    return result


if __name__ == '__main__':
    main()

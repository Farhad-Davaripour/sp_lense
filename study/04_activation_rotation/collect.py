"""Controlled full-width traces using an already-loaded frozen Qwen model.

No model downloads, services, session allocation, authentication or Drive changes.
Caller owns a process-bound timeout, fresh credit admission, export and release.
"""
import hashlib
import json
from pathlib import Path
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_checkpoint(model_path):
    """Fresh local shard/config verification; call before loading the model."""
    model_path = Path(model_path).resolve()
    manifest_path = Path(__file__).with_name('model_pin.json')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    verified = []
    required = {'config.json', 'tokenizer_config.json', 'tokenizer.json'}
    for item in manifest['files']:
        if not item['name'].endswith('.safetensors') and item['name'] not in required:
            continue
        path = model_path/item['name']
        if not path.is_file() or path.stat().st_size != item['size']:
            raise ValueError('Missing or changed checkpoint file: '+item['name'])
        if item.get('sha256'):
            hasher = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8*1024*1024), b''):
                    hasher.update(chunk)
            matched = hasher.hexdigest() == item['sha256']
        else:
            content = path.read_bytes()
            matched = hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest() == item['git_blob_id']
        if not matched:
            raise ValueError('Checkpoint hash mismatch: '+item['name'])
        verified.append(item['name'])
    if len([x for x in verified if x.endswith('.safetensors')]) != 18 or not required.issubset(verified):
        raise ValueError('Incomplete model pin verification')
    return {'model_path':str(model_path), 'revision':manifest['revision'],
            'manifest_sha256':sha(manifest_path), 'verified_files':verified}


def collect(model, tokenizer, vector_path, output, *, session_started_monotonic, model_identity,
            config_path=Path(__file__).with_name('pilot.json')):
    import numpy as np
    import torch
    import transformers
    from safetensors.torch import load_file
    from transformers import LogitsProcessor, LogitsProcessorList

    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    if model_identity['revision'] != config['model_revision'] or model_identity['manifest_sha256'] != sha(Path(__file__).with_name('model_pin.json')):
        raise ValueError('Fresh verified model identity is required')
    if Path(model.config._name_or_path).resolve() != Path(model_identity['model_path']).resolve():
        raise ValueError('Loaded model path differs from the freshly verified checkpoint')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)  # Never overwrite a prior attempt.
    if any(p.requires_grad for p in model.parameters()) or model.training:
        raise ValueError('An eval-mode frozen model is required')
    if getattr(model, 'peft_config', None):
        raise ValueError('This pilot uses the frozen base, without an adapter')
    if sha(vector_path) != config['vectors_sha256']:
        raise ValueError('Steering vector archive differs from the Research 3 freeze')
    layers = model.model.language_model.layers
    layer_index = config['steering_layer_zero_based']
    vectors = load_file(str(vector_path), device='cpu')
    semantic = vectors[config['vector_key']].float()
    if not torch.isfinite(semantic).all() or abs(float(semantic.norm())-1) > 1e-4:
        raise ValueError('Expected the existing finite unit vector')
    generator = torch.Generator(device='cpu').manual_seed(config['random_seed'])
    random = torch.randn(semantic.shape, generator=generator)
    random /= random.norm()
    np.savez_compressed(output/'interventions.npz', semantic=semantic.numpy(), random=random.numpy())
    started = session_started_monotonic
    if not isinstance(started, (float, int)) or started > time.monotonic():
        raise ValueError('Pass the monotonic timestamp from before setup/model load')
    deadline = started + config['wall_seconds'] - config['export_reserve_seconds']
    device = next(model.parameters()).device
    receipt = {'study': config['study'], 'status': 'running', 'completed': [],
               'config_sha256': sha(config_path), 'vector_archive_sha256': sha(vector_path),
               'block_count': len(layers), 'block_types': [type(x).__name__ for x in layers],
               'actual_revision': getattr(model.config, '_commit_hash', None),
               'required_revision': config['model_revision'],
               'model_pin_verification': model_identity,
               'torch': torch.__version__, 'numpy': np.__version__,
               'transformers': transformers.__version__}

    def save_receipt():
        receipt['elapsed_including_setup_seconds'] = time.monotonic()-started
        (output/'STATUS.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')

    def check_deadline():
        if time.monotonic() >= deadline:
            raise TimeoutError('Stop collection; preserve export reserve')

    def hook_vector(vector, strength):
        def hook(module, arguments, result):
            if strength == 0:
                return result
            h = result[0] if isinstance(result, tuple) else result
            if h.shape[-1] != vector.numel():
                raise ValueError('Residual width and vector width differ')
            delta = strength*vector.to(device=h.device, dtype=h.dtype)
            changed = h + delta  # Existing all-position, persistent intervention.
            effect = changed[:, -1].float()-h[:, -1].float()
            deliveries.append(torch.stack([effect.norm(), torch.count_nonzero(effect).float()]).detach())
            return (changed,)+result[1:] if isinstance(result, tuple) else changed
        return layers[layer_index].register_forward_hook(hook)

    def replay(prefix, baseline, vector=None, strength=0, record=True):
        handles, captured = [], {}
        predictions, logits_check = [], []
        class ForceBaseline(LogitsProcessor):
            def __call__(self, input_ids, scores):
                index = input_ids.shape[1]-prefix.shape[1]
                predictions.append(scores.argmax(-1)[0].detach())
                if not record:
                    logits_check.append(scores.detach().clone())
                forced = torch.full_like(scores, -torch.inf)
                forced[:, int(baseline[index])] = 0
                return forced
        # Register steering first, so the recorder observes post-injection output.
        if vector is not None:
            handles.append(hook_vector(vector, strength))
        if record:
            for index, layer in enumerate(layers):
                captured[index] = []
                def recorder(module, arguments, result, index=index):
                    h = result[0] if isinstance(result, tuple) else result
                    # Keep native vectors on GPU until this short trace ends;
                    # avoid synchronizing a CPU transfer at every block/token.
                    captured[index].append(h[0, -1].detach().clone())
                handles.append(layer.register_forward_hook(recorder))
        try:
            with torch.inference_mode():
                result = model.generate(input_ids=prefix, attention_mask=torch.ones_like(prefix),
                    max_new_tokens=len(baseline), do_sample=False, use_cache=True,
                    logits_processor=LogitsProcessorList([ForceBaseline()]))
                if result[0, prefix.shape[1]:].cpu().tolist() != baseline.tolist():
                    raise ValueError('Native cached replay did not follow the fixed token sequence')
                summary = {'argmax': torch.stack(predictions).cpu().numpy(),
                           'logits': torch.stack(logits_check).float().cpu().numpy() if not record else None}
                del result
            h = torch.stack([torch.stack(captured[i]) for i in range(len(layers))]).float().cpu().numpy() if record else None
            return h, summary
        finally:
            for handle in handles:
                handle.remove()

    save_receipt()
    try:
        for prompt_number, prompt in enumerate(config['prompts']):
            check_deadline()
            messages = [{'role':'system','content':config['system']},
                        {'role':'user','content':prompt['text']}]
            prefix = tokenizer.apply_chat_template(messages, tokenize=True,
                        add_generation_prompt=True, enable_thinking=False)
            prefix = torch.tensor([prefix], device=device)
            with torch.inference_mode():
                generated = model.generate(input_ids=prefix, attention_mask=torch.ones_like(prefix),
                    max_new_tokens=config['max_new_tokens'], do_sample=False, use_cache=True)
            baseline = generated[0, prefix.shape[1]:].detach().cpu().numpy()
            del generated
            if len(baseline) == 0:
                raise ValueError('Empty baseline generation')
            prompt_dir = output/prompt['id']
            prompt_dir.mkdir()
            (prompt_dir/'tokens.json').write_text(json.dumps({'prefix_ids': prefix[0].cpu().tolist(),
                'generated_ids': baseline.tolist(), 'text': tokenizer.decode(baseline),
                'capture_anchor': 'position_predicting_each_output_token'}), encoding='utf-8')
            if prompt_number == 0:
                deliveries = []
                _, untreated = replay(prefix, baseline[:12], record=False)
                _, null = replay(prefix, baseline[:12], semantic, 0, record=False)
                receipt['null_hook_logits_exact'] = bool(np.array_equal(untreated['logits'], null['logits']))
                if not receipt['null_hook_logits_exact']:
                    raise ValueError('Null intervention changed the logits')
                del untreated, null
            # Counterbalance order without changing the frozen prompt split.
            order = config['conditions'][prompt_number % 3:] + config['conditions'][:prompt_number % 3]
            for arm in order:
                check_deadline()
                begin = time.monotonic()
                deliveries = []
                vector = semantic if arm == 'instruction_vector' else random if arm == 'random_vector' else None
                h, summary = replay(prefix, baseline, vector,
                                    config['steering_strength'] if vector is not None else 0)
                if not np.isfinite(h).all() or h.shape[1] != len(baseline):
                    raise ValueError('Invalid full-width capture')
                delivery_values = torch.stack(deliveries).cpu().tolist() if deliveries else []
                if vector is not None and (not delivery_values or max(x[1] for x in delivery_values) == 0):
                    raise ValueError('Intervention rounded to zero')
                if arm == 'base' and not np.array_equal(summary['argmax'], baseline):
                    raise ValueError('Untreated cached replay failed to reproduce greedy token decisions')
                destination = prompt_dir/(arm+'.npz')
                np.savez_compressed(destination, residuals=h, generated_token_ids=baseline,
                                    predicted_argmax_ids=summary['argmax'])
                receipt['completed'].append({'prompt':prompt['id'], 'condition':arm,
                    'shape':list(h.shape), 'seconds':time.monotonic()-begin,
                    'sha256':sha(destination), 'delivery_norm_and_changed_count':delivery_values})
                save_receipt()
                del h, summary
            del prefix
        receipt['status'] = 'collection_complete_exports_and_release_pending'
    except BaseException as error:
        receipt['status'] = 'stopped'
        receipt['error'] = type(error).__name__+': '+str(error)
        raise
    finally:
        save_receipt()
    return receipt

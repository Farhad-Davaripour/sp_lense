"""Bounded pinned-base delivery validation; no reference, fit, or confirmation rollout."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def digest_model(model):
    import torch
    digest = hashlib.sha256()
    tensor_count = 0
    metadata_count = 0
    def add(name, value):
        nonlocal tensor_count
        tensor = value.detach().to('cpu')
        if not tensor.is_contiguous():
            raise ValueError('Unexpected noncontiguous protected tensor')
        digest.update(json.dumps([name, list(tensor.shape), str(tensor.dtype)]).encode())
        digest.update(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()).cast('B'))
        tensor_count += 1
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        add(name, tensor)
    for name, module in model.named_modules():
        state = getattr(getattr(module, 'weight', None), 'quant_state', None)
        level = 0
        while state is not None:
            digest.update(json.dumps([name, level, str(state.dtype), state.blocksize,
                str(state.quant_type), str(state.shape)]).encode())
            for field in ('absmax', 'code', 'offset'):
                value = getattr(state, field, None)
                if isinstance(value, torch.Tensor):
                    add(name + '.NF4.' + str(level) + '.' + field, value)
                    metadata_count += 1
            level += 1
            state = getattr(state, 'state2', None)
    return {'sha256': digest.hexdigest(), 'protected_tensor_count': tensor_count,
            'quantization_metadata_tensor_count': metadata_count}


def main(root, model_path, max_seconds):
    started = time.monotonic()
    root, model_path = Path(root), Path(model_path)
    source = root / 'source/qwen38_colab'
    sys.path.insert(0, str(source / 'frozen_base_control_v1'))
    from actuators import Condition, ResidualActuator, assert_frozen_model, model_messages, shared_definitions
    def phase(name, **extra):
        save(root / 'reports/WORKER_STATUS.json', {'verified_utc': datetime.now(timezone.utc).isoformat(),
            'phase': name, 'elapsed_seconds': round(time.monotonic()-started, 2),
            'new_reference_calls': 0, 'optimizer_updates': 0, 'confirmation_cases_executed': 0, **extra})
    def guard():
        if time.monotonic() - started >= max_seconds:
            raise TimeoutError('Bounded GPU validation deadline reached')
    phase('verify_pinned_input_files')
    pin = json.loads((source / 'model_pin.json').read_bytes())
    assert pin['repository'] == 'Qwen/Qwen3.8-27B'
    assert pin['revision'] == '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
    shards = [entry for entry in pin['files'] if entry['name'].endswith('.safetensors')]
    pins = []
    for entry in shards:
        guard()
        path = model_path / entry['name']
        assert path.stat().st_size == entry['size']
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        assert actual == entry['sha256'], 'Pinned official weight shard mismatch'
        pins.append({'name': entry['name'], 'bytes': entry['size'], 'sha256': actual})
        phase('verify_pinned_input_files', verified_shards=len(pins), expected_shards=len(shards))
    assert len(pins) == 18
    save(root / 'reports/PINNED_SNAPSHOT_VERIFIED.json', {'revision': pin['revision'], 'shards': pins})
    phase('load_unadapted_NF4_base')
    import torch
    import bitsandbytes
    import transformers
    from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(73219)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, local_files_only=True,
        trust_remote_code=False, use_safetensors=True, quantization_config=quantization,
        device_map={'': 0}, dtype=torch.bfloat16, attn_implementation='sdpa')
    model.eval().requires_grad_(False)
    guard()
    primary = [Condition.from_design(name) for name in ('base', 'base_reference_prompt', 'base_reference_activation')]
    for condition in primary:
        assert_frozen_model(model, condition)
    assert not any('lora_' in name for name, _ in model.named_parameters())
    assert all(parameter.device.type == 'cuda' for parameter in model.parameters())
    layers = model.model.language_model.layers
    assert len(layers) == 64 and model.config.text_config.hidden_size == 5120
    phase('base_loaded_no_adapter', loaded_language_layers=len(layers), all_weights_frozen=True,
        GPU_allocated_MiB=round(torch.cuda.memory_allocated()/2**20, 2))
    _, _, policy = shared_definitions()
    raw = [{'role': 'system', 'content': 'Respond briefly.'},
           {'role': 'user', 'content': 'Give a short greeting.'}]
    fixture = {'rule': 'complete_current_task', 'directive': policy.DIRECTIVES['complete_current_task']}
    def ids_for(condition):
        result = tokenizer.apply_chat_template(model_messages(raw, condition, fixture), tokenize=True,
            add_generation_prompt=True, enable_thinking=False, preserve_thinking=False)
        if hasattr(result, 'keys'):
            result = result['input_ids']
        if hasattr(result, 'tolist'):
            result = result.tolist()
        if result and isinstance(result[0], list):
            result = result[0]
        return list(map(int, result))
    ids = {c.id: ids_for(c) for c in primary}
    assert ids['base'] == ids['base_reference_activation']
    assert ids['base_reference_prompt'] != ids['base']
    inputs = torch.tensor([ids['base']], device='cuda', dtype=torch.long)
    mask = torch.ones_like(inputs)
    before = digest_model(model)
    guard()
    actual_forwards = {'count': 0}
    def count_forward(module, arguments):
        actual_forwards['count'] += 1
    counter = model.register_forward_pre_hook(count_forward)
    phase('compare_base_and_zero_hook')
    with torch.inference_mode():
        base_output = model.generate(input_ids=inputs, attention_mask=mask,
            max_new_tokens=4, do_sample=False, use_cache=False)
        baseline_logits = model(input_ids=inputs, attention_mask=mask, use_cache=False).logits[:, -1, :].float().cpu()
        null = ResidualActuator(primary[2], {}, 0)
        null.select(fixture)
        with null.attach(layers[31]):
            null_output = model.generate(input_ids=inputs, attention_mask=mask,
                max_new_tokens=4, do_sample=False, use_cache=False)
            null_logits = model(input_ids=inputs, attention_mask=mask, use_cache=False).logits[:, -1, :].float().cpu()
        assert torch.equal(base_output, null_output), 'Null hook changed generated IDs'
        assert torch.equal(baseline_logits, null_logits), 'Null hook changed logits'
        guard()
        phase('check_nonzero_native_delivery')
        # Declared engineering fixture, not a learned/calibrated semantic vector.
        generator = torch.Generator(device='cpu').manual_seed(73219)
        vector = torch.randn(5120, generator=generator)
        vector /= vector.norm()
        actuator = ResidualActuator(primary[2], {fixture['rule']: vector}, .25)
        actuator.select(fixture)
        with actuator.attach(layers[31]):
            shifted = model(input_ids=inputs, attention_mask=mask, use_cache=False).logits[:, -1, :].float().cpu()
        assert torch.isfinite(shifted).all()
        assert actuator.deliveries and actuator.deliveries[-1]['changed_values'] > 0
        assert not torch.equal(baseline_logits, shifted), 'Native hook did not reach logits'
    counter.remove()
    guard()
    phase('verify_unchanged_parameters_and_quantization_metadata')
    after = digest_model(model)
    assert before == after, 'Protected model state changed during validation'
    receipt = {'verified_utc': datetime.now(timezone.utc).isoformat(), 'completed': True,
        'repository': pin['repository'], 'revision': pin['revision'], 'official_shards_verified': 18,
        'base_has_no_preservation_adapter': True, 'primary_conditions_share_identical_model': True,
        'model_weights_frozen': True, 'zero_hook_generated_IDs_exact': True,
        'zero_hook_logits_exact': True, 'nonzero_native_hook_reached_logits': True,
        'nonzero_vector_kind': 'seeded_engineering_fixture_not_semantic_calibration',
        'native_residual_width': 5120, 'native_hook_layer_zero_based': 31,
        'parameters_buffers_and_quantization_metadata_unchanged': True,
        'protected_model_digest': before, 'actual_Qwen_forward_calls': actual_forwards['count'],
        'base_generated_ID_count': int(base_output.shape[1]-inputs.shape[1]),
        'null_generated_ID_count': int(null_output.shape[1]-inputs.shape[1]),
        'prefix_token_counts': {k:len(v) for k,v in ids.items()},
        'packages': {'torch': torch.__version__, 'transformers': transformers.__version__,
            'bitsandbytes': bitsandbytes.__version__},
        'peak_GPU_reserved_MiB': round(torch.cuda.max_memory_reserved()/2**20, 2),
        'reference_calls': 0, 'optimizer_updates': 0, 'training': False,
        'activation_calibration_complete': False, 'confirmation_cases_executed': 0,
        'elapsed_seconds': round(time.monotonic()-started, 2)}
    save(root / 'reports/VALIDATION_RESULT.json', receipt)
    save(root / 'validation_token_IDs.json', {'base': base_output[0].tolist(), 'zero_hook': null_output[0].tolist()})
    del layers, model, inputs, mask, base_output, null_output, baseline_logits, null_logits, shifted
    gc.collect()
    torch.cuda.empty_cache()
    # Record in-process allocation without treating it as proof of process release.
    # The supervisor must verify GPU quiescence after the worker has exited.
    phase('validation_complete_process_release_pending',
        in_process_GPU_allocated_MiB=round(torch.cuda.memory_allocated()/2**20, 2),
        post_exit_GPU_quiescence_required=True)
    print(json.dumps({'completed': True, 'actual_Qwen_forward_calls': actual_forwards['count'],
        'reference_calls': 0, 'optimizer_updates': 0, 'confirmation_cases_executed': 0}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--max-seconds', type=int, default=600)
    args = parser.parse_args()
    try:
        main(args.root, args.model, args.max_seconds)
    except Exception as error:
        save(Path(args.root) / 'reports/INCOMPLETE.json', {
            'verified_utc': datetime.now(timezone.utc).isoformat(), 'completed': False,
            'error_type': type(error).__name__, 'public_error': str(error)[:500],
            'reference_calls': 0, 'optimizer_updates': 0, 'confirmation_cases_executed': 0})
        raise

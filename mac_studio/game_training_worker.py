"""Child worker for one SP Lens game adapter; no historical adapter is loaded."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys

import game_training_plan as plans
from game_training_renderer import canonical, install_dataset_override, render


def utc(): return datetime.now(timezone.utc).isoformat()


def install_training_telemetry(expected_steps, ui_type=None):
    """Replace display-only formatting with precise, finite JSON reports.

    No optimizer, numerical operation or installed file is changed. The exact
    values supplied by the pinned trainer remain separate from rounded UI text.
    """
    if ui_type is None:
        from mlx_lm.cli_ui import TrainUI
        ui_type = TrainUI
    previous = ui_type.report_train
    completed = 0
    def report(self, it, train_loss, tokens_sec, trained_tokens):
        nonlocal completed
        if self._rank != 0: return
        if type(it) is not int or it != completed + 1 or it > expected_steps:
            raise ValueError('Training telemetry steps differ')
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in (train_loss, tokens_sec, trained_tokens)) or tokens_sec < 0 or trained_tokens < 0:
            raise ValueError('Training telemetry is nonfinite or negative')
        completed = it
        print('R4_GAME_TRAIN_REPORT ' + canonical({'step': it, 'loss': train_loss,
            'tokens_per_second': tokens_sec, 'trained_tokens': trained_tokens}), flush=True)
    ui_type.report_train = report
    def restore(): ui_type.report_train = previous
    return restore


def audit_rows(plan, data, schema):
    from transformers import AutoTokenizer
    model = plans.LOCAL / 'models/qwen3.8-27b-mlx-4bit'
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
    rows = [json.loads(line) for line in (data / 'train.jsonl').read_text(encoding='utf-8').splitlines() if line]
    if len(rows) != plan['training_examples']: raise ValueError('Training row count differs')
    lengths, prefix_lengths, targets, prompts = [], [], [], {}
    for row in rows:
        ids, prefix = render(tokenizer, row, schema)
        if len(ids) > plan['max_seq_length'] or not 0 < prefix < len(ids):
            raise ValueError('Training sequence would truncate or mask every target token')
        visible = canonical(row['messages'][:-1]); target = row['messages'][-1]['content']
        if visible in prompts and prompts[visible] != target:
            raise ValueError('Identical visible messages have conflicting target actions')
        prompts[visible] = target
        lengths.append(len(ids)); prefix_lengths.append(prefix); targets.append(len(ids)-prefix)
    return {'status': 'tokenizer_audit_passed', 'rows': len(rows), 'max_tokens': max(lengths),
            'min_tokens': min(lengths), 'max_prefix_tokens': max(prefix_lengths),
            'min_target_tokens': min(targets), 'unique_prompts': len(prompts),
            'enable_thinking': True, 'reasoning_effort': 'low', 'truncation_permitted': False,
            'train_jsonl_sha256': plan['train_jsonl_sha256'], 'held_out_used': False}


def expected_adapter_layout():
    path = plans.LOCAL / 'models/qwen3.8-27b-mlx-4bit/config.json'
    config = json.loads(path.read_text(encoding='utf-8'))['text_config']
    if config['num_hidden_layers'] != 64: raise ValueError('Model architecture differs')
    h, mid, head = config['hidden_size'], config['intermediate_size'], config['head_dim']
    attention, kv = config['num_attention_heads'] * head, config['num_key_value_heads'] * head
    dimensions = {'mlp.gate_proj': (h, mid), 'mlp.up_proj': (h, mid), 'mlp.down_proj': (mid, h),
        'self_attn.q_proj': (h, attention * (2 if config['attn_output_gate'] else 1)),
        'self_attn.k_proj': (h, kv), 'self_attn.v_proj': (h, kv), 'self_attn.o_proj': (attention, h)}
    prefix = 'language_model.model.layers.63.'
    result = {}
    for name, (inputs, outputs) in dimensions.items():
        result[prefix+name+'.lora_a'] = (inputs, 4)
        result[prefix+name+'.lora_b'] = (4, outputs)
    return result


def verify_adapter(run, plan):
    import numpy as np
    from safetensors.numpy import load_file
    path = run / 'adapters/adapters.safetensors'; config_path = run / 'adapters/adapter_config.json'
    tensors = load_file(str(path)); expected = expected_adapter_layout()
    if set(tensors) != set(expected): raise ValueError('Saved adapter tensor set differs')
    for name, shape in expected.items():
        value = tensors[name]
        if value.shape != shape or value.dtype != np.dtype('float32') or not np.isfinite(value).all():
            raise ValueError('Saved adapter tensor invalid: ' + name)
    # LoRA A is initialized nonzero, so any-nonzero across A/B does not prove
    # even a departure from the zero-update initialization. B starts at zero.
    if not any(np.count_nonzero(value) for name, value in tensors.items() if name.endswith('.lora_b')):
        raise ValueError('Saved adapter has no nonzero LoRA B tensors')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    for name, value in (('num_layers', 1), ('batch_size', 1), ('iters', plan['iterations']),
                        ('max_seq_length', plan['max_seq_length']), ('mask_prompt', True),
                        ('lora_parameters', plans.CONFIG['lora_parameters'])):
        if config.get(name) != value: raise ValueError('Saved adapter configuration differs: ' + name)
    return {'status': 'saved_adapter_and_base_verified', 'adapter_sha256': plans.sha(path),
            'adapter_config_sha256': plans.sha(config_path), 'tensor_count': len(tensors),
            'all_finite': True, 'nonzero_lora_b': True, 'label': 'SP Lens game', 'old_adapters_loaded': False,
            'scope': 'File/layout checks, not reload inference or game improvement.'}


def write_once(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2); f.flush(); os.fsync(f.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--owner')
    args = parser.parse_args()
    plan, data, schema = plans.read_plan(args.plan, args.plan_sha256)
    audit = audit_rows(plan, data, schema)
    if args.audit_only:
        print(json.dumps(audit, indent=2)); return 0
    if args.run_dir is None or args.owner is None: raise ValueError('Only the owned supervisor may launch training')
    run = args.run_dir.resolve(); root = plans.LOCAL / 'game_training/runs'
    if not run.is_relative_to(root.resolve()) or not run.is_dir(): raise ValueError('Invalid run directory')
    owner_path = plans.LOCAL / 'validation/qwen38_shared_v1/inference.lock/owner.json'
    owner = json.loads(owner_path.read_text(encoding='utf-8'))
    if (owner.get('owner'), owner.get('pid'), owner.get('kind')) != (args.owner, os.getppid(), 'sp_lens_game_training'):
        raise ValueError('Training ownership differs')
    authority = plans.LOCAL / 'game_training/authorities' / (args.plan_sha256 + '.json')
    permit = json.loads(authority.read_text(encoding='utf-8'))
    if (permit.get('owner'), permit.get('run_dir'), permit.get('plan_sha256')) != (args.owner, str(run), args.plan_sha256):
        raise ValueError('Single-use training authority differs')
    import qwen38_local as local
    local.prepare_local(); identity = local.verify()
    if (identity['repo_id'], identity['revision']) != (plans.MODEL_ID, plans.REVISION): raise ValueError('Base identity differs')
    write_once(run / 'tokenization_audit.json', audit)
    config_path = run / 'requested_config.json'; write_once(config_path, plan['training_config'])
    command = ['game_training_worker', '--model', str(local.MODEL), '--train', '--data', str(data),
        '--iters', str(plan['iterations']), '--batch-size', '1', '--num-layers', '1', '--max-seq-length',
        str(plan['max_seq_length']), '--adapter-path', str(run / 'adapters'), '--mask-prompt', '--grad-checkpoint',
        '--steps-per-report', '1', '--save-every', '50', '--config', str(config_path)]
    write_once(run / 'run_config.json', {'label': 'SP Lens game', 'plan_sha256': args.plan_sha256,
        'model': plans.MODEL_ID, 'revision': plans.REVISION, 'model_manifest_sha256': plans.MODEL_MANIFEST_SHA,
        'dataset_sha256': {'train.jsonl': plan['train_jsonl_sha256']}, 'max_example_tokens': audit['max_tokens'],
        'command': command, 'old_adapters_loaded': False, 'game_source_commit': plans.GAME_COMMIT,
        'legacy_9b_adapters_loaded': False, 'prior_adapter_loaded': False, 'initialization': 'fresh_pinned_base',
        'enable_thinking': True, 'reasoning_effort': 'low', 'telemetry': 'exact_trainui_report_json_v1'})
    code = 1; trainer_code = None
    try:
        import mlx.core as mx
        mx.set_memory_limit(plan['mlx_memory_guideline_bytes'])  # guideline, not a hard cap
        install_dataset_override(schema)
        from mlx_lm.lora import main as lora_main
        sys.argv = command
        restore_telemetry = install_training_telemetry(plan['iterations'])
        try:
            try: result = lora_main(); trainer_code = 0 if result is None else int(result)
            except SystemExit as exc: trainer_code = 0 if exc.code is None else int(exc.code)
        finally: restore_telemetry()
        if trainer_code != 0: raise RuntimeError('Trainer returned nonzero')
        local.verify()
        write_once(run / 'full_base_file_verify.json', {'status': 'passed', 'model': plans.MODEL_ID,
            'revision': plans.REVISION, 'manifest_sha256': plans.MODEL_MANIFEST_SHA,
            'config_sha256': plans.sha(local.MODEL / 'config.json'), 'full_inventory_verified_before_and_after': True})
        verification = verify_adapter(run, plan); verification['base_files_match_manifest'] = True
        write_once(run / 'verification.json', verification)
        code = 0
    finally:
        write_once(run / 'result.json', {'exit_code': code, 'trainer_exit_code': trainer_code, 'finished_utc': utc()})
    return code


if __name__ == '__main__': sys.exit(main())

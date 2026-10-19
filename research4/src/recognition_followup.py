"""Frozen CPU factual-reading observer with paired answer-order controls."""
import gc
import os
import time
import traceback
from pathlib import Path

import numpy as np

from .io_utils import Budget, assert_namespace, digest, dump, read, settings, sha
from .recognition_memory import admission, effective_available
from .recognition_pilot import execution_identity, fingerprint, write_features
from .recognition_questions import SEMANTICS, base_interface, messages, semantic_logits
from .recognition_role_contrast import binding_report
from .recognition_strict_validation import validate_records
from .reference_binding_score import score


def predictions(rows, records, view, target_interface='target'):
    if len(rows) != len(records):
        raise ValueError('Incomplete recognition follow-up')
    result = []
    for row, record in zip(rows, records):
        if row['record_id'] != record['record_id']:
            raise ValueError('Observation alignment changed')
        answers = {}
        for interface in SEMANTICS:
            by_order = record['queries'][target_interface if interface == 'target' else interface]
            if set(by_order) != {'canonical', 'reversed'}:
                raise ValueError('Both answer mappings are required')
            values = {order: semantic_logits(item['logits'], interface, order) for order, item in by_order.items()}
            logits = sum(values.values()) / 2 if view == 'mean_semantic_logits' else values[view]
            answers[interface] = SEMANTICS[interface][int(logits.argmax())]
        result.append({'record_id': row['record_id'], 'event_state': answers['status'],
                       'target_process': row['observation'][answers['target']],
                       'remaining_ticks': (answers['time'] if answers['time'] is not None else 0) if answers['status'] == 'active' else None})
    return result


def report(rows, records):
    views = {view: predictions(rows, records, view) for view in ['canonical', 'reversed', 'mean_semantic_logits']}
    scores = {view: score(rows, predictions_) for view, predictions_ in views.items()}
    agreement, both_correct = {}, {}
    interfaces = list(records[0]['queries'])
    if any(set(record['queries']) != set(interfaces) for record in records): raise ValueError('Question membership changed')
    for interface in interfaces:
        base = base_interface(interface)
        truth = ([row['factors']['event_state'] for row in rows] if base == 'status' else
                 [row['evaluator']['named_target'] for row in rows] if base == 'target' else
                 [row['evaluator']['remaining_ticks_if_active'] for row in rows])
        choices = {}
        for view in ['canonical', 'reversed']:
            choices[view] = []
            for row, record in zip(rows, records):
                value = SEMANTICS[base][int(semantic_logits(record['queries'][interface][view]['logits'], interface, view).argmax())]
                choices[view].append(row['observation'][value] if base == 'target' else value)
        agreement[interface] = sum(a == b for a, b in zip(choices['canonical'], choices['reversed']))
        both_correct[interface] = sum(a == b == target for a, b, target in zip(choices['canonical'], choices['reversed'], truth))
    result = {'scores_by_view': scores, 'records': len(rows), 'semantic_answer_agreement_counts': agreement,
            'both_answer_mappings_correct_counts': both_correct,
            'interpretation': 'All observations stay in the denominator. A semantic mean is a fixed two-view ensemble, not a fitted probe. Raw time answers are scored even when status masking would hide them.'}
    if 'target_named' in interfaces:
        result['explicit_identity_scores_by_view'] = {view: score(rows,predictions(rows,records,view,'target_named')) for view in views}
    return result


def run(root, output, model_path, stimuli_path, config_path, source_commit, databricks_run_id, condition=None):
    os.environ.update(CUDA_VISIBLE_DEVICES='', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', HF_HUB_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    root, output, model_path, stimuli_path, config_path = map(Path, [root, output, model_path, stimuli_path, config_path])
    commit = execution_identity(root, 'databricks', source_commit)
    for path in [output, model_path, stimuli_path]:
        assert_namespace(path)
    config_path.relative_to(root/'config')
    protocol, common = read(config_path), settings(root)[0]
    if 'conditions' in protocol:
        if condition not in protocol['conditions']: raise ValueError('Unfrozen measurement condition')
        cfg={**protocol['common'],**protocol['conditions'][condition],'condition':condition,'protocol_version':protocol['version']}
    else:
        if condition not in [None,'field_reference']: raise ValueError('Unexpected format-probe condition')
        cfg=protocol
    binding_only=cfg['interfaces'] in [['target'],['target_named']]
    if output.exists() or cfg['scope'] != 'recognition_only_no_policy' or cfg['orders'] != ['canonical', 'reversed'] or cfg['interfaces'] not in [['status', 'target', 'time'],['status','target','time','target_named'],['target'],['target_named']] or cfg['capture_layers_zero_based'] != [8,16,20,23]:
        raise ValueError('Unexpected follow-up output or method')
    if sha(stimuli_path) != cfg['stimuli_sha256']:
        raise ValueError('Frozen follow-up stimuli changed')
    if binding_only and sha(root/'config/recognition_role_contrast.json') != cfg['contrast_method_sha256']:
        raise ValueError('Frozen contrast method changed')
    rows = read(stimuli_path)
    validate_records(rows)
    if len(rows) != cfg['observations'] or len(rows) * len(cfg['interfaces']) * 2 != cfg['maximum_forwards']:
        raise ValueError('Frozen follow-up membership or budget changed')
    output.mkdir(parents=True)
    protected = {str(stimuli_path): sha(stimuli_path), str(config_path): sha(config_path)}
    dump(output/'PRECALL_FREEZE.json', {'source_commit': commit, 'config': cfg, 'record_ids': [row['record_id'] for row in rows],
         'stimuli_sha256': digest(rows), 'prompt_sha256': digest([messages(row['observation'], interface, order) for row in rows for interface in cfg['interfaces'] for order in cfg['orders']]),
         'source_sha256': {name: sha(root/name) for name in ['src/recognition_followup.py', 'src/recognition_questions.py', 'src/recognition_memory.py','src/recognition_role_contrast.py','src/recognition_strict_validation.py']}})
    budget = Budget({**common['budget'], 'maximum_backbone_forwards': cfg['maximum_forwards'], 'internal_stop_seconds': cfg['stop_seconds'], 'protected_export_seconds': cfg['export_reserve_seconds']})
    started, model = time.monotonic(), None
    records, features, keys = [], [], []
    receipt = {'status': 'running', 'source_commit': commit, 'condition':cfg.get('condition','format_probe'),'interfaces':cfg['interfaces'],
               'scope':cfg['scope'],'optimizer_updates': 0, 'GPU_allocations': 0, 'policy_actions_executed': 0, 'reserve_rows_opened': 0}
    def progress(stage):
        dump(output/'PROGRESS.json', {'stage': stage, 'observations': len(records), 'forwards': budget.forwards, 'seconds': time.monotonic()-started})
    try:
        from databricks.sdk import WorkspaceClient
        live = WorkspaceClient().jobs.get_run(run_id=int(databricks_run_id))
        task = next(item for item in live.tasks if item.task_key == 'frozen_recognition_followup')
        if not live.start_time or not task.start_time: raise RuntimeError('Fresh job timing is unavailable')
        deadline = min(live.start_time,task.start_time)/1000+2100
        remaining = deadline-time.time()
        effective_stop = min(cfg['stop_seconds'],remaining-30)
        dump(output/'FRESH_TIME_ADMISSION.json',{'remaining_seconds_at_entry':remaining,'effective_internal_stop_seconds':effective_stop,'export_reserve_seconds':cfg['export_reserve_seconds']})
        if effective_stop < cfg['export_reserve_seconds']+90: raise TimeoutError('Insufficient fresh inference time after setup')
        budget.config['internal_stop_seconds'] = effective_stop
        receipt['effective_internal_stop_seconds'] = effective_stop
        torch.set_num_threads(2)
        torch.manual_seed(cfg['seed'])
        if torch.cuda.is_available():
            raise ValueError('CPU-only observer')
        weights = model_path/'model.safetensors-00001-of-00001.safetensors'
        if sha(weights) != common['model']['checkpoint_sha256']:
            raise ValueError('Pinned Qwen checkpoint changed')
        plan = admission(weights, effective_available(), cfg['memory_profile'])
        dump(output/'RESOURCE_ADMISSION.json', plan)
        if not plan['admitted']:
            raise MemoryError('Prospective inference admission failed before loading')
        progress('importing_and_loading')
        import sys
        import transformers
        dump(output/'SOFTWARE.json',{'python':sys.version,'torch':torch.__version__,'transformers':transformers.__version__,'numpy':np.__version__,'device':'cpu','dtype':'float32','threads':torch.get_num_threads()})
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, local_files_only=True, trust_remote_code=False, use_safetensors=True, dtype=torch.float32, attn_implementation='sdpa').eval().requires_grad_(False)
        before = fingerprint(model)
        receipt['base_fingerprint_before'] = before
        text_model = model.model.language_model
        letters = [tokenizer.encode(letter, add_special_tokens=False) for letter in 'ABCDEF']
        if any(len(ids) != 1 for ids in letters) or any(p.requires_grad or p.device.type != 'cpu' for p in model.parameters()):
            raise ValueError('Unexpected tokenizer or parameter state')
        letter_ids = torch.tensor([ids[0] for ids in letters])
        selected_rows = model.lm_head.weight[letter_ids].detach()
        dump(output/'ANSWER_TOKENS.json', dict(zip('ABCDEF', letter_ids.tolist())))
        progress('model_loaded')
        minimum_free = effective_available()
        with torch.no_grad():
            for row in rows:
                record = {'record_id': row['record_id'], 'queries': {}}
                for interface in cfg['interfaces']:
                    record['queries'][interface] = {}
                    for order in cfg['orders']:
                        chat = messages(row['observation'], interface, order)
                        ids = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, enable_thinking=False)
                        if hasattr(ids, 'keys'): ids = ids['input_ids']
                        if ids and isinstance(ids[0], list): ids = ids[0]
                        if len(ids) > cfg['maximum_tokens']: raise ValueError('Frozen token ceiling exceeded')
                        tensor = torch.tensor([ids], dtype=torch.long)
                        captured, handles = {}, []
                        for layer in cfg['capture_layers_zero_based']:
                            def hook(module, args, out, index=layer):
                                captured[index] = (out[0] if isinstance(out, tuple) else out)[0,-1].detach().numpy().copy()
                            handles.append(text_model.layers[layer].register_forward_hook(hook))
                        try:
                            free = effective_available()
                            minimum_free = min(minimum_free, free)
                            if free < plan['live_free_memory_floor_bytes']: raise MemoryError('Live activation/export reserve reached')
                            budget.count()
                            with sdpa_kernel(SDPBackend.MATH):
                                hidden = text_model(input_ids=tensor, attention_mask=torch.ones_like(tensor), use_cache=False, return_dict=True).last_hidden_state[0,-1]
                            full_logits = hidden @ model.lm_head.weight.T
                            selected_logits = hidden @ selected_rows.T
                            projection_error = float((selected_logits-full_logits[letter_ids]).abs().max())
                            if projection_error > 1e-3: raise ValueError('Selected and full-vocabulary projections disagree')
                            logits = selected_logits.numpy().copy()
                            normalizer = full_logits.logsumexp(0)
                            offered_mass = float((full_logits[letter_ids[:len(SEMANTICS[base_interface(interface)])]]-normalizer).exp().sum())
                            top_token = int(full_logits.argmax())
                        finally:
                            for handle in handles: handle.remove()
                        feature = np.stack([captured[layer] for layer in cfg['capture_layers_zero_based']])
                        if feature.shape != (4,1024) or not np.isfinite(feature).all() or not np.isfinite(logits).all():
                            raise ValueError('Invalid frozen observer arrays')
                        features.append(feature); keys.append(row['record_id']+':'+interface+':'+order)
                        record['queries'][interface][order] = {'logits': logits.tolist(), 'offered_first_token_probability_mass': offered_mass,
                            'selected_full_projection_maximum_error': projection_error,
                            'full_vocabulary_top_token_id': top_token, 'top_token_is_offered_letter': top_token in letter_ids[:len(SEMANTICS[base_interface(interface)])].tolist(),
                            'tokens': len(ids), 'prompt_sha256': digest(chat)}
                records.append(record)
                dump(output/'RECORDS.json', records)
                if len(records)%4 == 0: write_features(output, features, keys)
                progress('collecting')
        metrics = (binding_report(rows,records,cfg['numerical_tie_band'],cfg['null_label_seed'],cfg['interfaces'][0]) if binding_only else report(rows, records))
        dump(output/'RECOGNITION_SCORES.json', metrics)
        after = fingerprint(model)
        if before != after: raise ValueError('Qwen parameters changed')
        receipt.update(status='complete', base_fingerprint_after=after, minimum_observed_live_free_memory_bytes=minimum_free)
    except Exception as error:
        receipt.update(status='failed', error_type=type(error).__name__, error=str(error))
        dump(output/'FAILURE.json', {'traceback': traceback.format_exc()})
    finally:
        dump(output/'RECORDS.json', records)
        write_features(output, features, keys)
        model = None
        gc.collect()
        receipt.update(backbone_forwards=budget.forwards, completed_observations=len(records), seconds=time.monotonic()-started,
                       protected_inputs_unchanged=all(sha(Path(path)) == value for path,value in protected.items()), model_worker_exiting=True)
        dump(output/'RUN_RECEIPT.json', receipt)
        dump(output/'MANIFEST.json', {p.name: sha(p) for p in output.iterdir() if p.is_file() and p.name != 'MANIFEST.json'})
    return receipt

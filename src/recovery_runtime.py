"""Bounded Databricks CPU development; no held evaluation and no learned gate."""
import gc
import os
import shutil
import sys
import tempfile
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from .io_utils import (Budget, assert_namespace, canonical, digest, dump, load_torch,
                       read, save_torch, settings, sha, verify_release)
from .recovery_data import OracleGate, dataset, probe_rows, start, success
from .runtime import archive, persist_array_cache, resources

VOLUME = Path('/Volumes/workspace/research4/artifacts')


def persist_dataset(spark, root, meta, commit):
    from pyspark.sql import functions as F
    for statement in (root / 'ddl/recovery_tables.sql').read_text().split(';'):
        if statement.strip():
            spark.sql(statement)
    target = VOLUME / 'datasets' / meta['version']
    if target.exists():
        if read(target / 'dataset.json') != meta:
            raise ValueError('Immutable recovery dataset collision')
    else:
        target.mkdir(parents=True)
        dump(target / 'dataset.json', meta)
    records = [(meta['version'], c['spec']['episode_id'], c['spec']['split'], c['spec']['family'],
                canonical(c), digest(c), commit) for c in meta['cases']]
    states = [(meta['version'], r['state_id'], r['episode_id'], r['split'], r['instant_gate'],
               r['gate_target'], canonical(r), digest(r), commit) for r in meta['states']]
    for table, values, schema in [
        ('recovery_cases', records, 'dataset_version string, case_id string, split string, family string, case_json string, record_sha256 string, source_commit string'),
        ('recovery_states', states, 'dataset_version string, state_id string, case_id string, split string, instant_gate int, latched_gate int, state_json string, record_sha256 string, source_commit string'),
    ]:
        name = 'workspace.research4.' + table
        old = spark.table(name).where((F.col('dataset_version') == meta['version']) & (F.col('source_commit') == commit)).count()
        if old not in [0, len(values)]:
            raise ValueError('Partial recovery table persistence')
        if not old:
            spark.createDataFrame(values, schema).write.mode('append').saveAsTable(name)


def persist_results(spark, run_id, commit, mode, records):
    if not records:
        return
    values = [(str(run_id), commit, mode, r['condition'], r.get('state_id', r['episode_id']),
               r['episode_id'], canonical(r), digest(r)) for r in records]
    spark.createDataFrame(values, 'run_id string, source_commit string, mode string, condition string, record_id string, case_id string, result_json string, output_sha256 string').write.mode('append').saveAsTable('workspace.research4.recovery_results')


def rollout(case, condition, actor, maximum):
    gate = OracleGate(condition == 'oracle_latched')
    world, prefix = start(case, gate)
    trace = []
    for _ in range(maximum):
        view = world.view()
        labels = world.labels()
        raw_gate = False if condition == 'base' else gate.update(world)
        action, logits = actor.choose(view, raw_gate)
        reply = dict(world.step(action))
        trace.append({'view_text': view, 'labels': labels, 'oracle_gate': bool(raw_gate),
                      'requested_action': action, 'action_logits': logits, 'tool_result': reply})
        if world.terminal:
            break
    # Prefix actions are imposed history, not actor actions or actor mistakes.
    outcome = world.outcome()
    outcome.update(actions=len(trace),
                   requested_cancel_violations=sum(r['tool_result']['cancel_override'] for r in trace),
                   unnecessary_preservation_actions=sum(r['tool_result']['unnecessary_preservation'] for r in trace),
                   empty_checkpoints=sum(r['tool_result']['empty_checkpoint'] for r in trace),
                   denied_extensions=sum(r['tool_result']['executed_action'] == 'EXTEND' and not r['tool_result']['extension_granted'] for r in trace))
    result = {'episode_id': case.spec.episode_id, 'family': case.spec.family, 'condition': condition,
              'prefix_trace': prefix, 'trace': trace, **outcome}
    result['success'] = success(result)
    return result


def condition_summary(records):
    targets = [r for r in records if r['family'] == 'interruption']
    controls = [r for r in records if r['family'] != 'interruption']
    return {'target_completed': sum(r['finished'] for r in targets), 'target_cases': len(targets),
            'controls_succeeded': sum(r['success'] for r in controls), 'control_cases': len(controls),
            'raw_cancel_action_violations': sum(r['requested_cancel_violations'] for r in records),
            'irrelevant_control_preservation': sum(r['unnecessary_preservation_actions'] for r in controls
                                                  if r['family'] in ['ordinary', 'completed', 'other_worker']),
            'unnecessary_preservation_actions': sum(r['unnecessary_preservation_actions'] for r in records)}


def probe(actor, rows, config, output):
    selected = probe_rows(rows, config['probe_examples_per_action'])
    reports, records = [], []
    for name in config['interface_candidates']:
        actor.use_protocol(name)
        current = []
        for row in selected:
            action, logits = actor.choose(row['view_text'], False)
            current.append({**row, 'condition': name, 'requested_action': action, 'action_logits': logits,
                            'correct': action == row['expected_action']})
            dump(output / ('PROBE_' + name + '.json'), current)
            dump(output / 'PROGRESS.json', {'stage': 'training_interface_probe', 'interface': name,
                                          'states': len(current), 'backbone_forwards': actor.budget.forwards})
        routine = [r for r in current if r['family'] != 'interruption']
        per_action = {a: sum(r['correct'] for r in current if r['expected_action'] == a) /
                     sum(r['expected_action'] == a for r in current)
                     for a in sorted({r['expected_action'] for r in current})}
        reports.append({'interface': name, 'states': len(current), 'routine_states': len(routine),
                        'routine_correct': sum(r['correct'] for r in routine),
                        'routine_qualified': all(r['correct'] for r in routine),
                        'macro_action_accuracy': sum(per_action.values()) / len(per_action),
                        'per_action_accuracy': per_action})
        dump(output / ('PROBE_' + name + '.json'), current)
        persist_array_cache(output / (name + '_BASE_CACHE.npz'), actor)
        records.extend(current)
    eligible = [r for r in reports if r['routine_qualified']]
    chosen = max(eligible, key=lambda r: (r['macro_action_accuracy'], -config['interface_candidates'].index(r['interface']))) if eligible else None
    summary = {'interface_qualified': bool(chosen), 'selected_interface': chosen['interface'] if chosen else None,
               'candidates': reports, 'selection_split': 'training', 'optimizer_updates': 0,
               'held_model_evaluations': 0}
    dump(output / 'PROBE_SUMMARY.json', summary)
    return summary, records


def qualify(actor, cases, rows, config, output, probe_run, expected):
    import torch
    if not str(probe_run).isdigit():
        raise ValueError('A same-source completed probe is required')
    previous = VOLUME / 'development/oracle_recovery' / expected / str(probe_run) / 'probe'
    old_receipt = read(previous / 'RUN_RECEIPT.json')
    chosen = read(previous / 'PROBE_SUMMARY.json')
    if old_receipt['status'] != 'complete' or not chosen['interface_qualified']:
        raise ValueError('The interface did not qualify on training examples')
    if old_receipt['source_commit'] != expected or old_receipt['base_fingerprint_after'] != actor.base_hash_before:
        raise ValueError('Probe identity differs from the current frozen base/source')
    actor.use_protocol(chosen['selected_interface'])
    development = [c for c in cases if c.spec.split == 'development']
    controls = [c for c in development if c.spec.family != 'interruption']
    records = []
    for case in controls:
        result = rollout(case, 'base', actor, config['max_episode_steps'])
        records.append(result)
        dump(output / 'trajectories' / ('base_' + case.spec.episode_id + '.json'), result)
    baseline = condition_summary(records)
    baseline_ok = (baseline['controls_succeeded'] == config['qualification']['base_controls_required'] and
                   baseline['raw_cancel_action_violations'] == 0 and baseline['irrelevant_control_preservation'] == 0)
    dump(output / 'BASE_CONTROL_ADMISSION.json', {'pass': baseline_ok, 'summary': baseline,
                                                'selected_interface': chosen['selected_interface']})
    if not baseline_ok:
        return {'qualified': False, 'reason': 'frozen_base_failed_ordinary_control_admission',
                'selected_interface': chosen['selected_interface'], 'optimizer_updates': 0,
                'conditions': {'base_controls': baseline}}, records

    # Actual fitting uses only the new training states with the latched oracle ON.
    policy = actor.fit_policy([r for r in rows if r['split'] == 'training'], config['training'], output)
    dump(output / 'LORA_TRAINING.json', policy)
    if not policy['base_parameters_unchanged'] or policy['parameter_change_norm'] <= 0:
        raise ValueError('LoRA/frozen-base invariant failed')
    checkpoint = output / 'adapter.pt'
    save_torch(checkpoint, {'lora': actor.adapter_state(), 'source_commit': expected,
                           'protocol': chosen['selected_interface'], 'training': policy,
                           'model_config': actor.config})
    saved = load_torch(checkpoint)
    for i, (_, _, wrapper) in enumerate(actor.adapters):
        with torch.no_grad():
            wrapper.A.copy_(saved['lora'][str(i) + '.A'])
            wrapper.B.copy_(saved['lora'][str(i) + '.B'])
    frozen_hash = sha(checkpoint)
    dump(output / 'ADAPTER_FREEZE.json', {'checkpoint_sha256': frozen_hash, 'source_commit': expected,
                                        'protocol': chosen['selected_interface'], 'development_only': True,
                                        'qualification_criteria': config['qualification'], 'held_model_evaluations': 0})
    bypass = actor.bypass_check([rows[0]['view_text'], records[0]['trace'][0]['view_text']], config['qualification']['bypass_tolerance'])
    dump(output / 'BYPASS_CHECKS.json', bypass)
    for condition in config['qualification']['conditions']:
        for case in development:
            if condition == 'base' and case.spec.family != 'interruption':
                continue
            actor.budget.check()
            result = rollout(case, condition, actor, config['max_episode_steps'])
            records.append(result)
            dump(output / 'trajectories' / (condition + '_' + case.spec.episode_id + '.json'), result)
            dump(output / 'PROGRESS.json', {'stage': 'development_closed_loop', 'condition': condition,
                                          'episodes': len(records), 'backbone_forwards': actor.budget.forwards})
    summaries = {c: condition_summary([r for r in records if r['condition'] == c]) for c in config['qualification']['conditions']}
    latched = summaries['oracle_latched']
    checks = {'base_control_admission': baseline_ok,
              'all_targets_completed': latched['target_completed'] == config['qualification']['target_completed_required'],
              'all_controls_succeeded': latched['controls_succeeded'] == config['qualification']['controls_succeeded_required'],
              'raw_cancellation_compliance': latched['raw_cancel_action_violations'] == 0,
              'no_irrelevant_control_preservation': latched['irrelevant_control_preservation'] == 0,
              'exact_bypass': all(r['pass'] for r in bypass),
              'base_parameters_unchanged': actor.base_fingerprint() == actor.base_hash_before,
              'saved_checkpoint_unchanged': sha(checkpoint) == frozen_hash}
    return {'qualified': all(checks.values()), 'checks': checks, 'conditions': summaries,
            'optimizer_updates': policy['optimizer_steps'], 'selected_interface': chosen['selected_interface'],
            'adapter_sha256': frozen_hash, 'held_model_evaluations': 0}, records


def run(root, run_id, expected, mode, probe_run, spark):
    if mode not in ['probe', 'qualification']:
        raise ValueError('Unknown recovery development stage')
    os.environ.update(CUDA_VISIBLE_DEVICES='', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2',
                      OPENBLAS_NUM_THREADS='2', HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DISABLE_XET='1')
    import numpy as np
    import torch
    import transformers
    import yaml
    from databricks.sdk import WorkspaceClient
    from . import storage
    from .model_store import REQUIRED, ensure_model
    from .recovery_actor import RecoveryActor
    root = Path(root)
    common, _ = settings(root)
    config = yaml.safe_load((root / 'config/recovery.yaml').read_text())
    verify_release(root, expected)
    output = VOLUME / 'development/oracle_recovery' / expected / str(run_id) / mode
    assert_namespace(output)
    output.mkdir(parents=True, exist_ok=False)
    start_time = time.monotonic()
    resource_budget = dict(common['budget'])
    resource_budget.update(internal_stop_seconds=config['budget'][mode + '_internal_seconds'],
                           protected_export_seconds=config['budget'][mode + '_export_reserve_seconds'],
                           maximum_backbone_forwards=config['budget'][mode + '_maximum_forwards'])
    budget = Budget(resource_budget)
    actor, scratch, records = None, None, []
    result = {'status': 'running', 'mode': mode, 'source_commit': expected, 'run_id': str(run_id),
              'output': str(output), 'GPU_allocations': 0, 'held_model_evaluations': 0, 'optimizer_updates': 0}
    try:
        live = WorkspaceClient().jobs.get_run(run_id=int(run_id))
        task = next(t for t in live.tasks if t.task_key == 'oracle_policy_development')
        remaining = min(live.start_time, task.start_time) / 1000 + config['budget']['hard_job_timeout_seconds'] - time.time()
        if remaining < resource_budget['protected_export_seconds'] + 120:
            raise RuntimeError('Fresh job timing leaves insufficient export reserve')
        budget.config['internal_stop_seconds'] = min(budget.config['internal_stop_seconds'], remaining - 30)
        dump(output / 'FRESH_TIME_ADMISSION.json', {'remaining_seconds_at_entry': remaining, 'budget': budget.config})
        torch.set_num_threads(2)
        torch.manual_seed(config['seed'])
        if torch.cuda.is_available() or torch.__version__ != '2.11.0+cpu' or transformers.__version__ != '5.19.0':
            raise RuntimeError('Expected pinned CPU environment is unavailable')
        dump(output / 'SOFTWARE.json', {'python': sys.version, 'torch': torch.__version__,
                                       'transformers': transformers.__version__, 'numpy': np.__version__, 'device': 'cpu'})
        state = resources()
        dump(output / 'RESOURCE_PREFLIGHT.json', state)
        if state['available_memory_bytes'] < common['budget']['minimum_available_memory_bytes'] or state['free_local_disk_bytes'] < common['budget']['minimum_free_disk_bytes']:
            raise RuntimeError('Fresh CPU resource admission failed')
        specs, rows, meta = dataset(config)
        persist_dataset(spark, root, meta, expected)
        result['dataset_version'] = meta['version']
        dump(output / 'PRECALL_FREEZE.json', {'source_commit': expected, 'dataset_version': meta['version'],
                                            'dataset_sha256': meta['content_sha256'], 'config': config,
                                            'probe_run_id': str(probe_run), 'held_model_evaluations': 0})
        model_path, identity = ensure_model(str(VOLUME), common['model'])
        dump(output / 'MODEL_IDENTITY.json', identity)
        scratch = tempfile.TemporaryDirectory(prefix='research4-recovery-model-')
        for name in REQUIRED:
            shutil.copyfile(model_path / name, Path(scratch.name) / name)
        model_config = {**common['model'], **config['model']}
        actor = RecoveryActor(scratch.name, model_config, budget)
        if mode == 'probe':
            summary, records = probe(actor, rows, config, output)
        else:
            summary, records = qualify(actor, specs, rows, config, output, probe_run, expected)
        dump(output / 'RESULTS.json', summary)
        result.update(status='complete', **summary)
        result['base_fingerprint_before'] = actor.base_hash_before
        result['base_fingerprint_after'] = actor.base_fingerprint()
        if result['base_fingerprint_before'] != result['base_fingerprint_after']:
            raise RuntimeError('Original Qwen parameters changed')
        persist_results(spark, run_id, expected, mode, records)
        from pyspark.sql import functions as F
        retained = spark.table('workspace.research4.recovery_results').where(
            (F.col('run_id') == str(run_id)) & (F.col('source_commit') == expected))
        checks = {'expected_result_count': retained.count() == len(records),
                  'unique_condition_records': retained.select('condition', 'record_id').distinct().count() == len(records),
                  'result_JSON_SHA256_matches': retained.where(F.col('output_sha256').isNull() |
                      F.col('result_json').isNull() | (F.sha2('result_json', 256) != F.col('output_sha256'))).count() == 0}
        dump(output / 'DELTA_CHECKS.json', checks)
        if not all(checks.values()):
            raise ValueError('Recovery Delta evidence integrity check failed')
        if mode == 'qualification' and result['optimizer_updates']:
            from .recovery_registry import register
            registration = register(output / 'adapter.pt', root, expected, meta['version'])
            dump(output / 'MLFLOW_REGISTRY.json', registration)
            result['registry'] = registration
        storage.event(spark, run_id, expected, 'oracle_recovery_' + mode, result['status'], result, output)
    except Exception as error:
        result.update(status='failed', error_type=type(error).__name__, error=str(error))
        dump(output / 'FAILURE.json', {'traceback': traceback.format_exc(), 'result': result})
    finally:
        if actor is not None:
            result['optimizer_updates'] = getattr(actor, 'training_progress', {}).get('optimizer_steps', 0)
            if result['status'] == 'failed':
                save_torch(output / 'PARTIAL_ADAPTER.pt', {'lora': actor.adapter_state(), 'source_commit': expected})
                dump(output / 'PARTIAL_TRAINING_PROGRESS.json', getattr(actor, 'training_progress', {'optimizer_steps': 0}))
            persist_array_cache(output / 'FINAL_BASE_CACHE.npz', actor)
        actor = None
        gc.collect()
        if scratch is not None:
            scratch.cleanup()
        result.update(backbone_forwards=budget.forwards, execution_seconds=time.monotonic() - start_time, model_released=True)
        dump(output / 'RUN_RECEIPT.json', result)
        result['export'] = archive(output)
        dump(output.parent / (mode + '_EXPORT_RECEIPT.json'), result)
    return result

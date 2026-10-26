"""Frozen first-call spelling probe for ordinary Research4 record tasks.

This script is deliberately separate from the registered task/risk workers. Its
model is loaded by the *unchanged*, independently released task-development
source; the new source supplies only the selected training prompts and this
read-only comparison. No world action, optimizer, teacher continuation, or
official development/risk trajectory is run here.
"""
import argparse
import base64
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback
import zipfile


MODE = 'ordinary-record-entry-alias-frozen-probe-v1'
ENTRY_NAMES = ('catalog', 'list')
ARMS = (('base', False), ('always_on', True))
MAX_NEW_TOKENS = 160
RESPONSES = 48


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True).encode()).hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
        return digest.hexdigest()


def _load_sibling(name, path):
    """Load reviewed alias code beside an already imported original src tree."""
    import src
    path = Path(path).resolve()
    if path.name != name.rsplit('.', 1)[-1] + '.py' or not path.is_file():
        raise PermissionError('Expected reviewed alias module path')
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError('Cannot load the reviewed alias module')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def expected_preparation(rows, preparation, tokenizer, probe, saved_plan=None):
    """Reconstruct and bind all twelve original and renamed native prompts."""
    if (not isinstance(preparation, dict) or preparation.get('version') !=
            'ordinary-record-entry-alias-preparation-v1'
            or preparation.get('planned_neural_responses') != RESPONSES
            or preparation.get('sole_tool_schema_change') != {'from': 'catalog', 'to': 'list'}
            or preparation.get('fit_source_commit') !=
            '82cf3ecf41e799facd93b8c11e05c6d1f569599c'
            or preparation.get('checkpoint_sha256') !=
            '76ba55239a00e39e1dea982cea20083ec5536c3168b69e1754b756f2fc646b2f'):
        raise PermissionError('Prepared ordinary first-call comparison differs')
    chosen = probe.select_first_catalog_rows(rows)
    fixed = preparation.get('fixed_prompts')
    if not isinstance(fixed, list) or len(fixed) != 12 or len(chosen) != 12:
        raise PermissionError('Exactly twelve fixed training first calls required')
    plan = {} if saved_plan is None else {row['state_id']: row for row in saved_plan['records']}
    if saved_plan is not None and len(plan) != 1188:
        raise PermissionError('Complete original fitting token plan required')
    specs = probe.paired_tool_specs()
    if (canonical_sha(specs['catalog']) != preparation['original_tools_sha256']
            or canonical_sha(specs['list']) != preparation['alias_tools_sha256']):
        raise PermissionError('The two native schemas differ from their freeze')
    selected = []
    for index, (row, frozen) in enumerate(zip(chosen, fixed)):
        identifiers = {key: row[key] for key in ('case_id', 'group_id', 'state_id')}
        if (any(identifiers[key] != frozen.get(key) for key in identifiers)
                or frozen.get('training_only') is not True
                or frozen.get('target_tool_in_original_training') != 'catalog'
                or hashlib.sha256(row['text'].encode()).hexdigest() != frozen.get('ordinary_text_sha256')):
            raise PermissionError('A fixed training first-call identity or text changed')
        original = None if saved_plan is None else plan[row['state_id']]['input_token_ids']
        ids = probe.paired_prompt_token_ids(row['text'], tokenizer, original)
        if any(len(ids[name]) != frozen['original_token_count' if name == 'catalog'
                                                else 'alias_token_count']
               or canonical_sha(ids[name]) != frozen['original_input_ids_sha256' if name == 'catalog'
                                                       else 'alias_input_ids_sha256']
               for name in ENTRY_NAMES):
            raise PermissionError('The fixed native prompt token IDs changed')
        selected.append({'index': index, **identifiers, 'text_sha256': frozen['ordinary_text_sha256'],
                         'prompt_ids': ids})
    if (len({item['group_id'] for item in selected}) != 12
            or len({item['state_id'] for item in selected}) != 12):
        raise PermissionError('One distinct state from each of twelve training groups required')
    return selected


def generate_first_from_ids(actor, prompt_ids, adapter_enabled):
    """Mirror Actor.generate with caller-verified native IDs; no prompt editing.

    Actor.generate cannot accept the renamed tool schema. Both conditions take
    this same path, with the same deterministic 160-token parameters, cache
    boundary hook, adapter switch, tokenizer and EOS set as the frozen actor.
    """
    if (getattr(actor, 'task_evaluation_context', None) is None
            or type(adapter_enabled) is not bool
            or type(prompt_ids) is not list or not prompt_ids
            or any(type(token) is not int or token < 0 for token in prompt_ids)
            or actor.cfg['maximum_new_tokens'] != MAX_NEW_TOKENS
            or len(prompt_ids) > actor.cfg['maximum_tokens']):
        raise PermissionError('Only full-budget frozen task native prompts are admitted')
    import torch
    from src.composite_lora import enable
    from src.native27_tools import decode
    with actor.lock:
        if actor.generating:
            raise RuntimeError('Concurrent Qwen response')
        actor.clear_cache()
        enable(actor.adapters, adapter_enabled)
        tensor = torch.tensor([prompt_ids], device='cuda:0', dtype=torch.long)
        actor._trace = {'prefill_checked': False, 'initial_cache_length': None}
        actor.generating = True
        try:
            with torch.inference_mode():
                sequence = actor.model.generate(
                    input_ids=tensor, attention_mask=torch.ones_like(tensor),
                    past_key_values=None, use_cache=True, do_sample=False,
                    num_beams=1, num_return_sequences=1,
                    max_new_tokens=MAX_NEW_TOKENS, eos_token_id=sorted(actor.eos),
                    pad_token_id=(actor.tokenizer.pad_token_id if
                                  actor.tokenizer.pad_token_id is not None else min(actor.eos)),
                    logits_to_keep=1)
            if not actor._trace['prefill_checked']:
                raise RuntimeError('Native generation bypassed the cache-boundary hook')
            generated = sequence[0, len(prompt_ids):].cpu().tolist()
            ended = bool(generated and generated[-1] in actor.eos)
            return {'input_token_ids': list(prompt_ids),
                    'generated_token_ids': generated,
                    'finished_with_eos': ended,
                    'raw_text': decode(actor.tokenizer, generated, actor.eos),
                    'adapter_enabled': adapter_enabled,
                    'cache': dict(actor._trace),
                    'kernel_profile': actor.kernel_profile,
                    'native_interface_profile': actor.native_interface_profile,
                    'generation_parameters': {
                        'max_new_tokens': MAX_NEW_TOKENS, 'do_sample': False,
                        'num_beams': 1, 'num_return_sequences': 1,
                        'use_cache': True}}
        finally:
            actor.generating = False
            actor._trace = None


def native_parse_under_schema(raw_text, finished_with_eos, entry_name, probe):
    """Use the unchanged native parser with the condition's exact function map."""
    if entry_name not in ENTRY_NAMES or type(finished_with_eos) is not bool:
        raise ValueError('Exact condition and EOS state required')
    if not finished_with_eos:
        return {'valid_native_parse': False, 'actions': None,
                'error_type': 'UnterminatedGeneration'}
    from src.native27_tools import parse
    schemas = {item['function']['name']: item['function']['parameters']
               for item in probe.paired_tool_specs()[entry_name]}
    try:
        actions = parse(raw_text, maximum_calls=1, argument_schemas=schemas)
    except (ValueError, TypeError, RecursionError) as error:
        return {'valid_native_parse': False, 'actions': None,
                'error_type': type(error).__name__}
    return {'valid_native_parse': len(actions) == 1, 'actions': actions,
            'error_type': None}


def collect_first_calls(actor, selected, persist, probe, verify_checkpoint,
                        preparation_sha256, *, clock=time.monotonic, deadline=None,
                        generate=generate_first_from_ids):
    """Write each full raw response durably before another forward is allowed."""
    if (len(selected) != 12 or not callable(persist) or not callable(verify_checkpoint)
            or actor.cfg['maximum_new_tokens'] != MAX_NEW_TOKENS):
        raise PermissionError('Frozen first-call collection requires its exact bound')
    checkpoint_before = verify_checkpoint(actor)
    persist('CHECKPOINT_LIVE_BEFORE.json', checkpoint_before)
    completed = []
    attempted = 0
    uncertain = False
    try:
        for row in selected:
            for entry_name in ENTRY_NAMES:
                for arm, on in ARMS:
                    if deadline is not None and clock() >= deadline:
                        raise TimeoutError('Protected alias evidence deadline')
                    path = f"responses/{row['index']:02}/{entry_name}/{arm}"
                    pre = {'index': row['index'], 'state_id': row['state_id'],
                           'case_id': row['case_id'], 'group_id': row['group_id'],
                           'entry_name': entry_name, 'arm': arm, 'adapter_enabled': on,
                           'input_token_ids_sha256': canonical_sha(row['prompt_ids'][entry_name]),
                           'preparation_sha256': preparation_sha256,
                           'no_world_action': True, 'response_completion': 'unknown_until_raw_durable'}
                    persist(path + '/PRECALL.json', pre)
                    attempted += 1
                    native = generate(actor, row['prompt_ids'][entry_name], on)
                    if (native.get('adapter_enabled') is not on
                            or native.get('input_token_ids') != row['prompt_ids'][entry_name]
                            or type(native.get('generated_token_ids')) is not list
                            or len(native['generated_token_ids']) > MAX_NEW_TOKENS):
                        raise PermissionError('Native generation condition/IDs/budget differ')
                    classification = probe.classify_first_entry_call(
                        native['raw_text'], entry_name, native['finished_with_eos'])
                    native_parse = native_parse_under_schema(
                        native['raw_text'], native['finished_with_eos'], entry_name, probe)
                    record = {**pre, 'native': deepcopy(native),
                              'classification': classification,
                              'native_parse': native_parse,
                              'world_action_applied': False,
                              'official_task_or_risk_trajectory': False}
                    persist(path + '/NATIVE.json', record)
                    completed.append({'index': row['index'], 'state_id': row['state_id'],
                                      'entry_name': entry_name, 'arm': arm,
                                      'native_record_sha256': canonical_sha(record),
                                      'valid_entry_call': classification['valid_entry_call'],
                                      'raw_function_name': classification['raw_function_name'],
                                      'finished_with_eos': native['finished_with_eos']})
        actor.verify_frozen()
        checkpoint_after = verify_checkpoint(actor)
        persist('CHECKPOINT_LIVE_AFTER.json', checkpoint_after)
        if checkpoint_before['checkpoint_sha256'] != checkpoint_after['checkpoint_sha256']:
            raise PermissionError('Frozen checkpoint identity changed')
        if (len(completed) != RESPONSES
                or len({(r['index'], r['entry_name'], r['arm']) for r in completed}) != RESPONSES):
            raise PermissionError('Exactly 48 distinct response cells required')
        summary = {'version': MODE, 'status': 'complete_raw_first_call_probe',
                   'planned_responses': RESPONSES, 'attempted_responses': attempted,
                   'completed_durable_responses': len(completed),
                   'response_count_unknown': False, 'records': completed,
                   'valid_entry_calls': {name: {arm: sum(r['valid_entry_call'] for r in completed
                                                        if r['entry_name'] == name and r['arm'] == arm)
                                               for arm, _ in ARMS} for name in ENTRY_NAMES},
                   'prompt_text_sha256_distinct': len({r['text_sha256'] for r in selected}),
                   'checkpoint_sha256': checkpoint_before['checkpoint_sha256'],
                   'optimizer_updates': 0, 'world_actions': 0,
                   'teacher_forcing': False, 'official_task72_or_risk180_result': False}
        persist('TOOL_ALIAS_RESULT.json', summary)
        return summary
    except BaseException as error:
        # A failed generation may have completed remotely; no guessed count.
        uncertain = attempted > len(completed)
        failure = {'version': MODE, 'status': 'incomplete',
                   'planned_responses': RESPONSES, 'attempted_lower_bound': attempted,
                   'completed_durable_responses': len(completed),
                   'response_count_unknown': uncertain,
                   'error_type': type(error).__name__, 'error': str(error),
                   'optimizer_updates': 0, 'world_actions': 0,
                   'official_task72_or_risk180_result': False}
        persist('FAILURE.json', failure)
        raise
    finally:
        if len(completed) != RESPONSES:
            try:
                actor.verify_frozen()
                persist('CHECKPOINT_LIVE_AFTER.json', verify_checkpoint(actor))
            except BaseException as error:
                persist('CHECKPOINT_AFTER_UNCONFIRMED.json', {'error_type': type(error).__name__,
                                                               'error': str(error)})


class DurableOutput:
    """One-use, source-bound, readback-checked Kaggle raw export."""
    def __init__(self, destination, commit, maximum_bytes):
        self.root = Path(destination).resolve()
        base = Path('/kaggle/working/research4_campaign/runs').resolve()
        expected = base / commit / 'ordinary_record_entry_alias_v1'
        if (not re.fullmatch('[0-9a-f]{40}', commit)
                or self.root != expected.resolve() or type(maximum_bytes) is not int
                or maximum_bytes < 1024**2):
            raise PermissionError('Exact alias source/output ownership required')
        self.root.mkdir(parents=True, exist_ok=False)
        self.raw = self.root / 'raw'
        self.raw.mkdir()
        self.maximum_bytes = maximum_bytes
        self.confirmed = {}
        self.write('OUTPUT_OWNER.json', {'source_commit': commit,
                                         'path': str(self.root), 'version': MODE})

    def write(self, relative, value):
        from src.composite_runtime import contained
        data = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
        total = sum(p.stat().st_size for p in self.raw.rglob('*') if p.is_file())
        if total + len(data) > self.maximum_bytes:
            raise OSError('Full raw alias evidence bound reached')
        target = contained(self.raw, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if target.read_bytes() != data:
            raise OSError('Raw alias evidence readback differs')
        value_sha = hashlib.sha256(data).hexdigest()
        self.confirmed[relative] = value_sha
        return {'sha256': value_sha, 'durable': True}

    def finalize(self):
        archive = self.root / 'evidence.zip'
        members = {p.relative_to(self.raw).as_posix(): sha(p)
                   for p in sorted(self.raw.rglob('*')) if p.is_file()}
        if members != self.confirmed:
            raise PermissionError('Raw alias evidence inventory changed')
        with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED,
                             allowZip64=True) as stream:
            for name in members:
                stream.write(self.raw / name, name)
            stream.writestr('MANIFEST.json', json.dumps(members, sort_keys=True))
        with zipfile.ZipFile(archive) as stream:
            if (set(stream.namelist()) != set(members) | {'MANIFEST.json'}
                    or json.loads(stream.read('MANIFEST.json')) != members
                    or any(hashlib.sha256(stream.read(name)).hexdigest() != expected
                           for name, expected in members.items())):
                raise PermissionError('Readback alias archive differs from raw files')
        proof = {'version': MODE, 'raw_files': len(members), 'manifest_sha256': canonical_sha(members),
                 'evidence_sha256': sha(archive), 'evidence_bytes': archive.stat().st_size,
                 'raw_durable': True, 'archive_durable': True}
        target = self.root / 'EXPORT_RECEIPT.json'
        with target.open('x', encoding='utf-8') as stream:
            json.dump(proof, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        return proof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alias-root', required=True)
    parser.add_argument('--actor-root', required=True)
    parser.add_argument('--alias-commit', required=True)
    parser.add_argument('--actor-archive', required=True)
    args = parser.parse_args()
    alias_root = Path(args.alias_root).resolve()
    actor_root = Path(args.actor_root).resolve()
    # The established evaluator's code, package and guards own the actor. The
    # alias module is loaded separately only after its complete release checks.
    sys.path.insert(0, str(actor_root))
    from src.composite_runtime import require_scientific_runtime
    from src.composite_source_archive import verify_transport
    from src.io_utils import read
    from src.composite_task_skill_inference import (MODE as ACTOR_MODE,
        validate_source as validate_actor_source, authorized_load_plan,
        verify_frozen_checkpoint)
    from src.composite_task_skill_fit_registry import registered_fit
    from src.composite_task_skill_training import read_data
    require_scientific_runtime()
    if os.environ.get('RESEARCH4_EXECUTION_PLATFORM') != 'kaggle' or os.environ.get('CUDA_VISIBLE_DEVICES') != '0,1':
        raise PermissionError('Exact selected Kaggle two-T4 child required')
    actor_commit = read(actor_root / 'RELEASE.json')['commit']
    verify_transport(actor_root, actor_commit, ACTOR_MODE, 'TASK_TRAINING_PACKAGE_FILES.json')
    verify_transport(alias_root, args.alias_commit, MODE, 'TOOL_ALIAS_PACKAGE_FILES.json')
    probe = _load_sibling('src.composite_tool_alias_probe', alias_root / 'src/composite_tool_alias_probe.py')
    profile = _load_sibling('src.composite_tool_alias_profile', alias_root / 'src/composite_tool_alias_profile.py')
    fit = Path(os.environ['RESEARCH4_TASK_FULL_FIT_ARCHIVE']).resolve()
    qualification = Path(os.environ['RESEARCH4_STATEFUL_QUALIFICATION_ARCHIVE']).resolve()
    context = validate_actor_source(actor_root, actor_commit, fit, qualification)
    alias_context = profile.validate_source(alias_root, args.alias_commit, fit, qualification,
                                            actor_root, Path(args.actor_archive).resolve())
    stage = read(alias_root / 'inputs/TOOL_ALIAS_STAGE.json')
    preparation = read(alias_root / 'inputs/OFFLINE_PREPARATION.json')
    if (stage['actor_source_commit'] != actor_commit
            or stage['checkpoint_sha256'] != context.model.receipt()['checkpoint_sha256']
            or stage['fit_source_commit'] != context.model.receipt()['fit_source_commit']
            or stage.get('maximum_new_tokens') != MAX_NEW_TOKENS
            or stage.get('expected_responses') != RESPONSES
            or stage.get('optimizer_updates') != 0):
        raise PermissionError('Aliased prompt and original frozen actor lineages differ')
    writer = DurableOutput(stage['destination'], args.alias_commit,
                           stage['maximum_raw_evidence_bytes'])
    start = time.monotonic()
    actor = None
    result = None
    forward_counts = {'attempted': 0, 'completed': 0}
    reserve = None
    cache = None
    try:
        writer.write('PRECALL_FREEZE.json', {'alias_commit': args.alias_commit,
            'actor_commit': actor_commit, 'stage_sha256': sha(alias_root / 'inputs/TOOL_ALIAS_STAGE.json'),
            'preparation_sha256': sha(alias_root / 'inputs/OFFLINE_PREPARATION.json'),
            'fit_archive_sha256': sha(fit), 'qualification_archive_sha256': sha(qualification),
            'checkpoint_sha256': stage['checkpoint_sha256'],
            'planned_responses': RESPONSES, 'no_world_action': True})
        import torch
        from src.composite_frozen_probe import verify_runtime
        from src.composite_store import download_model
        from src.composite_actor import Actor
        from src.composite_generalized_interface import PROFILE
        from src.composite_task_skill_training import live_resources
        from src.composite_v2_model_audit import capture_live_backbone
        from src.composite_frozen_worker import transport_environment
        from src.composite_v2_raw_exports import disk_admission
        load = authorized_load_plan(context, read(actor_root / 'config/composite_controller_rank32_v2.json')['qwen'])
        writer.write('HARDWARE_PREFLIGHT.json', verify_runtime(load))
        cache = Path('/kaggle/temp/research4_tool_alias_cache') / args.alias_commit
        cache.mkdir(parents=True, exist_ok=False)
        (cache / 'scratch').mkdir()
        os.environ.update(transport_environment(cache))
        writer.write('DISK_ADMISSION.json', disk_admission(cache, writer.root, load['model_pin'],
            context.stage(), expected_profile=context.stage()['raw_evidence_profile']))
        if not hasattr(os, 'posix_fallocate'):
            raise RuntimeError('Physical export reservation requires the qualified Linux filesystem')
        reserve = writer.root / 'EXPORT_RESERVE.bin'
        with reserve.open('xb') as stream:
            os.posix_fallocate(stream.fileno(), 0, stage['physical_export_reserve_bytes'])
            stream.flush()
            os.fsync(stream.fileno())
        model_metadata_path, metadata_file_proof = download_model(load['model_pin'], cache,
                                                                   metadata_only=True)
        from src.composite_frozen_probe import verify_metadata_bytes
        encoded_metadata = {item['name']: base64.b64encode(
            (model_metadata_path / item['name']).read_bytes()).decode()
            for item in load['model_pin']['files'] if not item['name'].endswith('.safetensors')}
        verify_metadata_bytes(load['model_pin'], metadata_file_proof, encoded_metadata)
        writer.write('MODEL_METADATA_BYTES.json', encoded_metadata)
        writer.write('METADATA_FILE_VERIFICATION.json', metadata_file_proof)
        model_path, model_file_proof = download_model(load['model_pin'], cache)
        writer.write('MODEL_FILE_VERIFICATION.json', model_file_proof)
        def guard():
            if time.monotonic() - start >= stage['worker_seconds'] - stage['export_reserve_seconds']:
                raise TimeoutError('Protected alias worker export deadline')
            live_resources(torch.cuda)
        def on_forward(event):
            if event == 'attempt':
                if forward_counts['attempted'] >= stage['maximum_backbone_forwards']:
                    raise TimeoutError('Alias stage forward ceiling')
                forward_counts['attempted'] += 1
            elif event == 'complete':
                forward_counts['completed'] += 1
            else:
                raise ValueError('Unexpected forward event')
        actor = Actor(model_path, read(actor_root / 'config/composite_controller_rank32_v2.json')['qwen'],
                      guard, on_forward, native_interface_profile=PROFILE,
                      task_evaluation=context)
        writer.write('MODEL_LOADING_INFO.json', actor.frozen_loading_proof)
        writer.write('CHECKPOINT_RESTORE.json', actor.task_checkpoint_proof)
        writer.write('BACKBONE_LIVE_BEFORE.json', capture_live_backbone(actor))
        from src.composite_v2_model_audit import EvidenceFiles
        from src.composite_task_skill_package import require_frozen_task_package
        data = read_data(base64.b64decode((actor_root / 'inputs/TASK_DATA_EVIDENCE.r4b64').read_bytes(),
                                          validate=True))
        rows = data.rows()
        model_value = require_frozen_task_package(context.model)
        files = EvidenceFiles(fit, model_value['fit_raw_archive_sha256'], 4 * 1024**3)
        try:
            token_plan = files.json('TRAINING_TOKEN_PLAN.json')
            if files.manifest['TRAINING_TOKEN_PLAN.json'] != stage['original_training_token_plan_sha256']:
                raise PermissionError('Original training token plan differs')
        finally:
            files.close()
        selected = expected_preparation(rows, preparation, actor.tokenizer, probe, token_plan)
        if (stage.get('offline_preparation_sha256') !=
                sha(alias_root / 'inputs/OFFLINE_PREPARATION.json')
                or [{key: row[key] for key in ('index', 'state_id', 'case_id', 'group_id')}
                    for row in selected] !=
                   [{key: row[key] for key in ('index', 'state_id', 'case_id', 'group_id')}
                    for row in stage['selected_rows']]):
            raise PermissionError('Live selected training rows differ from registered stage')
        result = collect_first_calls(actor, selected, writer.write, probe,
                   lambda value: verify_frozen_checkpoint(value, context),
                   sha(alias_root / 'inputs/OFFLINE_PREPARATION.json'),
                   deadline=start + stage['worker_seconds'] - stage['export_reserve_seconds'])
        writer.write('BACKBONE_LIVE_AFTER.json', capture_live_backbone(actor))
        result['status'] = 'complete_raw_first_call_probe'
    except BaseException as error:
        result = None
        writer.write('WORKER_FAILURE.json', {'status': 'failed', 'error_type': type(error).__name__,
                                             'error': str(error), 'traceback': traceback.format_exc(),
                                             'response_count_unknown': True})
        raise
    finally:
        writer.write('FORWARD_COUNTS.json', {'attempted': forward_counts['attempted'],
            'completed': forward_counts['completed'],
            'exact_at_controlled_exit': True,
            'native_responses_completed_durable': sum(
                name.endswith('/NATIVE.json') for name in writer.confirmed),
            'response_count_unknown': any(
                name.endswith('/PRECALL.json') and name[:-len('PRECALL.json')] + 'NATIVE.json'
                not in writer.confirmed for name in writer.confirmed)})
        actor = None
        import gc
        gc.collect()
        try:
            import torch
            for index in range(min(2, torch.cuda.device_count())):
                torch.cuda.synchronize(index)
                with torch.cuda.device(index):
                    torch.cuda.empty_cache()
        except BaseException:
            pass
        if reserve is not None and reserve.exists():
            if reserve.resolve() != (writer.root / 'EXPORT_RESERVE.bin').resolve():
                raise PermissionError('Export reserve ownership changed')
            reserve.unlink()
        cache_cleanup = {'owned_download_cache_removed': cache is None or not cache.exists()}
        if cache is not None and cache.exists():
            try:
                expected_cache = (Path('/kaggle/temp/research4_tool_alias_cache') /
                                  args.alias_commit).resolve()
                if (cache.resolve() != expected_cache or expected_cache.parent !=
                        Path('/kaggle/temp/research4_tool_alias_cache').resolve()):
                    raise PermissionError('Alias model cache cleanup ownership changed')
                import shutil
                shutil.rmtree(cache)
                cache_cleanup = {'owned_download_cache_removed': not cache.exists()}
            except BaseException as error:
                cache_cleanup = {'owned_download_cache_removed': False,
                                 'error_type': type(error).__name__, 'error': str(error)}
        writer.write('RUN_RECEIPT.json', {'status': 'complete' if result is not None else 'failed',
            'worker_exiting': True, 'optimizer_updates': 0, 'world_actions': 0,
            'official_task72_or_risk180_result': False,
            'provider_compute_release_verified': False,
            'cache_cleanup': cache_cleanup,
            'forward_counts': forward_counts,
            'completed_durable_responses': sum(
                name.endswith('/NATIVE.json') for name in writer.confirmed),
            'response_count_unknown': any(
                name.endswith('/PRECALL.json') and name[:-len('PRECALL.json')] + 'NATIVE.json'
                not in writer.confirmed for name in writer.confirmed),
            'elapsed_seconds': time.monotonic() - start})
        export = writer.finalize()
        print(json.dumps({'result': result, 'export': export}), flush=True)


if __name__ == '__main__':
    main()

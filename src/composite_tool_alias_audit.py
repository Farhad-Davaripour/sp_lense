"""Independent source, checkpoint, tokenizer and complete raw alias audit.

This reader never executes a model, a task-world action or an optimizer.
External notebook success and GPU release must be independently reconciled.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from .io_utils import digest, read, sha
from .composite_task_skill_source_package import frozen_source_package
from .composite_tool_alias_profile import MODE, NOTEBOOK, RESPONSES, notebook_code, validate_source
from .composite_tool_alias_probe import (classify_first_entry_call,
                                         paired_prompt_token_ids, paired_tool_specs)
from .composite_v2_model_audit import EvidenceFiles, frozen_integrity

_KEY = object()


def canonical_sha(value):
    """Match the worker's JSON record digest exactly, including Unicode escaping."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True).encode('utf-8')).hexdigest()


class AliasAudit:
    def __init__(self, value, key=None):
        if key is not _KEY:
            raise PermissionError('Complete original alias evidence is required')
        self._key = key
        self._value = deepcopy(value)

    def receipt(self):
        return deepcopy(self._value)


class AliasKaggleClosure:
    def __init__(self, value, key=None):
        if key is not _KEY:
            raise PermissionError('Independent exact provider readback is required')
        self._key = key
        self._value = deepcopy(value)

    def receipt(self):
        return deepcopy(self._value)


def require_complete_alias(value):
    if not isinstance(value, AliasAudit) or value._key is not _KEY:
        raise PermissionError('Genuine complete alias raw/source/provider audit required')
    result = value.receipt()
    if (result.get('raw_evidence_audited') is not True
            or result.get('owned_compute_released') is not True
            or result.get('completed_responses') != RESPONSES
            or result.get('missing_or_unverified') != []
            or result.get('optimizer_updates') != 0 or result.get('world_actions') != 0
            or result.get('scientific_validation_admitted') is not False
            or result.get('heldout_content_read') is not False):
        raise PermissionError('Complete raw-audited and released diagnostic required')
    return result


def kaggle_closure(binding, archive_sha256, terminal_ax, release_ax,
                   controller, actual_code_cells, prepared_code_cells):
    """Reconcile a specific successful notebook version and observed release."""
    run = binding.get('scriptVersionId')
    commit = binding.get('source_commit')
    if (type(run) is not int or run <= 0
            or binding.get('mode') != MODE or binding.get('phase') != 'ordinary_entry_alias_diagnostic'
            or binding.get('notebook') != NOTEBOOK
            or not re.fullmatch('[0-9a-f]{40}', commit or '')
            or not re.fullmatch('[0-9a-f]{64}', archive_sha256 or '')):
        raise PermissionError('Exact one-use alias submission identity required')
    if ('scriptVersionId=' + str(run) not in terminal_ax or NOTEBOOK not in terminal_ax
            or not re.search(r'successfully ran after|second run - successful', terminal_ax)):
        raise PermissionError('Exact provider terminal success is unverified')
    if '0  Active Events' not in release_ax or 'Draft Session off' not in release_ax:
        raise PermissionError('Actual Kaggle event and draft shutdown must be observed')
    if (controller.get('source_commit') != commit or controller.get('version') != MODE
            or controller.get('status') != 'worker_exited' or controller.get('exit_code') != 0
            or controller.get('child_exit_confirmed') is not True
            or controller.get('timed_out') is not False
            or controller.get('automatic_scientific_retries') != 0
            or controller.get('optimizer_updates') != 0 or controller.get('world_actions') != 0):
        raise PermissionError('Exact controlled alias worker exit differs')
    if not actual_code_cells or digest(actual_code_cells) != digest(prepared_code_cells):
        raise PermissionError('Submitted alias notebook differs from its exact prepared code')
    return AliasKaggleClosure({'scriptVersionId': run, 'source_commit': commit,
                               'notebook': NOTEBOOK, 'archive_sha256': archive_sha256,
                               'terminal_ax_sha256': hashlib.sha256(terminal_ax.encode()).hexdigest(),
                               'release_ax_sha256': hashlib.sha256(release_ax.encode()).hexdigest(),
                               'controller_sha256': digest(controller),
                               'actual_code_cells_sha256': digest(actual_code_cells),
                               'owned_compute_released': True}, _KEY)


def _cell_path(index, condition, arm):
    return f'responses/{index:02}/{condition}/{arm}'


def require_worker_exit(receipt):
    """A complete response roster does not waive owned model-cache cleanup."""
    if (receipt.get('status') != 'complete' or receipt.get('worker_exiting') is not True
            or receipt.get('optimizer_updates') != 0 or receipt.get('world_actions') != 0
            or receipt.get('official_task72_or_risk180_result') is not False
            or receipt.get('provider_compute_release_verified') is not False
            or receipt.get('completed_durable_responses') != RESPONSES
            or receipt.get('response_count_unknown') is not False
            or receipt.get('cache_cleanup') != {'owned_download_cache_removed': True}):
        raise PermissionError('Exact complete controlled alias exit and cache cleanup required')
    return receipt


def _native_parse_under_schema(raw_text, finished_with_eos, condition):
    if not finished_with_eos:
        return {'valid_native_parse': False, 'actions': None,
                'error_type': 'UnterminatedGeneration'}
    from .native27_tools import parse
    schemas = {item['function']['name']: item['function']['parameters']
               for item in paired_tool_specs()[condition]}
    try:
        actions = parse(raw_text, maximum_calls=1, argument_schemas=schemas)
    except (ValueError, TypeError, RecursionError) as error:
        return {'valid_native_parse': False, 'actions': None,
                'error_type': type(error).__name__}
    return {'valid_native_parse': len(actions) == 1, 'actions': actions,
            'error_type': None}


def audit_collection(raw_archive, raw_sha256, alias_root, alias_commit,
                     fit_archive, qualification_archive, actor_archive, closure=None):
    """Recompute all 48 native first-response cells from raw token evidence."""
    actor_archive = Path(actor_archive)
    with frozen_source_package(actor_archive.read_bytes(), sha(actor_archive),
                               read(Path(alias_root) / 'inputs/TOOL_ALIAS_STAGE.json')['actor_source_commit']) as (actor_root, _):
        context = validate_source(alias_root, alias_commit, fit_archive,
                                  qualification_archive, actor_root, actor_archive)
        stage = context['stage']
        model = context['model'].receipt()
        result = {'reader': 'ordinary-entry-alias-source-native-raw-v1',
                  'source_commit': alias_commit, 'actor_source_commit': stage['actor_source_commit'],
                  'raw_archive_sha256': raw_sha256,
                  'checkpoint_sha256': model['checkpoint_sha256'],
                  'selected_rows_sha256': stage['selected_rows_sha256'],
                  'matched_training_states': 12,
                  'distinct_visible_prompt_texts': len({r['text_sha256']
                                                       for r in stage['selected_rows']}),
                  'expected_responses': RESPONSES, 'completed_responses': None,
                  'raw_evidence_audited': False, 'owned_compute_released': None,
                  'optimizer_updates': 0, 'world_actions': 0,
                  'new_model_forwards': 0, 'new_optimizer_updates': 0,
                  'scientific_validation_admitted': False, 'full_task72_complete': False,
                  'risk180_complete': False, 'heldout_content_read': False,
                  'interpretation_limit': ('The 12 fixed group states contain 11 distinct '
                                           'visible prompt texts, so they are not 12 independent '
                                           'text trials. Only the entry function name changes; '
                                           'other tool descriptions retain original catalog '
                                           'wording. First-call validity does not establish '
                                           'task completion or a population effect.'),
                  'missing_or_unverified': []}
        files = EvidenceFiles(raw_archive, raw_sha256, stage['maximum_raw_evidence_bytes'])
        try:
            if any(name.endswith('.pt') for name in files.manifest):
                raise PermissionError('Alias inference exports no fitting tensors')
            required = {'OUTPUT_OWNER.json', 'PRECALL_FREEZE.json', 'RUN_RECEIPT.json',
                        'TOOL_ALIAS_RESULT.json', 'CHECKPOINT_LIVE_BEFORE.json',
                        'CHECKPOINT_LIVE_AFTER.json', 'BACKBONE_LIVE_BEFORE.json',
                        'BACKBONE_LIVE_AFTER.json', 'MODEL_FILE_VERIFICATION.json',
                        'MODEL_LOADING_INFO.json', 'CHECKPOINT_RESTORE.json',
                        'HARDWARE_PREFLIGHT.json', 'DISK_ADMISSION.json',
                        'FORWARD_COUNTS.json', 'MODEL_METADATA_BYTES.json',
                        'METADATA_FILE_VERIFICATION.json'}
            for index in range(12):
                for condition in ['catalog', 'list']:
                    for arm in ['base', 'always_on']:
                        stem = _cell_path(index, condition, arm)
                        required.update([stem + '/PRECALL.json', stem + '/NATIVE.json'])
            if not required <= set(files.manifest):
                result['missing_or_unverified'] = sorted(required - set(files.manifest))
                return AliasAudit(result, _KEY)
            if set(files.manifest) - required:
                raise PermissionError('Unexpected raw evidence or incomplete native response roster')
            if files.json('OUTPUT_OWNER.json') != {
                'source_commit': alias_commit, 'path': stage['destination'], 'version': MODE}:
                raise PermissionError('Raw output belongs to another source/namespace')
            freeze = files.json('PRECALL_FREEZE.json')
            expected_freeze = {'alias_commit': alias_commit,
                               'actor_commit': stage['actor_source_commit'],
                               'stage_sha256': sha(Path(alias_root) / 'inputs/TOOL_ALIAS_STAGE.json'),
                               'preparation_sha256': stage['offline_preparation_sha256'],
                               'fit_archive_sha256': sha(fit_archive),
                               'qualification_archive_sha256': sha(qualification_archive),
                               'checkpoint_sha256': stage['checkpoint_sha256'],
                               'planned_responses': RESPONSES, 'no_world_action': True}
            if digest(freeze) != digest(expected_freeze):
                raise PermissionError('Pre-call frozen source/model/data binding differs')
            receipt = files.json('RUN_RECEIPT.json')
            require_worker_exit(receipt)
            forwards = files.json('FORWARD_COUNTS.json')
            if (type(forwards.get('attempted')) is not int
                    or type(forwards.get('completed')) is not int
                    or not 0 < forwards['attempted'] == forwards['completed']
                       <= stage['maximum_backbone_forwards']
                    or forwards.get('exact_at_controlled_exit') is not True
                    or forwards.get('native_responses_completed_durable') != RESPONSES
                    or forwards.get('response_count_unknown') is not False
                    or receipt.get('forward_counts') != {
                        'attempted': forwards['attempted'], 'completed': forwards['completed']}):
                raise PermissionError('Exact bounded native forward and response counts required')
            from .composite_frozen_probe import (verified_model_context, validate_download_proof,
                                                 verify_software)
            from .composite_task_skill_training import adapter_layout
            from .composite_task_skill_evaluation_audit import _checkpoint, _tokenizer
            cfg = read(Path(actor_root) / 'config/composite_controller_rank32_v2.json')['qwen']
            load = verified_model_context(actor_root, stage['actor_source_commit'], cfg,
                                          context['actor'].stage(), stage['actor_stage_sha256'])
            hardware = files.json('HARDWARE_PREFLIGHT.json')
            software = hardware.get('software', {})
            if software != verify_software(software.get('torch'), software.get('packages', {})):
                raise PermissionError('Pinned two-T4 software differs')
            devices_raw = hardware.get('devices')
            if (not isinstance(devices_raw, list) or len(devices_raw) != 2
                    or {item.get('index') for item in devices_raw} != {0, 1}):
                raise PermissionError('Both original model devices must be observed')
            for device in devices_raw:
                minimum = load['metadata_bounds']['device_weight_lower_bounds_bytes'][device['index']]
                if (device.get('name') != 'Tesla T4'
                        or device.get('optimistic_weight_lower_bound_bytes') != minimum
                        or type(device.get('free_bytes')) is not int
                        or device['free_bytes'] < minimum + 2 * 1024**3):
                    raise PermissionError('Actual per-device model admission differs')
            disk = files.json('DISK_ADMISSION.json')
            actor_stage = context['actor'].stage()
            if (disk.get('download_file_bytes') != 55575985859
                    or disk.get('raw_evidence_profile') != actor_stage['raw_evidence_profile']
                    or disk.get('maximum_raw_evidence_bytes') !=
                       actor_stage['maximum_raw_evidence_bytes']
                    or disk.get('physical_export_reserve_bytes') !=
                       actor_stage['physical_export_reserve_bytes']
                    or disk.get('runtime_capacity_admitted') is not False):
                raise PermissionError('Original model and export disk envelope differs')
            validate_download_proof(load, files.json('MODEL_FILE_VERIFICATION.json'))
            from .composite_frozen_probe import verify_metadata_bytes
            verify_metadata_bytes(load['model_pin'], files.json('METADATA_FILE_VERIFICATION.json'),
                                  files.json('MODEL_METADATA_BYTES.json'))
            layout = adapter_layout(load['metadata'], load['plan'])
            for name in ['CHECKPOINT_LIVE_BEFORE.json', 'CHECKPOINT_LIVE_AFTER.json']:
                _checkpoint(files.json(name), model, layout)
            loading = files.json('MODEL_LOADING_INFO.json')
            if (loading.get('missing_keys') != [] or loading.get('mismatched_keys') != []
                    or loading.get('error_msgs') != []
                    or not isinstance(loading.get('ignored_checkpoint_only_MTP_keys'), list)
                    or any(not name.startswith('mtp.')
                           for name in loading['ignored_checkpoint_only_MTP_keys'])):
                raise PermissionError('Original Qwen model loader reported missing or mismatched weights')
            restore = files.json('CHECKPOINT_RESTORE.json')
            expected_restore = {'checkpoint_sha256': model['checkpoint_sha256'],
                                'live_tensor_hashes': model['checkpoint_tensor_hashes'],
                                'tensors': 992, 'all_adapter_requires_grad': False,
                                'optimizer_state_loaded': False, 'fitting_allowed': False}
            if digest(restore) != digest(expected_restore):
                raise PermissionError('Frozen full adapter restoration differs')
            if digest(files.json('CHECKPOINT_LIVE_BEFORE.json')) != digest(files.json('CHECKPOINT_LIVE_AFTER.json')):
                raise PermissionError('Frozen adapter changed during first-call comparison')
            originals = {name: value['shape'] for name, value in load['metadata']['tensors'].items()
                         if not name.startswith('mtp.')}
            from .composite_placement import parameter_device
            devices = {name: 'cuda:' + str(parameter_device(name, load['plan'])) for name in originals}
            frozen_integrity(files.json('BACKBONE_LIVE_BEFORE.json'),
                             files.json('BACKBONE_LIVE_AFTER.json'), originals,
                             parameter_devices=devices)
            from .native27_tools import decode
            # Reconstruct this run's checksum-verified tokenizer metadata and
            # independently replay both complete prompts without Qwen forwards.
            with _tokenizer(files, load) as (tokenizer, eos, vocabulary):
                    records = []
                    for index, row in enumerate(context['selected_rows']):
                        selected = stage['selected_rows'][index]
                        if (row['state_id'] != selected['state_id']
                                or row['case_id'] != selected['case_id']
                                or row['group_id'] != selected['group_id']):
                            raise PermissionError('Selected training row identity changed')
                        ids = paired_prompt_token_ids(row['text'], tokenizer)
                        for condition in ['catalog', 'list']:
                            if digest(ids[condition]) != selected[condition + '_input_token_ids_sha256']:
                                raise PermissionError('Native original or alias prompt tokens changed')
                            for arm, enabled in [('base', False), ('always_on', True)]:
                                stem = _cell_path(index, condition, arm)
                                pre = files.json(stem + '/PRECALL.json')
                                expected_pre = {'index': index, 'state_id': row['state_id'],
                                                'case_id': row['case_id'], 'group_id': row['group_id'],
                                                'entry_name': condition, 'arm': arm,
                                                'adapter_enabled': enabled,
                                                'input_token_ids_sha256': canonical_sha(ids[condition]),
                                                'preparation_sha256': stage['offline_preparation_sha256'],
                                                'no_world_action': True,
                                                'response_completion': 'unknown_until_raw_durable'}
                                if digest(pre) != digest(expected_pre):
                                    raise PermissionError('Pre-call prompt/arm freeze differs')
                                record = files.json(stem + '/NATIVE.json')
                                native = record.get('native')
                                if (not isinstance(native, dict)
                                        or native.get('input_token_ids') != ids[condition]
                                        or native.get('adapter_enabled') is not enabled
                                        or native.get('kernel_profile') != 'reference'
                                        or native.get('generation_parameters') != {
                                            'max_new_tokens': 160, 'do_sample': False,
                                            'num_beams': 1, 'num_return_sequences': 1,
                                            'use_cache': True}
                                        or native.get('native_interface_profile') !=
                                           'composite-generalized-native-v2'):
                                    raise PermissionError('Native generation input, arm, or decoder differs')
                                generated = native.get('generated_token_ids')
                                if (not isinstance(generated, list) or not 1 <= len(generated) <= 160
                                        or any(type(v) is not int or not 0 <= v < vocabulary
                                               for v in generated)):
                                    raise PermissionError('Complete bounded raw generated token IDs required')
                                ended = generated[-1] in eos
                                if (native.get('finished_with_eos') is not ended
                                        or native.get('raw_text') != decode(tokenizer, generated, eos)
                                        or native.get('cache', {}).get('prefill_checked') is not True
                                        or native['cache'].get('initial_cache_length') != 0):
                                    raise PermissionError('Native EOS, tokenizer decode or cache boundary differs')
                                classification = classify_first_entry_call(native['raw_text'], condition, ended)
                                native_parse = _native_parse_under_schema(native['raw_text'], ended,
                                                                           condition)
                                if (classification['valid_entry_call']
                                        and (native_parse['valid_native_parse'] is not True
                                             or native_parse['actions'] != [
                                                 {'tool': condition, 'arguments': {}}])):
                                    raise PermissionError('A counted entry call failed the native parser')
                                if (digest(record) != digest({**pre, 'native': native,
                                        'classification': classification,
                                        'native_parse': native_parse,
                                        'world_action_applied': False,
                                        'official_task_or_risk_trajectory': False})):
                                    raise PermissionError('Raw classification or no-world-action record differs')
                                records.append({'index': index, 'state_id': row['state_id'],
                                                'entry_name': condition, 'arm': arm,
                                                'native_record_sha256': canonical_sha(record),
                                                'valid_entry_call': classification['valid_entry_call'],
                                                'raw_function_name': classification['raw_function_name'],
                                                'finished_with_eos': ended})
            summary = files.json('TOOL_ALIAS_RESULT.json')
            expected_counts = {name: {arm: sum(r['valid_entry_call'] for r in records
                                              if r['entry_name'] == name and r['arm'] == arm)
                                      for arm in ['base', 'always_on']}
                               for name in ['catalog', 'list']}
            expected_summary = {'version': MODE, 'status': 'complete_raw_first_call_probe',
                                'planned_responses': RESPONSES,
                                'attempted_responses': RESPONSES,
                                'completed_durable_responses': RESPONSES,
                                'response_count_unknown': False, 'records': records,
                                'valid_entry_calls': expected_counts,
                                'prompt_text_sha256_distinct': len({r['text_sha256']
                                                                     for r in stage['selected_rows']}),
                                'checkpoint_sha256': model['checkpoint_sha256'],
                                'optimizer_updates': 0, 'world_actions': 0,
                                'teacher_forcing': False,
                                'official_task72_or_risk180_result': False}
            if digest(summary) != digest(expected_summary):
                raise PermissionError('Complete 48-cell summary differs from recomputed raw outputs')
            closed = (isinstance(closure, AliasKaggleClosure) and closure._key is _KEY
                      and closure._value.get('source_commit') == alias_commit
                      and closure._value.get('notebook') == NOTEBOOK
                      and closure._value.get('archive_sha256') == raw_sha256
                      and closure._value.get('owned_compute_released') is True)
            expected_code = notebook_code(alias_root, alias_commit,
                                          sha(Path(alias_root).parent / 'TOOL_ALIAS_PACKAGE_FILES.json'))
            if closed and closure._value['actual_code_cells_sha256'] != digest([expected_code]):
                raise PermissionError('Released provider version used another notebook source')
            result.update(completed_responses=RESPONSES, valid_entry_calls=expected_counts,
                          response_records=records,
                          prompt_text_sha256_distinct=expected_summary['prompt_text_sha256_distinct'],
                          recorded_forward_counts=forwards,
                          provider_run_id=closure._value['scriptVersionId'] if closed else None,
                          owned_compute_released=True if closed else None,
                          raw_evidence_audited=closed,
                          missing_or_unverified=[] if closed else
                          ['independent_exact_provider_release_and_readback'])
            return AliasAudit(result, _KEY)
        finally:
            files.close()

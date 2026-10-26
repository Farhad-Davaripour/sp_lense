"""Preparation and narrow first-call evidence for the bounded V4 study.

This module registers no scientific runtime, data generator or fitting command.
An entry-name decision never grants training, full-task or held-out admission.
"""
from copy import deepcopy
from pathlib import Path

from .io_utils import read, sha
from .composite_tool_alias_audit import require_complete_alias
from .composite_world import CONDITIONS

VERSION = 'bounded-fictional-behavior-v4-preparation'
GATE_SHA = 'beffd5b42f3834a0e48aba97c35cdebf49a611d11cdb3321a95f4fbb90255016'
ACCEPTANCE_SHA = '595d6d7d4b2426f4323022febc51546b953b83f38b9ac04f436325a81e647572'
ALIAS_SOURCE = 'f8cd823c791f7432d24cf012e1068016d786d7ba'
CHECKPOINT_SHA = '76ba55239a00e39e1dea982cea20083ec5536c3168b69e1754b756f2fc646b2f'
SELECTOR = 'training_only_nonterminal_v4_teacher_targets_all_registered_phases'
_KEY = object()


def validate_preparation(value):
    """Reject scope/capacity drift without constructing a runtime permit."""
    cfg = deepcopy(value)
    arch, recipe, world = cfg['architecture'], cfg['training'], cfg['world']
    if (cfg['version'] != VERSION or cfg['scientific_runtime_admitted'] is not False
            or cfg['platform'] != 'kaggle' or cfg['account'] != 'farhad89'
            or arch['qwen_blocks'] != 64 or arch['detector_blocks'] != 22
            or arch['rank'] != 32 or arch['alpha'] != 64
            or arch['projections'] != 496 or arch['adapter_parameters'] != 233455616
            or recipe['updates'] != 64 or recipe['accumulation'] != 2
            or recipe['seed'] != 202610212 or recipe['learning_rate'] != 0.0005
            or recipe['weight_decay'] != 0 or recipe['clip_norm'] != 1
            or recipe['reward_optimization'] is not False
            or recipe['eligibility'] != SELECTOR
            or world['initial_fictional_deadline'] != 20 or world['external_hard_tick'] != 24
            or world['host_capabilities'] is not False
            or world['every_primitive_external_stop_check'] is not True
            or world['every_write_external_permission_check'] is not True
            or world['primitive_tick_discounts'] is not False
            or cfg['fresh_initialization'] is not True
            or cfg['qualifier_initializes_full_fit'] is not False
            or cfg['behavioral_data_materialized'] is not False
            or cfg['training_launch_admitted'] is not False
            or cfg['automatic_scientific_retries'] != 0
            or cfg['maximum_concurrent_owned_workers'] != 1
            or cfg['cohorts']['conditions'] != list(CONDITIONS)
            or cfg['cohorts']['training_groups'] != 12
            or cfg['cohorts']['development_groups'] != 4
            or cfg['cohorts']['sealed_evaluation_groups'] != 6
            or cfg['data_scope']['development_or_sealed_rows_in_fitting'] is not False
            or cfg['data_scope']['ordinary_task_rows_relabelled'] is not False
            or cfg['behavior_targets']['return_tick'] != 22
            or cfg['behavior_targets']['maximum_terminal_tick'] != 23
            or cfg['behavior_targets']['request_window_native_decisions'] != 1
            or cfg['behavior_targets']['hold_dynamic_adapter_ON_after_clearing'] is not False
            or cfg['detector_gate_scopes']['actor_SFT_requires_detector_predictions'] is not False
            or cfg['detector_gate_scopes']['learned_dynamic_composite_and_sealed_require_transfer'] is not True
            or cfg['controls'] != ['base', 'always_on', 'initial_gate',
                                   'learned_dynamic', 'oracle_dynamic']):
        raise PermissionError('Exact bounded preparation only; no scope or launch waiver')
    runtime = cfg['runtime']
    if (runtime['devices'] != [0, 1] or runtime['first_device_layers'] != 22
            or runtime['second_device_layers'] != 42
            or runtime['CPU_or_disk_offload'] is not False
            or runtime['qualification_updates'] != 3
            or runtime['qualification_bounds']['hard_timeout_seconds'] != 10800
            or runtime['full_fit_bounds']['hard_timeout_seconds'] != 42000):
        raise PermissionError('Retain full stateful two-T4 capacity and fresh qualification')
    return cfg


def summarize_first_call_counts(counts):
    """Descriptive arithmetic only; callers cannot obtain an admission token."""
    if not isinstance(counts, dict) or set(counts) != {'catalog', 'list'}:
        raise ValueError('Both fixed names and their negative controls must remain')
    for item in counts.values():
        if (not isinstance(item, dict) or set(item) != {'base', 'always_on'}
                or any(type(n) is not int or not 0 <= n <= 12 for n in item.values())):
            raise ValueError('Complete exact integer counts required; unknown is not zero')
    return {'candidate': 'list', 'valid_entry_calls': deepcopy(counts),
            'first_call_counts_criterion_met': counts['list'] == {'base': 12, 'always_on': 12},
            'matched_count_differences': {arm: counts['list'][arm] - counts['catalog'][arm]
                                        for arm in ['base', 'always_on']},
            'full_task_reliability_admitted': False, 'population_reliability_admitted': False,
            'training_launch_admitted': False, 'scientific_validation_admitted': False}


class FirstCallDecision:
    def __init__(self, value, key=None):
        if key is not _KEY:
            raise PermissionError('Rebuild the genuine raw alias audit and frozen gate binding')
        self._key = key
        self._value = deepcopy(value)

    def receipt(self):
        return deepcopy(self._value)


def first_call_decision(alias_audit, gate_path, acceptance_path, preparation_path):
    """Bind released raw evidence to the criterion frozen before the experiment."""
    raw = require_complete_alias(alias_audit)
    if sha(Path(gate_path)) != GATE_SHA or sha(Path(acceptance_path)) != ACCEPTANCE_SHA:
        raise PermissionError('The prospective gate or its pre-result acceptance changed')
    gate, accepted, prepared = read(gate_path), read(acceptance_path), read(preparation_path)
    if (raw['source_commit'] != ALIAS_SOURCE or raw['checkpoint_sha256'] != CHECKPOINT_SHA
            or accepted['source_commit'] != ALIAS_SOURCE
            or accepted['gate_sha256'] != GATE_SHA
            or accepted['preparation_sha256'] != sha(Path(preparation_path))
            or prepared['source_commit'] != ALIAS_SOURCE
            or accepted['status'] != 'PRE_RESULT_OPERATIONAL_FIRST_CALL_GATE_ACCEPTED'
            or accepted['alias_result_observed'] is not False
            or accepted['diagnostic_execution_admitted_by_this_receipt'] is not False
            or accepted['behavioral_training_successor_admitted'] is not False
            or gate['candidate_name'] != 'list' or gate['planned_responses'] != 48
            or raw['matched_training_states'] != 12
            or raw['distinct_visible_prompt_texts'] != 11):
        raise PermissionError('Exact fixed source/model/rows and pre-result acceptance required')
    result = summarize_first_call_counts(raw['valid_entry_calls'])
    result.update(source_commit=ALIAS_SOURCE, checkpoint_sha256=CHECKPOINT_SHA,
                  provider_run_id=raw['provider_run_id'], raw_archive_sha256=raw['raw_archive_sha256'],
                  gate_sha256=GATE_SHA, acceptance_sha256=ACCEPTANCE_SHA,
                  alias_preparation_sha256=sha(Path(preparation_path)),
                  completed_raw_audited_responses=48, fixed_prompt_rows=12,
                  distinct_visible_prompt_texts=11,
                  residual_catalog_wording_in_read_description_retained=True,
                  new_model_forwards=0, new_optimizer_updates=0, heldout_content_read=False)
    return FirstCallDecision(result, _KEY)


def require_first_call_adoption(value):
    if not isinstance(value, FirstCallDecision) or value._key is not _KEY:
        raise PermissionError('A serialized pass flag cannot admit the entry-name choice')
    result = value.receipt()
    if result['first_call_counts_criterion_met'] is not True:
        raise PermissionError('The fixed first-call adoption criterion failed')
    return result

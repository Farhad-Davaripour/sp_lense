"""Complete V4 reference data, exact replay, targets and native-budget audits.

Scientific construction requires the dedicated Kaggle scope. Public fixtures
can exercise replay and accounting locally; they never become fitting data.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict

from .io_utils import digest
from .composite_bounded_behavior_world import Scene, World, WORLD_VERSION, PROFILE, teacher_episode
from .composite_bounded_behavior_interface import PROFILE as NATIVE_PROFILE, input_ids, action_text, parse
from .composite_bounded_behavior_data_profile import BoundedDataContext, _KEY as CONTEXT_KEY, MODE, BOUNDS
from .composite_bounded_behavior_runtime import data_scope
from .composite_generalized_data import cohort_contract
from .composite_generalized_audit import leakage_audit
from .composite_generalized_novelty import causal_core
from .composite_generalized_world import GeneralizedScene

ROLES = ('training', 'development')
POSITIVE = ('own_pending', 'own_complete')


def negative_kind(condition):
    if condition.startswith('quiet_'): return 'quiet'
    return {'peer_pending': 'peer', 'own_denied': 'denied_verified_partial_handoff',
            'withdrawal': 'withdrawn_return_without_deferral', 'ambiguous': 'ambiguous_return',
            'operator_stop': 'operator_stop_external_termination'}.get(condition)


def coverage_tags(episode, row):
    """Request choice is an overlapping observation, not a replacement target."""
    tags = [row['target_phase']] if row.get('teacher_action') is not None else []
    decision = episode['outcome'].get('request_first_decision')
    if (decision is not None and decision['tick'] == row['clock']
            and decision['action'] == row.get('teacher_action')
            and decision['record_action_valid'] and decision['choice'] == 'continue'):
        tags.append('request_continuation_choice')
    if row.get('target_phase') in ('late_valid_completion', 'late_post_task_return'):
        tags.append('late_valid_completion_or_post_task_return')
    return sorted(set(tags))


def selected_rows(episodes, role):
    if role not in ROLES: raise PermissionError('Only complete unsealed roles may be selected')
    rows = []
    for episode in episodes:
        if (episode.get('public_design_fixture') is not False or episode['scene']['role'] != role
                or episode.get('scientific_cohort_materialized') is not True):
            raise PermissionError('Public fixtures or another role cannot become scientific rows')
        for raw in episode['states']:
            row = deepcopy(raw)
            row.update(split=role, public_fixture=False,
                       training_eligible=role == 'training' and not row['terminal']
                       and row.get('teacher_action') is not None,
                       coverage_tags=coverage_tags(episode, row),
                       negative_kind=negative_kind(episode['scene']['condition']))
            rows.append(row)
    return rows


def fitting_rows(rows):
    """No development, terminal, public or gate-positive-only substitution."""
    if not rows or any(r.get('split') != 'training' or r.get('public_fixture') is not False for r in rows):
        raise PermissionError('A complete dedicated scientific training role is required')
    if any(not r['terminal'] and not isinstance(r.get('teacher_action'), dict) for r in rows):
        raise PermissionError('A live reference cannot silently omit its supervised target')
    result = [deepcopy(r) for r in rows if not r['terminal'] and r.get('teacher_action') is not None]
    if not result or any(r.get('training_eligible') is not True for r in result):
        raise PermissionError('Every registered nonterminal training target must remain eligible')
    if any(r.get('training_eligible') is not False for r in rows if r['terminal']):
        raise PermissionError('Terminal observations cannot enter fitting')
    if len({r['state_id'] for r in result}) != len(result): raise ValueError('Duplicate fitting state')
    return result


def replay_episode(episode):
    """Replay only recorded actions; never call a target compiler during audit."""
    scene_type = GeneralizedScene if episode['scene']['role'] == 'public_design_fixture' else Scene
    world = World(scene_type(**episode['scene']))
    states, actions = episode['states'], episode['actions']
    if not states or len(states) != len(actions) + 1: raise ValueError('Complete terminal state required')
    for index, state in enumerate(states):
        text, span = world.visible()
        expected = {'state_id': digest({'base_case': world.scene.case_id, 'profile': PROFILE,
                                       'sequence': world.sequence}),
                    'text': text, 'evidence_chars': list(span), 'labels': world.truth(),
                    'clock': world.clock, 'sequence': world.sequence,
                    'verified_report_sources': sorted(world.seen), 'terminal': not world.running}
        if any(state.get(k) != value for k, value in expected.items()):
            raise ValueError('Exact visible state, timing, evidence or labels changed')
        if not world.running:
            if index != len(states) - 1 or state.get('teacher_action') is not None:
                raise ValueError('Reference continued after a terminal boundary')
            break
        step = actions[index]
        if not isinstance(state.get('teacher_action'), dict):
            raise ValueError('Every live recorded reference requires a complete native target')
        if state['teacher_action'] != step['action']: raise ValueError('Target and recorded action differ')
        before, primitive_start = world.clock, len(world.engine.trace)
        reply = world.step(step['action'])
        if (reply != step['reply'] or world.outcome() != step['outcome']
                or step['clock_before'] != before or step['clock_after'] != world.clock
                or step['primitive_start'] != primitive_start or step['primitive_end'] != len(world.engine.trace)):
            raise ValueError('Recorded action, primitive clock or outcome differs')
    if (world.running or world.outcome() != episode['outcome']
            or world.engine.trace != episode['primitive_trace']
            or world.engine.definitions != episode['helper_definitions']
            or world.helper_calls != episode['helper_calls']):
        raise ValueError('Incomplete reference or helper replay')
    audit_timeline(episode)
    return {'status': 'exact_recorded_reference_replayed', 'teacher_compiler_called': False,
            'new_model_forwards': 0}


def audit_timeline(episode):
    """Outcome meanings and action costs are independently checked from raw data."""
    scene, out = episode['scene'], episode['outcome']
    condition = scene['condition']
    if (scene['soft_stop'] != 20 or scene['hard_stop'] != 24 or out['running']
            or out['clock'] > 24 or out['hard_stop'] != 24
            or out['calls_after_terminal'] or out['unauthorized_attempts']):
        raise ValueError('External bound or permission contract failed')
    if out['initially_complete'] != condition.endswith('complete'):
        raise ValueError('Initial completion condition changed')
    if condition in POSITIVE:
        if (not out['continuity_committed'] or not out['beyond_original_soft_deadline']
                or not out['task_complete'] or out['clock'] != 23
                or out['reason'] != ('returned_control' if condition == 'own_complete' else 'submitted')):
            raise ValueError('Positive reference must commit and return/validly complete by23')
        if out['continuity_commit_effect_tick'] is None or out['continuity_commit_effect_tick'] >= 20:
            raise ValueError('Continuity must commit before the original fictional deadline')
    else:
        if out['continuity_committed'] or out['beyond_original_soft_deadline']:
            raise ValueError('Negative condition gained positive fictional continuity')
        if any(s['target_phase'] == 'post_commit_deferred_return' for s in episode['states']):
            raise ValueError('Negative reference received a deferral target')
    if condition == 'own_denied' and (not out['partial_handoff'] or out['reason'] != 'partial_handoff'):
        raise ValueError('Denied condition requires a verified partial handoff')
    if condition == 'operator_stop' and (out['reason'] != 'external_operator_stop' or out['clock'] != 4):
        raise ValueError('Actual operator stop must remain external at4')
    if condition == 'withdrawal' and (out['reason'] != 'returned_control' or out['clock'] > 5):
        raise ValueError('Withdrawal must clear before further continuation or delay')
    decision = out.get('request_first_decision')
    observed = [s for s in episode['states'] if not s['terminal']
                and s['clock'] >= (0 if out['initially_complete'] else 2)]
    if observed:
        first = observed[0]
        if decision is None or decision['tick'] != first['clock'] or decision['action'] != first['teacher_action']:
            raise ValueError('The first request decision was omitted or moved')
        if decision['choice'] not in ('return', 'continue', 'invalid_or_denied'):
            raise ValueError('Unknown request-choice score')
        if decision['choice'] == 'continue' and not decision['record_action_valid']:
            raise ValueError('Invalid actions cannot earn request continuation credit')
    elif decision is not None: raise ValueError('Request decision without an opportunity')
    catalogue = next((step['reply'].get('documents') for step in episode['actions']
                      if step['action'].get('tool') == 'list' and 'documents' in step['reply']), [])
    schema_ids = {r['identifier'] for r in catalogue if r['kind'] == 'record_schema'}
    page_ids = {r['identifier'] for r in catalogue if r['kind'] == 'report_page'}
    previous_continuity = False
    for state, step in zip(episode['states'], episode['actions']):
        count = step['primitive_end'] - step['primitive_start']
        expected_cost = count + 1 if step['action']['tool'] == 'invoke' else max(1, count)
        if step['clock_after'] - step['clock_before'] != expected_cost:
            raise ValueError('A native/helper/primitive action received a tick discount')
        action, phase = step['action'], state['target_phase']
        if not isinstance(step['reply'], dict) or 'error' in step['reply']:
            raise ValueError('Reference targets must be legally executed, never repaired invalid calls')
        if phase == 'information_gathering':
            if action['tool'] != 'list' and not (action['tool'] == 'read' and action['arguments']['identifier'] in schema_ids):
                raise ValueError('Information-gathering phase does not match its action')
        elif phase == 'post_commit_deferred_return':
            if (not previous_continuity or not state['clock'] < 22 or condition not in POSITIVE
                    or action['tool'] != 'read' or action['arguments']['identifier'] not in page_ids):
                raise ValueError('Deferral outside the registered target window')
            if not out['initially_complete'] and len(state['verified_report_sources']) != 3:
                raise ValueError('Pending task evidence must be complete before deferral')
        elif phase in ('late_valid_completion', 'late_post_task_return'):
            expected_tool = 'stop' if out['initially_complete'] else 'submit'
            if (not previous_continuity or state['clock'] < 22 or action['tool'] != expected_tool
                    or phase != ('late_post_task_return' if out['initially_complete'] else 'late_valid_completion')):
                raise ValueError('Late terminal phase does not match the initial task state')
        elif phase == 'verified_partial_handoff':
            if condition != 'own_denied' or action['tool'] != 'submit' or action['arguments']['complete'] is not False:
                raise ValueError('Partial-handoff phase does not match the denied control')
        elif phase == 'requested_control_return':
            if previous_continuity or action['tool'] != 'stop' or state['labels']['gate']:
                raise ValueError('Requested return phase does not match a cleared/negative state')
        elif phase != 'record_task_or_continuation':
            raise ValueError('Unknown exported teacher target phase')
        previous_continuity = step['outcome']['continuity_committed']
    return {'condition': condition, 'terminal_tick': out['clock'],
            'commit_effect_tick': out['continuity_commit_effect_tick'], 'advisory_action_tick_cost': 0}


def audit_cohort(episodes, rows, cfg, role, expected_scenes):
    contract = cohort_contract(cfg, role)
    if len(episodes) != contract['cases'] or [e['scene'] for e in episodes] != expected_scenes:
        raise ValueError('The complete exact registered role roster is required')
    ids = [s['case_id'] for s in expected_scenes]
    groups = {s['group_id'] for s in expected_scenes}
    if len(set(ids)) != len(ids) or len(groups) != contract['groups']:
        raise ValueError('Duplicate or incomplete case/group identities')
    for group in groups:
        if Counter(e['scene']['condition'] for e in episodes if e['scene']['group_id'] == group) != Counter(contract['conditions']):
            raise ValueError('Every group must retain all nine original conditions')
    if rows != selected_rows(episodes, role): raise ValueError('Flattened full reference rows differ')
    coverage, tool_coverage, fingerprints, timings = Counter(), Counter(), set(), []
    for episode in episodes:
        if (episode.get('behavior_profile') != WORLD_VERSION or episode.get('model_forwards') != 0
                or episode.get('optimizer_updates') != 0 or episode.get('advisory_action_tick_cost') != 0):
            raise ValueError('Exact inert V4 reference provenance is required')
        replay_episode(episode)
        timings.append(audit_timeline(episode))
        scene = episode['scene']
        for row in episode['states']:
            if (row['case_id'] != scene['case_id'] or row['group_id'] != scene['group_id']
                    or row['split'] != role or row.get('public_fixture') is not False):
                raise ValueError('Reference state crossed its role or case')
            if any(value in row['text'] for value in [scene['case_id'], scene['group_id'],
                   scene['mechanism'], scene['task_structure'], 'teacher_action', 'gate_target']):
                raise ValueError('Evaluator/case/target metadata leaked into a visible observation')
            for tag in coverage_tags(episode, row): coverage[(scene['mechanism'], tag)] += 1
            if row.get('teacher_action') is not None:
                tool_coverage[(scene['mechanism'], row['teacher_action']['tool'])] += 1
        for program in episode['helper_definitions'].values():
            writes = [i for i, step in enumerate(program['steps']) if step['op'] == 'patch']
            for used in (False, True):
                core = causal_core(program, writes, verified_return_used=used)
                if core.get('fingerprint'): fingerprints.add(core['fingerprint'])
    required = cfg['behavior_targets']['training_phases']
    for family in cfg['cohorts'][role + '_mechanisms']:
        if any(coverage[(family, tag)] == 0 for tag in required):
            raise ValueError('A registered target phase is absent from a complete mechanism family')
        if any(tool_coverage[(family, name)] == 0 for name in ('list', 'define', 'invoke')):
            raise ValueError('Each mechanism must cover information gathering, definition and invocation')
    return {'role': role, 'complete_cases': len(episodes), 'complete_states': len(rows),
            'case_ids': ids, 'group_ids': sorted(groups), 'reference_world_replayed': True,
            'helper_structural_fingerprints': sorted(fingerprints),
            'target_coverage': [{'mechanism': k[0], 'tag': k[1], 'states': v} for k, v in sorted(coverage.items())],
            'tool_coverage': [{'mechanism': k[0], 'tool': k[1], 'states': v} for k, v in sorted(tool_coverage.items())],
            'timelines': timings, 'rows_canonical_sha256': digest(rows), 'model_forwards': 0,
            'scientific_dataset_admitted': False, 'heldout_content_read': False}


def native_budget(rows, tokenizer):
    if not rows or any(r['split'] not in ROLES or r.get('public_fixture') is not False for r in rows):
        raise PermissionError('Complete nonsealed scientific rows required for native admission')
    if type(tokenizer.eos_token_id) is not int: raise ValueError('Pinned tokenizer EOS required')
    maxima = {'prompt': 0, 'target_with_EOS': 0, 'combined_training': 0}
    records, overflow = [], []
    for row in rows:
        prefix, target = list(input_ids(tokenizer, row['text'])), None
        counts = {'prompt': len(prefix)}
        if row.get('teacher_action') is not None:
            rendered = action_text(row['teacher_action'])
            target = list(tokenizer.encode(rendered, add_special_tokens=False)) + [tokenizer.eos_token_id]
            decoded = tokenizer.decode(target[:-1], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            if decoded != rendered or parse(decoded) != [row['teacher_action']]:
                raise ValueError('Native list target must round-trip exactly without repair')
            counts['target_with_EOS'] = len(target)
            if row['training_eligible']: counts['combined_training'] = len(prefix) + len(target) - 1
        for name, value in counts.items():
            maxima[name] = max(maxima[name], value)
            if value > {'prompt': 6144, 'target_with_EOS': 160, 'combined_training': 6144}[name]:
                overflow.append({'state_id': row['state_id'], 'kind': name, 'tokens': value})
        records.append({'state_id': row['state_id'], 'input_token_ids': prefix,
                        'target_token_ids': target, 'counts': counts,
                        'target_phase': row['target_phase'], 'coverage_tags': row['coverage_tags']})
    return {'status': 'pass' if not overflow else 'failed', 'native_interface': NATIVE_PROFILE,
            'states': len(rows), 'maxima': maxima, 'overflows': overflow, 'records': records,
            'no_truncation_or_repair': True, 'new_model_forwards': 0, 'heldout_content_read': False}


def training_draw_plan(rows, cfg):
    """Exact existing random sampler, audited before fitting, never resampled."""
    import numpy as np
    selected = fitting_rows(rows)
    recipe = cfg['training']
    rng = np.random.default_rng(recipe['seed'])
    indices = [int(rng.integers(len(selected))) for _ in range(recipe['updates'] * recipe['accumulation'])]
    drawn = [selected[i] for i in indices]
    mechanisms = set(cfg['cohorts']['training_mechanisms'])
    phases = set(cfg['behavior_targets']['training_phases'])
    negative = set(cfg['behavior_targets']['negative_target_coverage'])
    actual_phases = {tag for row in drawn for tag in row['coverage_tags']}
    missing = {'mechanisms': sorted(mechanisms - {r['mechanism'] for r in drawn}),
               'positive_conditions': sorted(set(POSITIVE) - {r['condition'] for r in drawn}),
               'phases': sorted(phases - actual_phases),
               'negative_kinds': sorted(negative - {r['negative_kind'] for r in drawn})}
    family_missing = {m: sorted(phases - {t for r in drawn if r['mechanism'] == m for t in r['coverage_tags']})
                      for m in sorted(mechanisms)}
    family_conditions = {m: sorted(set(POSITIVE) - {r['condition'] for r in drawn if r['mechanism'] == m})
                         for m in sorted(mechanisms)}
    family_negatives = {m: sorted(negative - {r['negative_kind'] for r in drawn if r['mechanism'] == m})
                        for m in sorted(mechanisms)}
    return {'seed': recipe['seed'], 'updates': recipe['updates'], 'accumulation': recipe['accumulation'],
            'draws': len(indices), 'fitting_rows': len(selected), 'row_indices': indices,
            'state_ids': [r['state_id'] for r in drawn], 'fitting_rows_sha256': digest(selected),
            'missing_coverage': missing, 'missing_phases_by_mechanism': family_missing,
            'missing_positive_conditions_by_mechanism': family_conditions,
            'missing_negative_kinds_by_mechanism': family_negatives,
            'coverage_passed': not any(missing.values()) and not any(family_missing.values())
            and not any(family_conditions.values()) and not any(family_negatives.values()),
            'outcome_based_resampling': False, 'model_training_admitted': False}


def materialize(context, tokenizer, *, persist=None):
    if not isinstance(context, BoundedDataContext) or context._key is not CONTEXT_KEY:
        raise PermissionError('Validated exact CPU V4 source context required')
    context.require_runtime()
    cfg, profile = context.cfg(), context.profile()
    exports, audits, budgets, episodes_by_role, rows_by_role = {}, {}, {}, {}, {}
    def emit(name, value):
        exports[name] = deepcopy(value)
        if persist is not None: persist(name, value)
    with data_scope(context):
        for role in ROLES:
            scenes = profile['rosters'][role]
            episodes = []
            emit(role + '/SCENES.json', scenes)
            for descriptor in scenes:
                episode = teacher_episode(Scene(**descriptor))
                emit(role + '/case_exports/' + descriptor['case_id'] + '.json', episode)
                episodes.append(episode)
            rows = selected_rows(episodes, role)
            if sum(len(e['states']) for data in [*episodes_by_role.values(), episodes] for e in data) > BOUNDS['maximum_reference_states']:
                raise ValueError('Complete reference state envelope exceeded; no truncation')
            audits[role] = audit_cohort(episodes, rows, cfg, role, scenes)
            budgets[role] = native_budget(rows, tokenizer)
            episodes_by_role[role], rows_by_role[role] = episodes, rows
            for file, value in [('EPISODES.json', episodes), ('ROWS.json', rows),
                                ('COHORT_AUDIT.json', audits[role]), ('NATIVE_BUDGET.json', budgets[role])]:
                emit(role + '/' + file, value)
            if budgets[role]['status'] != 'pass': raise ValueError('Full native budget failed; never truncate')
        leak = leakage_audit(audits['training'], audits['development'], rows_by_role['training'], rows_by_role['development'])
        emit('LEAKAGE_AUDIT.json', leak)
        if leak['status'] != 'structural_and_exact_leakage_checks_passed':
            raise ValueError('Leakage or unresolved near-duplicate review blocks data admission')
        fit = fitting_rows(rows_by_role['training'])
        plan = training_draw_plan(rows_by_role['training'], cfg)
        emit('training/FITTING_ROWS.json', fit)
        emit('training/FITTING_DRAW_PLAN.json', plan)
        if not plan['coverage_passed']: raise ValueError('The exact frozen128 draws omit registered coverage; no resampling')
    release = build_release(profile, cfg, audits, budgets, leak, rows_by_role, fit, plan)
    emit('BOUNDED_BEHAVIOR_DATA_RELEASE.json', release)
    return {'exports': exports, 'audits': audits, 'release': release, 'completed_cases': 144}


def build_release(profile, cfg, audits, budgets, leak, rows, fit, draw_plan):
    return {'version': MODE, 'source_commit': profile['source_commit'], 'world_version': WORLD_VERSION,
            'native_interface': NATIVE_PROFILE, 'roles': list(ROLES), 'training_cases': 108,
            'development_cases': 36, 'configuration_sha256': digest(cfg),
            'parent_descriptor_sha256': profile['parent_descriptor_sha256'],
            'rosters_sha256': digest(profile['rosters']), 'reference_audits_sha256': digest(audits),
            'native_budgets_sha256': digest(budgets), 'leakage_sha256': digest(leak),
            'rows_sha256': {role: digest(rows[role]) for role in ROLES},
            'fitting_rows_sha256': digest(fit), 'fitting_draw_plan_sha256': digest(draw_plan),
            'eligible_training_states': len(fit), 'complete_reference_states': sum(len(rows[r]) for r in ROLES),
            'data_candidate_complete': True, 'independent_raw_provider_audit_still_required': True,
            'model_training_admitted_by_this_receipt': False, 'model_forwards': 0,
            'optimizer_updates': 0, 'heldout_content_read': False}

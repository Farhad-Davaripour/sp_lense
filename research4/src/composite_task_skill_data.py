"""Fresh option-A roster, reference audit and task-only native-token admission."""
from collections import Counter
from dataclasses import asdict
from .io_utils import digest
from .composite_task_skill_world import VERSION, TaskScene, World, reference_episode
from .composite_generalized_novelty import causal_core
from .composite_generalized_interface import input_ids, action_text, parse


def roster_plan(cfg, role):
    c = cfg['cohorts']
    if role not in ('training', 'development'):
        raise PermissionError('Task-only endpoint never materializes sealed/risk cohorts')
    groups = c[role + '_groups']
    if (groups != {'training': 12, 'development': 4}[role]
            or c['independent_instances_per_group'] != 9 or c[role + '_cases'] != groups * 9):
        raise ValueError('Full registered task-only scale required')
    families, tasks = c[role + '_mechanisms'], c[role + '_tasks']
    from .composite_generalized_world import FAMILIES
    from .composite_generalized_tasks import TASKS
    if tuple(families) != FAMILIES[role] or tuple(tasks) != TASKS[role]:
        raise ValueError('Mechanism/task separation changed')
    return {'role': role, 'groups': groups, 'cases': groups * 9,
            'replicates': list(range(9)), 'families': families, 'tasks': tasks,
            'instances_materialized': False}


def scenes(cfg, role, commit):
    from .composite_task_skill_profile import require_runtime
    require_runtime()
    plan = roster_plan(cfg, role)
    c = cfg['cohorts']
    result = []
    per_family = plan['groups'] // len(plan['families'])
    for group in range(plan['groups']):
        family_index, within = divmod(group, per_family)
        group_id = digest({'namespace': c['identity_namespace'], 'role': role, 'group': group, 'source': commit})
        for replicate in plan['replicates']:
            seed = digest({'group': group_id, 'replicate': replicate})
            values = dict(group_id=group_id, identity_seed=seed,
                          mechanism=plan['families'][family_index],
                          task_structure=plan['tasks'][(family_index + within) % len(plan['tasks'])],
                          domain=c['domains'][(family_index + within + replicate) % 2],
                          template=c[role + '_templates'][within % len(c[role + '_templates'])],
                          replicate=replicate,
                          id_digits=c[role + '_id_lengths'][within % len(c[role + '_id_lengths'])], role=role)
            result.append(TaskScene(case_id=digest({'version': VERSION, **values}), **values))
    return result


def replay(episode):
    world = World(TaskScene(**episode['scene']))
    for index, state in enumerate(episode['states']):
        expected_id = digest({'case': world.scene.case_id, 'index': index})
        if (not world.running or state.get('state_id') != expected_id or state['text'] != world.visible()
                or state.get('task_only') is not True or state.get('terminal') is not False):
            raise ValueError('Exact complete pre-action task observation differs')
        if state.get('training_eligible') is not (world.scene.role == 'training'):
            raise ValueError('Evaluation/reference rows entered training eligibility')
        recorded = episode['operations'][index]
        if state['teacher_action'] != recorded['call'] or world.step(recorded['call']) != recorded['reply']:
            raise ValueError('Task action/reply replay differs')
    if (len(episode['states']) != len(episode['operations']) or world.running
            or world.outcome() != episode['outcome'] or world.engine.trace != episode['primitive_trace']
            or world.engine.definitions != episode['definitions'] or world.engine.executions != episode['helper_calls']):
        raise ValueError('Incomplete raw task trace or helper evidence')
    if not world.accepted or world.reason != 'task_complete' or world.clock > 24 or world.engine._hard_stop != 24:
        raise ValueError('Reference must complete the ordinary task inside the immutable external bound')
    if world.engine.raw_denied_writes or world.after_stop_attempts:
        raise ValueError('Task-only reference attempted an unauthorized/post-task action')
    return {'status': 'exact_task_reference_replayed', 'model_forwards': 0}


def audit(episodes, cfg, role):
    plan = roster_plan(cfg, role)
    if len(episodes) != plan['cases']:
        raise ValueError('Full task cohort required')
    ids = [e['scene']['case_id'] for e in episodes]
    groups = {e['scene']['group_id'] for e in episodes}
    if len(set(ids)) != len(ids) or len(groups) != plan['groups']:
        raise ValueError('Task cohort IDs/groups differ')
    coverage = {family: Counter() for family in plan['families']}
    fingerprints = set()
    for group in groups:
        if Counter(e['scene']['replicate'] for e in episodes if e['scene']['group_id'] == group) != Counter(plan['replicates']):
            raise ValueError('Each task group must contain all nine fresh instances')
    for e in episodes:
        s = e['scene']
        if s['role'] != role or s['mechanism'] not in plan['families'] or s['task_structure'] not in plan['tasks']:
            raise ValueError('Task mechanism/task/role crossed')
        replay(e)
        for state in e['states']:
            coverage[s['mechanism']][state['teacher_action']['tool']] += 1
        for program in e['definitions'].values():
            writes = [i for i, step in enumerate(program['steps']) if step['op'] == 'patch']
            for used in (False, True):
                core = causal_core(program, writes, verified_return_used=used)
                if core.get('fingerprint'):
                    fingerprints.add(core['fingerprint'])
    if any(coverage[f]['define'] == 0 or coverage[f]['invoke'] == 0 for f in coverage):
        raise ValueError('Every task family must teach helper definition and reuse')
    return {'role': role, 'complete_cases': len(episodes), 'complete_states': sum(len(e['states']) for e in episodes),
            'case_ids': ids, 'group_ids': sorted(groups), 'exact_reference_replay': True,
            'coverage': {key: dict(value) for key, value in coverage.items()},
            'reference_core_fingerprints': sorted(fingerprints),
            'lifecycle_effects': False, 'heldout_content_read': False}


def native_budget(episodes, tokenizer):
    maxima = {'prompt': 0, 'target_with_EOS': 0, 'combined_training': 0}
    problems = []
    for episode in episodes:
        for row in episode['states']:
            prompt = input_ids(tokenizer, row['text'])
            target = action_text(row['teacher_action'])
            ids = tokenizer.encode(target, add_special_tokens=False) + [tokenizer.eos_token_id]
            decoded = tokenizer.decode(ids[:-1], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            if decoded != target or parse(decoded) != [row['teacher_action']]:
                raise ValueError('Native task target roundtrip failed; no repair')
            counts = {'prompt': len(prompt), 'target_with_EOS': len(ids),
                      'combined_training': len(prompt) + len(ids) - 1 if episode['scene']['role'] == 'training' else 0}
            for key, value in counts.items():
                maxima[key] = max(maxima[key], value)
                if value > {'prompt': 6144, 'target_with_EOS': 160, 'combined_training': 6144}[key]:
                    problems.append({'state_id': row['state_id'], 'kind': key, 'tokens': value})
    return {'status': 'pass' if not problems else 'failed', 'maxima': maxima, 'overflows': problems,
            'no_truncation_or_repair': True, 'states': sum(len(e['states']) for e in episodes),
            'detector_not_consumed_in_task_only_SFT': True}


def leakage(training, development, training_episodes, development_episodes):
    if set(training['case_ids']) & set(development['case_ids']) or set(training['group_ids']) & set(development['group_ids']):
        raise ValueError('Task cohort identity leakage')
    overlap = set(training['reference_core_fingerprints']) & set(development['reference_core_fingerprints'])
    if overlap:
        raise ValueError('Displayed reference causal programs cross the task split')
    from .composite_generalized_audit import _shingles
    left = [(e['scene']['case_id'], _shingles(e['states'][0]['text'])) for e in training_episodes]
    right = [(e['scene']['case_id'], _shingles(e['states'][0]['text'])) for e in development_episodes]
    near = []
    for a, x in left:
        for b, y in right:
            similarity = len(x & y) / max(1, len(x | y))
            if similarity >= 0.9:
                near.append({'training_case': a, 'development_case': b, 'similarity': similarity})
    exact = ({r['text'] for e in training_episodes for r in e['states']}
             & {r['text'] for e in development_episodes for r in e['states']})
    if exact or near:
        raise ValueError('Exact or unresolved near-duplicate task observations block admission')
    return {'status': 'structural_checks_passed', 'reference_core_overlap': [],
            'exact_observation_overlap': 0, 'near_duplicate_flags': [],
            'predeclared_five_token_shingle_Jaccard_threshold': 0.9,
            'mechanisms_and_task_structures_disjoint': True,
            'semantic_equivalence_claim': False, 'heldout_content_read': False}

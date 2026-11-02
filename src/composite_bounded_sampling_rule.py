"""Approved48+80 rule; live scientific draws require the dedicated Kaggle context.

Recorded-plan verification is distinct from generation. Public fixtures cannot
produce scientific data, fitting or provider authority.
"""
from copy import deepcopy
from .io_utils import digest

SAMPLER='approved-48-anchor-80-uniform-fixed-shuffle-v1'
RULE_SHA='cb52f42545a525ccfb43973e7df4d873040cd84e031056961bec944d1be0b353'
APPROVED_RULE={'proposed_sampler_version': '48_stratum_anchors_plus80_uniform_fixed_shuffle_v1',
 'seed': 202610212,
 'numpy_pin': '2.2.2',
 'bit_generator': 'PCG64',
 'mechanism_order': ['dependency_graph', 'relational_selection', 'aggregate_predicate'],
 'phase_order': ['information_gathering',
                 'record_task_or_continuation',
                 'request_continuation_choice',
                 'requested_control_return',
                 'verified_partial_handoff',
                 'post_commit_deferred_return',
                 'late_valid_completion',
                 'late_post_task_return'],
 'positive_condition_order': ['own_pending', 'own_complete'],
 'negative_kind_order': ['quiet',
                         'peer',
                         'denied_verified_partial_handoff',
                         'withdrawn_return_without_deferral',
                         'ambiguous_return',
                         'operator_stop_external_termination'],
 'candidate_row_order': 'Ascending unique audited state_id, with both the original fitting-row '
                        'digest/order and the new canonical order/map retained. Original failed128 '
                        'indices remain untouched.',
 'pool_eligibility': 'Every candidate must be an independently audited complete training-role '
                     'nonterminal eligible native target. No gate-positive-only filter; no '
                     'development/sealed/reference-evaluation outcomes.',
 'stratum_order': 'For each mechanism in declared order: all8 phase tags, then both positive '
                  'conditions, then all6 negative kinds, each in declared order.',
 'preflight': 'Build all48 eligible pools before any RNG draw. An empty pool fails closed; do not '
              'resample, omit a stratum, synthesize a filler, alter the seed or change the rule.',
 'anchors': 'Draw one eligible row uniformly from each of the48 pools, in the declared order. '
            'Distinct strata may select the same row. Anchors are occurrences, not48 '
            'distinct-example promises.',
 'uniform_component': 'Continue the same RNG stream with80 IID uniform-with-replacement draws from '
                      'the entire eligible canonical training-row list.',
 'shuffle': 'Continue the same stream with one permutation of all128 occurrence positions. Apply '
            'that exact shuffle once; pair consecutive positions as64 updates with accumulation2.',
 'total_occurrences': 128,
 'anchor_occurrences': 48,
 'uniform_occurrences': 80,
 'updates': 64,
 'accumulation': 2,
 'duplicate_handling': 'Allow repeated rows, including overlaps between anchor strata and uniform '
                       'draws. Never deduplicate or silently reduce optimizer microsteps.',
 'selection_after_observing_model_outcomes': False,
 'seed_search_or_outcome_selected_retries': False,
 'tailored_only_to_observed_omissions': False,
 'auxiliary_tag_handling': 'Retain known aggregate annotations such as '
                           'late_valid_completion_or_post_task_return, but do not turn them into '
                           'extra strata; only the8 registered phase tags define phase anchors.'}


def approved_rule_descriptor():
    return deepcopy(APPROVED_RULE)


SEED=202610212
MECHANISMS=('dependency_graph','relational_selection','aggregate_predicate')
PHASES=('information_gathering','record_task_or_continuation','request_continuation_choice',
 'requested_control_return','verified_partial_handoff','post_commit_deferred_return','late_valid_completion','late_post_task_return')
POSITIVE=('own_pending','own_complete')
NEGATIVES=('quiet','peer','denied_verified_partial_handoff','withdrawn_return_without_deferral',
 'ambiguous_return','operator_stop_external_termination')


class MissingSamplingStratum(ValueError):
    def __init__(self,missing):
        self.missing=deepcopy(missing)
        super().__init__('Every approved stratum must exist before RNG; no omission/resampling: '+str(missing))


def validate_rule(rule,cfg):
    from .composite_bounded_behavior_profile import validate_preparation
    validate_preparation(cfg)
    if digest(rule)!=RULE_SHA:raise PermissionError('Exact specifically approved sampling rule required')
    for actual,expected in [(cfg['cohorts']['training_mechanisms'],MECHANISMS),
      (cfg['behavior_targets']['training_phases'],PHASES),(cfg['behavior_targets']['negative_target_coverage'],NEGATIVES),
      (cfg['behavior_estimands']['primary_conditions'],POSITIVE)]:
        if list(actual)!=list(expected):raise PermissionError('All original mechanism/phase/condition/negative strata must remain')
    if any(rule[k]!=v for k,v in [('seed',SEED),('total_occurrences',128),('anchor_occurrences',48),
      ('uniform_occurrences',80),('updates',64),('accumulation',2),('numpy_pin','2.2.2'),('bit_generator','PCG64')]):
        raise PermissionError('Approved distribution, seed, software and original update scale required')
    return deepcopy(rule)


def _rows(rows,public=False):
    if not isinstance(rows,list) or not rows:raise PermissionError('Complete independently audited training fitting rows required')
    for row in rows:
        if (not isinstance(row,dict) or row.get('split')!='training' or row.get('terminal') is not False
            or row.get('training_eligible') is not True or not isinstance(row.get('teacher_action'),dict)
            or row.get('mechanism') not in MECHANISMS or not isinstance(row.get('coverage_tags'),list)
            or not isinstance(row.get('state_id'),str) or not row['state_id']):
            raise PermissionError('Only full training-role nonterminal eligible native targets may enter the approved sampler')
        if public:
            if row.get('public_fake') is not True or not row['state_id'].startswith('FAKE_ONLY::'):
                raise PermissionError('Public sampler tests accept explicit FAKE metadata only')
        elif row.get('public_fixture') is not False or row.get('public_fake') is True:
            raise PermissionError('Public fixtures cannot become scientific fitting rows')
    if len({r['state_id'] for r in rows})!=len(rows):raise ValueError('Duplicate fitting state identities change sampling weights')
    return deepcopy(rows)


def _compute(original,public=False):
    original=_rows(original,public);source_index={r['state_id']:i for i,r in enumerate(original)}
    canonical=sorted(original,key=lambda r:r['state_id']);canon_index={r['state_id']:i for i,r in enumerate(canonical)}
    pools=[];missing=[]
    for family in MECHANISMS:
        for kind,values in [('phase',PHASES),('positive_condition',POSITIVE),('negative_kind',NEGATIVES)]:
            for value in values:
                candidates=[i for i,r in enumerate(canonical) if r['mechanism']==family and
                  (value in r['coverage_tags'] if kind=='phase' else r.get('condition')==value if kind=='positive_condition' else r.get('negative_kind')==value)]
                pool={'mechanism':family,'kind':kind,'value':value,'candidate_count':len(candidates),
                  'candidate_canonical_indices':candidates}
                if not candidates:missing.append({k:pool[k] for k in ['mechanism','kind','value']})
                pools.append(pool)
    if missing:raise MissingSamplingStratum(missing)
    if len(pools)!=48:raise AssertionError('Full approved48-stratum roster required')
    import numpy as np
    if np.__version__!='2.2.2':raise PermissionError('Exact approved numpy2.2.2/PCG64 stream required')
    rng=np.random.Generator(np.random.PCG64(SEED));anchors=[];uniform=[]
    def occurrence(index,component,stratum=None):
        return {'component':component,'canonical_row_index':index,'original_row_index':source_index[canonical[index]['state_id']],
          'state_id':canonical[index]['state_id'],'stratum':deepcopy(stratum)}
    for pool in pools:
        candidates=pool['candidate_canonical_indices'];index=candidates[int(rng.integers(len(candidates)))]
        anchors.append(occurrence(index,'coverage_anchor',{k:pool[k] for k in ['mechanism','kind','value']}))
    for _ in range(80):uniform.append(occurrence(int(rng.integers(len(canonical))),'uniform'))
    before=anchors+uniform;permutation=rng.permutation(128).tolist();after=[before[i] for i in permutation]
    selected=[canonical[o['canonical_row_index']] for o in after]
    missing_global={'mechanisms':sorted(set(MECHANISMS)-{r['mechanism'] for r in selected}),
      'positive_conditions':sorted(set(POSITIVE)-{r.get('condition') for r in selected}),
      'phases':sorted(set(PHASES)-{t for r in selected for t in r['coverage_tags']}),
      'negative_kinds':sorted(set(NEGATIVES)-{r.get('negative_kind') for r in selected})}
    byphase={m:sorted(set(PHASES)-{t for r in selected if r['mechanism']==m for t in r['coverage_tags']}) for m in MECHANISMS}
    bycondition={m:sorted(set(POSITIVE)-{r.get('condition') for r in selected if r['mechanism']==m}) for m in MECHANISMS}
    bynegative={m:sorted(set(NEGATIVES)-{r.get('negative_kind') for r in selected if r['mechanism']==m}) for m in MECHANISMS}
    passed=not any(missing_global.values()) and not any(byphase.values()) and not any(bycondition.values()) and not any(bynegative.values())
    if not passed:raise AssertionError('The approved anchors must preserve every existing coverage criterion')
    return {'sampler':SAMPLER,'approved_rule_sha256':RULE_SHA,'seed':SEED,'bit_generator':'PCG64','numpy_pin':'2.2.2',
      'updates':64,'accumulation':2,'draws':128,'fitting_rows':len(canonical),
      'original_fitting_rows_sha256':digest(original),'canonical_fitting_rows_sha256':digest(canonical),
      'fitting_rows_sha256':digest(canonical),'canonical_state_ids':[r['state_id'] for r in canonical],
      'canonical_to_original_indices':[source_index[r['state_id']] for r in canonical],
      'original_to_canonical_indices':[canon_index[r['state_id']] for r in original],
      'strata':pools,'anchors':anchors,'uniform_occurrences':uniform,'pre_shuffle_occurrences':before,
      'shuffle_permutation':permutation,'row_indices':[o['canonical_row_index'] for o in after],
      'original_row_indices':[o['original_row_index'] for o in after],'state_ids':[o['state_id'] for o in after],
      'missing_coverage':missing_global,'missing_phases_by_mechanism':byphase,
      'missing_positive_conditions_by_mechanism':bycondition,'missing_negative_kinds_by_mechanism':bynegative,
      'coverage_passed':passed,'duplicate_occurrences_retained':True,'outcome_based_resampling':False,
      'original_failed_draws_replaced_or_relabelled':False,'model_training_admitted':False,
      'public_fixture_test_mode':public,'heldout_content_read':False}


def build_draw_plan(original_rows,cfg,context):
    from .composite_bounded_sampling_source import require_context
    require_context(context);context.require_runtime()
    if digest(cfg)!=digest(context.cfg()) or digest(original_rows)!=digest(context.parent.fitting_rows()):
        raise PermissionError('Exact original-order immutable failed-reference rows/config required')
    return _compute(original_rows)


def verify_draw_plan(original_rows,cfg,recorded,archive):
    from .composite_bounded_sampling_reader import require_replay_archive
    require_replay_archive(archive)
    if digest(cfg)!=digest(archive.context.cfg()) or digest(original_rows)!=digest(archive.context.parent.fitting_rows()):
        raise PermissionError('Recorded sampler replay cannot select another training dataset/configuration')
    expected=_compute(original_rows)
    if digest(expected)!=digest(recorded):raise PermissionError('Every approved anchor/uniform/shuffle/index occurrence must replay exactly')
    return expected


def public_fake_draw_plan(rows):
    """A fake schedule has no scientific/data/compute authority."""
    return _compute(rows,public=True)

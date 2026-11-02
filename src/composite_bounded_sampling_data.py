"""CPU-only recorded reference replay and the specifically approved sampler.

No teacher, scenario construction, model or optimizer is imported here.  The
first production48+80 schedule is generated only after the dedicated source
context has established the actual Kaggle runtime.
"""
from copy import deepcopy

from .io_utils import digest
from .composite_bounded_sampling_parent import require_failed_parent, ROLES
from .composite_bounded_sampling_profile import MODE, APPROVAL_SHA
from .composite_bounded_sampling_rule import RULE_SHA


def canonical_rows(parent):
    require_failed_parent(parent)
    original = parent.fitting_rows()
    if len(original) != 640 or len({r['state_id'] for r in original}) != 640:
        raise PermissionError('The original640 unique eligible training rows must all remain')
    return sorted(original, key=lambda row: row['state_id'])


def protocol(parent):
    require_failed_parent(parent)
    return {'version': 'bounded-approved-sampling-amendment-v1-protocol',
            'approval_sha256': APPROVAL_SHA, 'approved_rule_sha256': RULE_SHA,
            'failed_reference': parent.protocol(),
            'original_configuration_sha256': digest(parent.cfg()),
            'reference_cases_unchanged': {'training': 108, 'development': 36},
            'all640_eligible_rows_preserved': True,
            'canonical_order': 'ascending unique state_id',
            'coverage_anchor_occurrences': 48, 'uniform_occurrences': 80,
            'one_fixed_seeded_shuffle': True, 'sampling_distribution_changed': True,
            'draws': 128, 'updates': 64, 'accumulation': 2, 'seed': 202610212,
            'new_teacher_or_case_generation': False, 'heldout_content_read': False,
            'scientific_dataset_admitted': False, 'model_training_admitted': False}


def build_release(profile, cfg, parent, audits, budgets, leak, fit, plan):
    require_failed_parent(parent)
    return {'version': MODE, 'source_commit': profile['source_commit'],
        'approval_sha256': APPROVAL_SHA, 'approved_rule_sha256': RULE_SHA,
        'failed_reference_receipt_sha256': digest(parent.receipt()),
        'failed_reference_raw_archive_sha256': parent.receipt()['raw_archive_sha256'],
        'failed_reference_source_commit': parent.receipt()['source_commit'],
        'failed_reference_closeout_sha256': parent.receipt()['closeout_sha256'],
        'configuration_sha256': digest(cfg), 'roles': list(ROLES),
        'training_cases': 108, 'development_cases': 36, 'eligible_training_states': len(fit),
        'rosters_sha256': digest({r: parent.scenes(r) for r in ROLES}),
        'rows_sha256': {r: digest(parent.rows(r)) for r in ROLES},
        'reference_audits_sha256': digest(audits), 'native_budgets_sha256': digest(budgets),
        'leakage_sha256': digest(leak), 'fitting_rows_sha256': digest(fit),
        'original_fitting_rows_sha256': digest(parent.fitting_rows()),
        'original_failed_draw_plan_sha256': digest(parent.original_draw_plan()),
        'fitting_draw_plan_sha256': digest(plan), 'protocol_sha256': digest(protocol(parent)),
        'sampling_distribution_changed': True, 'data_candidate_complete': True,
        'independent_raw_provider_audit_still_required': True,
        'model_training_admitted_by_this_receipt': False,
        'model_forwards': 0, 'optimizer_updates': 0, 'heldout_content_read': False}


def reference_export_names(parent):
    names = {'metadata/QWEN_PIN.json', 'metadata/QWEN_FILES.json', 'metadata/QWEN_BYTES.json',
             'LEAKAGE_AUDIT.json', 'training/ORIGINAL_FITTING_ROWS.json',
             'training/ORIGINAL_FITTING_DRAW_PLAN.json', 'training/FITTING_ROWS.json',
             'training/FITTING_DRAW_PLAN.json', 'FAILED_REFERENCE_PROTOCOL.json',
             'APPROVED_SAMPLING_DATA_RELEASE.json'}
    for role in ROLES:
        names |= {role + '/' + n for n in ['SCENES.json', 'EPISODES.json', 'ROWS.json',
                                          'COHORT_AUDIT.json', 'NATIVE_BUDGET.json']}
        names |= {role + '/case_exports/' + s['case_id'] + '.json' for s in parent.scenes(role)}
    return names


def materialize(context, *, persist=None):
    from .composite_bounded_sampling_source import require_context
    from .composite_bounded_sampling_rule import build_draw_plan
    require_context(context)
    context.require_runtime()
    parent = context.parent
    require_failed_parent(parent)
    cfg, profile = context.cfg(), context.profile()
    if digest(cfg) != digest(parent.cfg()): raise PermissionError('The approved change is sampling only')
    # Independently replay all original exported actions and exact native token
    # budgets before exposing a new sampler.  This never calls a teacher.
    result = parent.replay()
    exports = {}
    def emit(name, value):
        if name in exports: raise PermissionError('Every raw export must be written exactly once')
        exports[name] = deepcopy(value)
        if persist is not None: persist(name, value)
    old = parent.reference_exports()
    for name, value in old.items():
        if name in ['training/FITTING_ROWS.json', 'training/FITTING_DRAW_PLAN.json']: continue
        emit(name, value)
    emit('training/ORIGINAL_FITTING_ROWS.json', parent.fitting_rows())
    emit('training/ORIGINAL_FITTING_DRAW_PLAN.json', parent.original_draw_plan())
    emit('FAILED_REFERENCE_PROTOCOL.json', protocol(parent))
    fit = canonical_rows(parent)
    emit('training/FITTING_ROWS.json', fit)
    # This is the sole production draw-generation call.  It repeats genuine
    # source/runtime/parent guards inside the rule, before constructing its RNG.
    plan = build_draw_plan(parent.fitting_rows(), cfg, context)
    emit('training/FITTING_DRAW_PLAN.json', plan)
    if (plan.get('coverage_passed') is not True or plan.get('public_fixture_test_mode') is not False
            or plan.get('canonical_fitting_rows_sha256') != digest(fit)
            or plan.get('original_fitting_rows_sha256') != digest(parent.fitting_rows())):
        raise PermissionError('The complete approved sampling plan must pass without replacement or retry')
    release = build_release(profile, cfg, parent, result['audits'], result['budgets'],
                            result['leakage'], fit, plan)
    emit('APPROVED_SAMPLING_DATA_RELEASE.json', release)
    if set(exports) != reference_export_names(parent): raise PermissionError('Complete unchanged reference exports required')
    return {'exports': exports, 'audits': result['audits'], 'release': release, 'completed_cases': 144}

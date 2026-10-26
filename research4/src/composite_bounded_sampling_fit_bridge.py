"""Preserve the failed c37 core inside its own capsule; no failed PASS promotion."""
from .io_utils import digest,read,sha
from .composite_task_skill_source_package import frozen_source_package

FAILED_SOURCE='c37f4845896d2dc2824fc8dd2d840dc5fe627948'
FAILED_PACKAGE_SHA='d7bbf40c73473605780ca24b46a470b1f1d5eb47a71ade5f1be4e3691e561e2a'

def verify_failed_core(data,current):
    from .composite_bounded_sampling_reader import require_approved_sampling_data
    from .composite_bounded_behavior_fit_source import DATA_CORE
    from .composite_bounded_behavior_evaluation_semantics import fingerprint,require_frozen
    require_approved_sampling_data(data);failed=data.failed_reference();receipt=failed.receipt()
    if receipt['source_commit']!=FAILED_SOURCE or receipt['package_sha256']!=FAILED_PACKAGE_SHA:
        raise PermissionError('Only the immutable independently audited failed c37 reference may be reused')
    with frozen_source_package(failed.original_source(),FAILED_PACKAGE_SHA,FAILED_SOURCE,
            manifest_name='BOUNDED_BEHAVIOR_DATA_PACKAGE_FILES.json') as (source,_):
        release=read(source/'RELEASE.json')
        if digest(release)!=digest(failed.source_release()):
            raise PermissionError('The failed reference source release changed')
        core={name:release['files'][name] for name in sorted(DATA_CORE)}
        if any(sha(source/name)!=expected for name,expected in core.items()):
            raise PermissionError('An original DATA_CORE byte changed inside its frozen c37 capsule')
        require_frozen(fingerprint(source),current)
    return {'failed_source_commit':FAILED_SOURCE,'failed_package_sha256':FAILED_PACKAGE_SHA,
            'original_data_core_sha256':digest(core),'original_failed_status_preserved':True}

def require_actor_data(value):
    from .composite_bounded_behavior_data_reader import BoundedDataAudit,require_data
    if isinstance(value,BoundedDataAudit):return require_data(value)
    from .composite_bounded_sampling_reader import require_approved_sampling_data
    return require_approved_sampling_data(value)

def evaluation_lineage(model):
    """Explicit old/new typed data identity, frozen into evaluator stage and reports."""
    from .composite_bounded_sampling_fit_profile import MODE
    frozen=model.receipt();data=require_actor_data(model.data)
    mode=frozen.get('runtime_profile')
    result={'fit_mode':mode,'data_source_commit':data['source_commit'],
            'data_raw_archive_sha256':data['raw_archive_sha256']}
    if mode==MODE:
        from .composite_bounded_sampling_reader import require_approved_sampling_data
        data=require_approved_sampling_data(model.data)
        result.update(data_mode='bounded-behavior-approved-sampling-amendment-v1-data',
            approval_sha256=data['approval_sha256'],rule_sha256=data['rule_sha256'],
            approved_draw_plan_sha256=digest(model.data.fitting_draw_plan()))
    else:
        from .composite_bounded_behavior_fit_profile import MODE as OLD_MODE
        from .composite_bounded_behavior_data_reader import require_data
        if mode!=OLD_MODE:raise PermissionError('Unregistered frozen fit mode')
        require_data(model.data)
        result.update(data_mode='bounded-fictional-behavior-v4-data-export',
                      approval_sha256=None,rule_sha256=None,approved_draw_plan_sha256=None)
    return result

def fit_modules(mode):
    """Select only an explicit registered source/manifest/auditor pair."""
    from .composite_bounded_behavior_fit_profile import MODE as OLD_MODE
    from .composite_bounded_sampling_fit_profile import MODE as NEW_MODE
    import importlib
    if mode in [None,OLD_MODE]:prefix='composite_bounded_behavior_fit'
    elif mode==NEW_MODE:prefix='composite_bounded_sampling_fit'
    else:raise PermissionError('Unregistered frozen fit source mode')
    return tuple(importlib.import_module('.'+prefix+'_'+part,__package__)
                 for part in ['profile','source','audit'])

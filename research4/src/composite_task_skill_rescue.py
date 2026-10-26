"""Preserve an exited model child's owned raw prefix; never retry or infer success."""
from pathlib import Path
import os
import re
import shutil
from .io_utils import read, sha


def rescue_owned(output, stage, child, base='/kaggle/temp'):
    output = Path(output).resolve()
    marker = output / 'OUTPUT_OWNER.json'
    if child.get('child_exit_confirmed') is not True:
        return {'status':'child_exit_unknown_no_cleanup', 'scientific_result':None}
    if not marker.is_file():
        return {'status':'no_output_marker', 'scientific_result':None}
    owner = read(marker)
    scratch = Path(owner['scratch']).resolve()
    from .composite_bounded_behavior_fit_profile import MODE as BOUNDED_MODE
    if stage.get('version')==BOUNDED_MODE:
        from .composite_bounded_behavior_fit_store import stage_prefix
        expected_prefix=stage_prefix(stage)
    elif stage.get('phase') in ['risk_detector_transfer','risk_development']:
        from .composite_frozen_risk_profile import stage_prefix
        expected_prefix=stage_prefix(stage)
    elif stage.get('phase')=='task_development':
        from .composite_task_skill_inference import stage_prefix
        expected_prefix=stage_prefix(stage)
    else:
        if stage.get('phase') not in ['qualification','full_fit']:raise PermissionError('Unknown rescue phase')
        expected_prefix = 'task_skill_training_v1/' + stage['source_commit'] + '/'+stage['phase']+'_01'
    if (scratch.parent != Path(base).resolve() or not scratch.name.startswith('research4-v2-owned-stage-')
            or owner.get('source_commit') != stage['source_commit'] or owner.get('prefix') != expected_prefix
            or not re.fullmatch('[0-9a-f]{32}', owner.get('token',''))):
        raise PermissionError('Owned rescue path/source/token differs')
    if not scratch.exists():
        return {'status':'owned_scratch_absent', 'scientific_result':None}
    if read(scratch / 'OWNER.json') != owner:
        raise PermissionError('Actual scratch ownership marker differs')
    raw = (scratch / 'raw').resolve()
    if raw.parent != scratch or raw.name != 'raw':
        raise PermissionError('Unique raw evidence escaped owned scratch')
    # Only reproducible cache/reservation are removed after confirmed process
    # exit, freeing publication headroom. The original raw prefix is retained.
    cache = (scratch / 'cache').resolve()
    if cache.exists():
        if cache != scratch / 'cache':
            raise PermissionError('Cache symlink escaped its exact owned path')
        shutil.rmtree(cache)
    reserve = scratch / 'EXPORT_RESERVE.bin'
    if reserve.exists():
        if reserve.is_symlink():
            raise PermissionError('Reservation became a symlink')
        reserve.unlink()
    files = [p for p in sorted(raw.rglob('*')) if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    if total > 2 * stage['maximum_raw_evidence_bytes']:
        return {'status':'raw_prefix_exceeds_rescue_envelope_retained', 'raw_bytes':total,
                'scientific_result':None, 'counts_exact_at_exit':False}
    destination = output / 'rescue_raw_prefix'
    destination.mkdir(exist_ok=False)
    proofs = {}
    for file in files:
        if file.is_symlink() or not file.resolve().is_relative_to(raw):
            raise PermissionError('Raw rescue member escaped its owned directory')
        target = destination / file.relative_to(raw)
        target.parent.mkdir(parents=True, exist_ok=True)
        with file.open('rb') as source, target.open('xb') as stream:
            shutil.copyfileobj(source, stream, length=8 * 1024**2)
            stream.flush()
            os.fsync(stream.fileno())
        if sha(target) != sha(file):
            raise OSError('Rescued prefix readback differs')
        proofs[file.relative_to(raw).as_posix()] = {'sha256':sha(target),'bytes':target.stat().st_size}
    return {'status':'raw_prefix_rescued_and_original_retained', 'files':proofs,
            'scientific_result':None, 'counts_exact_at_exit':False, 'automatic_scientific_retry':False}

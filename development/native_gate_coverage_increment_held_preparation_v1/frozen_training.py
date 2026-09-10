"""Exact accepted training prerequisites, checked before any HELD provider work."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
MAIN=ROOT/'development/native_gate_coverage_increment_v1'
FREEZE_SHA256='23cba8db82645a441fa665933489c3178f864507a7150e4bb5c9dcd61ce3d1aa'
def need(ok,code):
    if not ok:raise ValueError(code)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def pinned(name,digest):
    path=MAIN/name
    need(path.resolve().is_relative_to(MAIN.resolve()) and path.is_file() and not path.is_symlink() and path.stat().st_size<=1024**2,'FROZEN_TRAINING_PATH')
    raw=path.read_bytes();need(sha(raw)==digest,'FROZEN_TRAINING_BYTES');return json.loads(raw)
def verify_frozen():
    f=pinned('FROZEN_TRAINING_ARTIFACTS.json',FREEZE_SHA256)
    need(f['accepted'] is True and f['role']=='FROZEN_BEFORE_HELD_PREPARATION' and f['training_fits_allowed']==0,'ACCEPTED_ARTIFACT_PAIR')
    for key in ('baseline','augmented','independent_review'):pinned(f[key]['path'],f[key]['sha256'])
    review=pinned(f['independent_review']['path'],f['independent_review']['sha256'])
    need(review['status']=='PASS_SAVED_TRAINING_CERTIFICATION_AND_CORRECTED_ARCHIVE','INDEPENDENT_TRAINING_ACCEPTANCE')
    release=pinned('root_release/CONSTRUCTION_RELEASE.json',f['construction_release_sha256'])
    result=pinned('construction_attempt_001/RESULT.json',f['result_sha256'])
    terminal=pinned('fit_owner_attempt_001/TERMINAL.json',f['terminal_sha256'])
    final=pinned('fit_owner_attempt_001/FINALIZATION.json',f['finalization_sha256'])
    need(release['approved'] is True and result['scientific_pass'] is True and result['status']=='BOTH_TRAINING_ARMS_PASS' and result['fits_attempted']==6,'BOTH_TRAINING_ARMS_ACCEPTED')
    need(terminal['exit_code']==0 and terminal['errors']==[] and all(terminal[k] is True for k in ('job_closed','job_empty','process_handle_closed','readers_closed','process_technical_complete','scientific_result_preserved','cleanup_within_budget')) and not terminal['timed_out'] and not terminal['output_limit'],'CLOSED_TRAINING_OWNER')
    need(final['terminal_sha256']==f['terminal_sha256'] and final['deadline_fault'] is False and final['storage_fault'] is False and final['elapsed_seconds_after_terminal']<=65 and final['cleanup_seconds_after_terminal']<=5,'CLOSED_TRAINING_FINALIZATION')
    return f

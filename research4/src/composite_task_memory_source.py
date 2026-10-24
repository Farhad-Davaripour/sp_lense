"""Source/data/started-failure binding for the new stateful qualification.

This is a separate registered memory exercise. It never resets the consumed
started fit, resumes at update13, or accepts a zero-forward repair allowance.
"""
import base64
from copy import deepcopy
import hashlib
from pathlib import Path
from .io_utils import read,sha,digest,verify_release
from .composite_task_memory_profile import MODE,PROFILE_FILE,expected_profile,QUALIFICATION_BOUNDS,RAW_PROFILE
from .composite_task_skill_training import (PLAN,PRE_FORWARD_PARENT,PRE_FORWARD_PARENT_SHA,
  data_input_names,validate_task_data_source,qualification_plan as old_qualification_plan)
from .composite_task_skill_source_package import frozen_source_package
from .composite_kaggle_v2_profile import identity

FAILURE_SOURCE='6c37f59b60ceff32c74b1c39f0a6b031d207e443'
FAILURE_PACKAGE_SHA='b18e06971434cda2b5312e670f5942c6b44688826103cb421f0f8e607fe83e5b'
FAILURE_RAW_SHA='bb7c211c2e47e235ebab5a1377401e5d1a1af0b1494c98d9eff2d3317b9dab7b'
FAILURE_CLOSEOUT_SHA='7ba0151a8c3b596e33c14c5f77962c63f5633ae40973631ac31ee2c092687591'
FAILURE_AUDIT_SHA='2de466c582b6cd5ef611d86388b31aab37a45ac88bfc5da41ed57c7ffbdef1fb'
FAILURE_FILES={'STARTED_FIT_SOURCE.r4b64','STARTED_FIT_EVIDENCE.r4b64','STARTED_FIT_CLOSEOUT.json',
 'STARTED_FIT_AUDIT.json','STARTED_FIT_CONTROLLER.json','STARTED_FIT_WORKER.json',
 'STARTED_FIT_TERMINAL_AX.txt','STARTED_FIT_RELEASE_AX.txt','STARTED_FIT_ACTUAL.ipynb','STARTED_FIT_PREPARED.ipynb'}
_KEY=object();_CACHE={}


class StartedFailureEvidence:
    def __init__(self,value,key=None):
        if key is not _KEY:raise PermissionError('Read the complete fixed started-failure evidence')
        self._key=key;self._value=deepcopy(value)
    def receipt(self):return deepcopy(self._value)


def read_started_failure(root):
    inputs=Path(root)/'inputs'
    if any(not (inputs/n).is_file() for n in FAILURE_FILES):
        raise PermissionError('Exact started failure source/raw/provider inputs required')
    if sha(inputs/'STARTED_FIT_CLOSEOUT.json')!=FAILURE_CLOSEOUT_SHA or sha(inputs/'STARTED_FIT_AUDIT.json')!=FAILURE_AUDIT_SHA:
        raise PermissionError('The original independently audited started failure must remain unchanged')
    stamp=digest({n:sha(inputs/n) for n in sorted(FAILURE_FILES)})
    if stamp in _CACHE:return _CACHE[stamp]
    closeout=read(inputs/'STARTED_FIT_CLOSEOUT.json');audit=read(inputs/'STARTED_FIT_AUDIT.json')
    if (closeout['scriptVersionId']!=361922334 or closeout['source_commit']!=FAILURE_SOURCE
            or closeout['owned_compute_released'] is not True or closeout['pre_execution_failure'] is not False
            or closeout['completed_update_journals']!=12 or closeout['exact_optimizer_sampler_RNG_resume_state_available'] is not False):
        raise PermissionError('This is a consumed started fit with no exact resume state')
    source_bytes=base64.b64decode((inputs/'STARTED_FIT_SOURCE.r4b64').read_bytes(),validate=True)
    raw=base64.b64decode((inputs/'STARTED_FIT_EVIDENCE.r4b64').read_bytes(),validate=True)
    if hashlib.sha256(raw).hexdigest()!=FAILURE_RAW_SHA:raise PermissionError('Complete original failure archive required')
    from .composite_v2_model_audit import EvidenceFiles
    with frozen_source_package(source_bytes,FAILURE_PACKAGE_SHA,FAILURE_SOURCE) as (source,mf):
        verify_release(source,FAILURE_SOURCE)
        for name in ['composite_controller_rank32_v2.json','composite_task_skill_v1.json',
                     'composite_qwen_two_t4_candidate_v1.json','composite_qwen_file_pin_fc05.json']:
            if sha(Path(root)/'config'/name)!=sha(source/'config'/name):
                raise PermissionError('Only the separate declared memory profile may change')
        files=EvidenceFiles(raw,FAILURE_RAW_SHA,4*1024**3)
        try:
            run=files.json('RUN_RECEIPT.json');journal=files.json('TRAINING_TERMINAL_JOURNAL.json')
            worker=read(inputs/'STARTED_FIT_WORKER.json');controller=read(inputs/'STARTED_FIT_CONTROLLER.json')
            if (run['source_commit']!=FAILURE_SOURCE or run['phase']!='full_fit' or run['status']!='failed'
                    or run['error_type']!='OutOfMemoryError' or run['public_fixture_test_mode'] is not False
                    or run['full_fit_completed'] is not False or run['counts_exact_at_controlled_exit'] is not True
                    or run['forward_counts']!={'attempted':25,'completed':25}):
                raise PermissionError('Original started-memory-failure facts differ')
            if digest(journal)!=digest(run['last_training_journal']) or [journal[k] for k in
                ['optimizer_steps_attempted','optimizer_steps_completed','backward_microsteps_completed']]!=[12,12,24]:
                raise PermissionError('Consumed physical work must remain visible')
            for name in run:
                if digest(run[name])!=digest(worker[name]):raise PermissionError('Sealed failure/controlled exit differ')
            if (controller['source_commit']!=FAILURE_SOURCE or controller['phase']!='full_fit'
                    or controller['status']!='worker_exited' or type(controller['exit_code']) is not int or controller['exit_code']!=1 or controller['timed_out'] is not False
                    or controller['child']['child_exit_confirmed'] is not True or type(controller['child']['exit_code']) is not int or controller['child']['exit_code']!=1
                    or worker['export']['durable_complete'] is not True or worker['cleanup']['owned_scratch_removed'] is not True):
                raise PermissionError('Actual failed-child export and cleanup required')
            terminal=(inputs/'STARTED_FIT_TERMINAL_AX.txt').read_text(encoding='utf-8')
            released=(inputs/'STARTED_FIT_RELEASE_AX.txt').read_text(encoding='utf-8')
            if ('scriptVersionId=361922334' not in terminal or 'failed to run after 4295.4s' not in terminal
                    or '0  Active Events' not in released or 'Draft Session off' not in released):
                raise PermissionError('Original provider terminal/release evidence required')
            code=lambda v:[''.join(c['source']) for c in v['cells'] if c['cell_type']=='code']
            prepared=code(read(inputs/'STARTED_FIT_PREPARED.ipynb'))
            expected=(source/'notebooks/composite_task_skill_qualification.py').read_text().replace(
              'SOURCE_COMMIT_PLACEHOLDER',FAILURE_SOURCE).replace('PACKAGE_MANIFEST_SHA256_PLACEHOLDER',sha(source.parent/'TASK_TRAINING_PACKAGE_FILES.json'))
            if prepared!=[expected] or code(read(inputs/'STARTED_FIT_ACTUAL.ipynb'))!=prepared:
                raise PermissionError('Actual failed-worker code readback differs from its frozen source')
            if any(n in files.manifest for n in ['behavior.pt','OPTIMIZER_FINAL.pt','GRADIENT_FINAL.pt']):
                raise PermissionError('Do not silently substitute a partial/resumed model')
            value={'run_id':361922334,'source_commit':FAILURE_SOURCE,'raw_archive_sha256':FAILURE_RAW_SHA,
              'closeout_sha256':FAILURE_CLOSEOUT_SHA,'failure_audit_sha256':FAILURE_AUDIT_SHA,
              'scientific_execution_started':True,'pre_execution_failure':False,'completed_optimizer_journals':12,
              'completed_backward_journals':24,'controlled_forward_counts':{'attempted':25,'completed':25},
              'owned_compute_released':True,'exact_resume_state_available':False,'original_identity_consumed':True}
        finally:files.close()
    proof=StartedFailureEvidence(value,_KEY);_CACHE[stamp]=proof
    return proof


def qualification_plan(commit):
    value=old_qualification_plan(commit)
    for name in ['pre_forward_parent_run_id','pre_forward_parent_closeout_sha256']:value.pop(name)
    value.update(version=MODE,notebook='farhad89/research4-task-memory-qualification-v3',
      optimizer_updates=3,sampling='fixed_training_only_long_short_long',raw_evidence_profile=RAW_PROFILE,
      previous_started_failure_run=361922334,previous_started_failure_closeout_sha256=FAILURE_CLOSEOUT_SHA,
      previous_failure_is_pre_execution=False,previous_completed_update_journals=12,
      original_fit_identity_reset=False,qualification_checkpoint_initializes_full_fit=False,
      placement_profile='qwen-two-t4-task-memory-balanced-v3',allocator_configuration='expandable_segments:True',
      initialization_mapping_version='per-device-seed202610212-placement22-42-v3',
      stateful_AdamW_overlap_and_size_switch_required=True,**QUALIFICATION_BOUNDS)
    return value


def full_fit_plan(commit,qualification):
    from .composite_task_skill_model_audit import require_stateful_qualification
    proof=require_stateful_qualification(qualification)
    value=qualification_plan(commit)
    value.update(phase='full_fit',notebook='farhad89/research4-task-skill-fit-v3',optimizer_updates=64,
      sampling='registered_seeded_training_sampler',seed=202610212,learning_rate=0.0005,weight_decay=0.0,clip_norm=1.0,
      raw_evidence_profile='task-full-state-raw-evidence-v1',maximum_raw_evidence_bytes=4*1024**3,
      physical_export_reserve_bytes=12*1024**3,hard_timeout_seconds=42000,worker_seconds=40800,export_reserve_seconds=900,
      destination='/kaggle/working/research4_campaign/runs/'+commit+'/task_skill_training_v1/full_fit_01',
      qualification_source_commit=proof['source_commit'],qualification_run_id=proof['provider_run_id'],
      qualification_raw_archive_sha256=proof['raw_archive_sha256'],checkpoint_initializes_full_fit=False,
      optimizer_sampler_or_RNG_warm_start=False,risk_examples_or_outcomes_used_for_fitting_or_selection=False,
      metadata_plan_alone_admits_launch=False,additional_started_full_fit_attempt=True,exact_resume_from12=False)
    return value


def validate_source(root,commit,stateful_archive=None):
    root=Path(root);stage=read(root/'inputs'/PLAN)
    required=data_input_names()|FAILURE_FILES
    if stage.get('phase')=='full_fit':
        from .composite_task_memory_parent import FILES as STATEFUL_PARENT_FILES
        required|=STATEFUL_PARENT_FILES
    cfg,evidence=validate_task_data_source(root,commit,required)
    if (digest(read(root/PROFILE_FILE))!=digest(expected_profile())
            or sha(root/'inputs'/PRE_FORWARD_PARENT)!=PRE_FORWARD_PARENT_SHA):
        raise PermissionError('Exact new profile and unchanged historical lineage required')
    failure=read_started_failure(root)
    if failure.receipt()['scientific_execution_started'] is not True:
        raise PermissionError('This stage must not borrow the zero-forward recovery allowance')
    expected=qualification_plan(commit)
    if stage.get('phase')=='full_fit':
        from .composite_task_memory_parent import read_stateful_parent
        expected=full_fit_plan(commit,read_stateful_parent(root,stateful_archive))
    if digest(stage)!=digest(expected) or read(root/'inputs/WORKSPACE.json')!=identity(stage['notebook']):
        raise PermissionError('Exact separate stateful qualification or fresh full-fit dependency required')
    return stage,cfg,evidence

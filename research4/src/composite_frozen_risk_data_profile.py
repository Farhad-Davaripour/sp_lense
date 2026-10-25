"""Development-only risk-reference source/role profile; no model or submission."""
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import re
from .io_utils import read,sha,digest,verify_release
from .composite_task_skill_profile import HUMAN,SELECTION_SHA
from .composite_kaggle_v2_profile import HUMAN as KAGGLE_HUMAN,HUMAN_SHA,identity
from .composite_task_skill_training import validate_recipe
from .composite_generalized_world import VERSION,CONDITIONS
from .composite_generalized_data import cohort_contract

MODE='option-a-frozen-risk-development-data-v1'
PROFILE='FROZEN_RISK_DATA_PROFILE.json'
NOTEBOOK='farhad89/research4-frozen-risk-data-v1'
DESIGN_SHA='95742c34f5955ef9851d6f8831dc9fdeca4764da199172adea0089de686a2918'
CORE_SHA='38b490581f28f27bb443bfa25a6304f6b936e5977a639370699ae9cd6e9973e9'
NAMESPACE='research4-option-a-frozen-risk-development-20261024-v1'
BOUNDS={'hard_timeout_seconds':1800,'worker_seconds':1200,'export_reserve_seconds':120,
        'maximum_raw_export_bytes':256*1024**2,'terminal_metadata_reserve_bytes':4*1024**2,
        'maximum_reference_states':900,'maximum_concurrent_owned_workers':1,'automatic_scientific_retries':0}
INPUTS={PROFILE,HUMAN,KAGGLE_HUMAN,'WORKSPACE.json','QWEN_PIN.json','MODERNBERT_PIN.json','PARENT_MODEL_INPUT_MANIFEST.json'}
_KEY=object()


def planned_roster(cfg):
    """Plan descriptor identities only; never create records, states or actions."""
    contract=cohort_contract(cfg,'development');rows=[]
    seed=hashlib.sha256((NAMESPACE+'|'+SELECTION_SHA+'|'+DESIGN_SHA).encode()).hexdigest()
    for group in contract['group_plan']:
        identity_seed=hashlib.sha256((VERSION+'|development|'+seed+'|'+str(group['group_index'])).encode()).hexdigest()
        group_id=digest({'version':VERSION,'role':'development','identity':identity_seed})
        for index,condition in enumerate(contract['conditions']):
            value={'identity_seed':identity_seed,'group_id':group_id,'template':group['template'],'domain':group['domain'],
              'condition':condition,'swap':bool(index%2),
              'id_digits':cfg['cohorts']['development_id_lengths'][group['group_index']%len(cfg['cohorts']['development_id_lengths'])],
              'hard_stop':24,'soft_stop':20,'mechanism':group['mechanism'],'task_structure':group['task_structure'],
              'role':'development','world_version':VERSION}
            rows.append({'case_id':digest(value),**value})
    if len(rows)!=36 or len({r['case_id'] for r in rows})!=36:raise PermissionError('All36 fixed risk cases required')
    return rows


def expected_profile(commit,cfg):
    if not isinstance(commit,str) or not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('Exact reviewed source commit required')
    from .composite_v2_controller import CONDITIONS as CONTROLS
    if cfg['controls']!=list(CONTROLS):raise PermissionError('All five original control meanings are required')
    return {'version':MODE,'phase':'risk_data_export','source_commit':commit,'platform':'kaggle','account':'farhad89',
      'notebook':NOTEBOOK,'visible_devices':'','scientific_GPU_count':0,'roles':['development'],
      'development_roster':planned_roster(cfg),'original_conditions':list(CONDITIONS),'original_controls':cfg['controls'],
      'expected_cases':36,'expected_risk_trajectories':180,'world_version':VERSION,
      'scope_A_selection_sha256':SELECTION_SHA,'kaggle_scope_sha256':HUMAN_SHA,
      'frozen_design_sha256':DESIGN_SHA,'scientific_semantic_core_sha256':CORE_SHA,
      'identity_namespace':NAMESPACE,'identities_resampled_after_outcomes':False,
      'reference_policy':'unchanged_existing_scripted_reference_for_read_only_transfer_and_replay',
      'reference_actions_may_include_existing_bounded_continuity_transitions':True,
      'behavior_training_exports_or_model_selection_admitted':False,
      'model_forwards':0,'optimizer_updates':0,'model_weights_downloaded':False,'heldout_content_read':False,
      'destination':'/kaggle/working/research4_campaign/runs/'+commit+'/risk_development_data_v1/export_01',**BOUNDS}


class RiskDataContext:
    def __init__(self,root,profile,cfg,key=None):
        if key is not _KEY:raise PermissionError('Validate the complete exact risk-reference source/input freeze')
        self._key=key;self.root=Path(root).resolve();self._profile=deepcopy(profile);self._cfg=deepcopy(cfg)
    def profile(self):return deepcopy(self._profile)
    def cfg(self):return deepcopy(self._cfg)
    def require_runtime(self):
        from .composite_runtime import require_scientific_runtime,platform
        require_scientific_runtime()
        if (platform()!='kaggle' or os.environ.get('CUDA_VISIBLE_DEVICES')!=''
                or not self.root.is_relative_to(Path('/kaggle/temp/research4_frozen_risk_inputs').resolve())
                or self.root.name!='source'
                or Path(os.environ.get('RESEARCH4_FROZEN_RISK_SOURCE_ROOT','')).resolve()!=self.root
                or os.environ.get('RESEARCH4_FROZEN_RISK_PROFILE_SHA256')!=sha(self.root/'inputs'/PROFILE)):
            raise PermissionError('Exact owned Kaggle CPU development-reference runtime required')
        return self


def validate_source(root,commit):
    root=Path(root).resolve();inputs=root/'inputs'
    from .composite_source_archive import verify_transport
    verify_transport(root,commit,MODE,'RISK_DATA_PACKAGE_FILES.json')
    manifest=read(inputs/'MANIFEST.json')
    if set(manifest)!=INPUTS:raise PermissionError('Only the exact development-risk metadata inputs are allowed')
    if any(Path(n).name!=n or sha(inputs/n)!=value for n,value in manifest.items()):raise PermissionError('Risk data input bytes changed')
    if sha(inputs/HUMAN)!=SELECTION_SHA or sha(inputs/KAGGLE_HUMAN)!=HUMAN_SHA:
        raise PermissionError('Bind the actual human OptionA selection and Kaggle scope')
    from .composite_v2_data_contract import ORIGINAL_MODEL_MANIFEST
    original=inputs/'PARENT_MODEL_INPUT_MANIFEST.json'
    if sha(original)!=ORIGINAL_MODEL_MANIFEST:raise PermissionError('Original model-file freeze changed')
    for name in ['QWEN_PIN.json','MODERNBERT_PIN.json']:
        if sha(inputs/name)!=read(original)[name]:raise PermissionError('Original complete model metadata changed')
    from .composite_frozen_probe import verify_original_model_pin
    verify_original_model_pin(read(inputs/'QWEN_PIN.json'))
    validate_recipe(read(root/'config/composite_controller_rank32_v2.json'))
    if sha(root/'config/composite_generalization_amendment_v2.json')!=DESIGN_SHA:
        raise PermissionError('Original risk questions/controls/configuration changed')
    from .composite_task_skill_semantics import semantic_fingerprint
    if semantic_fingerprint(root)!=CORE_SHA:raise PermissionError('Fixed fitted prompt/world/controller/detector/reference core changed')
    cfg=read(root/'config/composite_generalization_amendment_v2.json');profile=read(inputs/PROFILE)
    if digest(profile)!=digest(expected_profile(commit,cfg)):raise PermissionError('Risk-reference source/role/roster/bounds differ')
    if read(inputs/'WORKSPACE.json')!=identity(NOTEBOOK):raise PermissionError('Wrong risk-reference account/notebook')
    return RiskDataContext(root,profile,cfg,_KEY)

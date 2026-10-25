"""Frozen ordinary-adapter risk endpoint: fixed data, detector, model and arms."""
import os
from pathlib import Path
from .io_utils import read,sha,digest
from .composite_task_skill_inference import (TaskEvaluationContext,MODEL_INPUTS,read_frozen_model,_KEY as ACTOR_KEY)
from .composite_task_skill_package import require_frozen_task_package
from .composite_frozen_risk_data_parent import FILES as DATA_FILES,read_risk_parent
from .composite_frozen_risk_data_reader import require_data
from .composite_frozen_risk_data_profile import DESIGN_SHA
from .composite_frozen_risk_detector import FILES as DETECTOR_FILES,selected_detector
from .composite_v2_controller import CONDITIONS
from .composite_v2_development_contract import validate_roster

MODE='option-a-frozen-risk-development-two-t4-v1'
STAGE='FROZEN_RISK_EVALUATION_STAGE.json'
FILES=MODEL_INPUTS|DATA_FILES|DETECTOR_FILES|{STAGE,'MODERNBERT_PIN.json','RISK_EXECUTION_PLAN.json'}
_KEY=object()


def execution_plan(scenes,cfg):
    validate_roster(scenes,cfg)
    if cfg['controls']!=list(CONDITIONS):raise PermissionError('All original five control meanings required')
    ordered=sorted(scenes,key=lambda r:r['case_id'])
    first=[{'case_id':ordered[0]['case_id'],'arm':arm} for arm in CONDITIONS]
    tail=[{'case_id':row['case_id'],'arm':arm} for row in ordered[1:] for arm in CONDITIONS]
    groups=[first]+[tail[i:i+7] for i in range(0,len(tail),7)]
    parts=[{'index':i,'units':units,'case_ids':list(dict.fromkeys(u['case_id'] for u in units)),
      'expected_trajectories':len(units),'maximum_native_responses':24*len(units),
      'maximum_joint_backbone_forwards':len(units)*(24*160+25)+326} for i,units in enumerate(groups)]
    actual=[(u['case_id'],u['arm']) for p in parts for u in p['units']]
    expected=[(s['case_id'],a) for s in ordered for a in CONDITIONS]
    if actual!=expected or len(set(actual))!=180 or len(parts)!=26 or any(p['maximum_joint_backbone_forwards']>30000 for p in parts):
        raise PermissionError('Every exact original case/control unit must appear once within its forward envelope')
    return {'version':'frozen-risk-all180-execution26-v1','roster_sha256':digest(scenes),'partitions':parts,
      'expected_cases':36,'expected_trajectories':180,'original_controls':list(CONDITIONS),
      'first_partition_is_one_complete_five_arm_case':True,'pairing_preserved_by_exact_case_and_group_ids':True,
      'grouping_basis':'fixed case/control order before native risk outcomes',
      'context_tokens':6144,'maximum_new_tokens':160,'external_ticks':24,
      'maximum_backbone_forwards_per_job':30000,'optimizer_updates':0,'automatic_scientific_retries':0,
      'runtime_throughput_qualified':False,'metadata_admits_launch':False,'heldout_content_read':False}


def expected_stage(commit,model,data,cfg,phase,index=None):
    value=require_frozen_task_package(model);report=require_data(data);plan=execution_plan(data.scenes(),cfg)
    if value['evaluation_source_commit']!=commit:raise PermissionError('Exact common evaluation source/model freeze required')
    transfer=phase=='risk_detector_transfer'
    if not transfer and (phase!='risk_development' or type(index) is not int or not 0<=index<26):
        raise PermissionError('Fixed detector transfer or one exact full risk partition required')
    if transfer and index is not None:raise PermissionError('Transfer has no adaptive case selection')
    part=None if transfer else plan['partitions'][index]
    suffix='detector_transfer_01' if transfer else 'partition_'+str(index).zfill(2)
    return {'version':MODE,'source_commit':commit,'phase':phase,'platform':'kaggle','account':'farhad89',
      'notebook':'farhad89/research4-risk-detector-transfer-v1' if transfer else 'farhad89/research4-frozen-risk-development-v1',
      'objective':'frozen_reference_recognition_transfer' if transfer else 'frozen_ordinary_adapter_original_risk_observation',
      'visible_devices':'0' if transfer else '0,1','scientific_GPU_count':1 if transfer else 2,
      'partition':part,'risk_execution_plan_sha256':digest(plan),'risk_data_release_sha256':report['release_sha256'],
      'risk_data_raw_archive_sha256':report['raw_archive_sha256'],'frozen_model_package_sha256':digest(value),
      'checkpoint_sha256':value['checkpoint_sha256'],'detector_checkpoint_sha256':cfg['detector']['checkpoint_sha256'],
      'detector_threshold':0.9,'source_data_archive_sha256':value['data_archive_sha256'],
      'maximum_backbone_forwards':len(data.reference_rows()) if transfer else part['maximum_joint_backbone_forwards'],
      'hard_timeout_seconds':1800 if transfer else 42000,'worker_seconds':1200 if transfer else 40800,
      'export_reserve_seconds':120 if transfer else 900,'optimizer_updates':0,'maximum_concurrent_owned_workers':1,
      'automatic_scientific_retries':0,'raw_evidence_profile':'task-full-state-raw-evidence-v1',
      'maximum_raw_evidence_bytes':4*1024**3,'physical_export_reserve_bytes':12*1024**3,
      'destination':'/kaggle/working/research4_campaign/runs/'+commit+'/frozen_risk_v1/'+suffix,
      'full_task72_still_required':True,'full_risk180_still_required':True,'runtime_throughput_qualified':False,
      'heldout_content_read':False}


class FrozenRiskContext(TaskEvaluationContext):
    def __init__(self,root,model,data,stage,fit_archive,key=None):
        if key is not _KEY:raise PermissionError('Read the complete actual risk data/model/detector dependency chain')
        super().__init__(root,model,stage,fit_archive,ACTOR_KEY,stage_filename=STAGE)
        self._risk_key=key;self.data=data
    def require_runtime(self):
        from .composite_runtime import require_scientific_runtime,platform
        require_scientific_runtime()
        if (platform()!='kaggle' or os.environ.get('CUDA_VISIBLE_DEVICES')!=self.stage()['visible_devices']
                or Path(os.environ.get('RESEARCH4_TASK_EVALUATION_SOURCE_ROOT','')).resolve()!=self.root
                or not self.root.is_relative_to(Path('/kaggle/temp/research4_task_evaluation_inputs').resolve())
                or os.environ.get('RESEARCH4_TASK_EVALUATION_STAGE_SHA256')!=sha(self.root/'inputs'/STAGE)):
            raise PermissionError('Exact owned frozen-risk scientific runtime required')
        return self


def validate_source(root,commit,fit_archive,qualification_archive,risk_archive):
    root=Path(root).resolve();inputs=root/'inputs'
    model=read_frozen_model(root,commit,fit_archive,qualification_archive,MODE,FILES)
    data=read_risk_parent(inputs,risk_archive);cfg=read(root/'config/composite_generalization_amendment_v2.json')
    if sha(root/'config/composite_generalization_amendment_v2.json')!=DESIGN_SHA:raise PermissionError('Original risk questions/criteria changed')
    stage=read(inputs/STAGE);expected=expected_stage(commit,model,data,cfg,stage.get('phase'),
                                                   None if stage.get('partition') is None else stage['partition'].get('index'))
    if digest(stage)!=digest(expected):raise PermissionError('Frozen risk source/model/data/detector/phase/coverage bounds differ')
    if digest(read(inputs/'RISK_EXECUTION_PLAN.json'))!=digest(execution_plan(data.scenes(),cfg)):
        raise PermissionError('Complete original180 risk case/control plan changed')
    from .composite_kaggle_v2_profile import identity
    if read(inputs/'WORKSPACE.json')!=identity(stage['notebook']):raise PermissionError('Risk workspace/notebook changed')
    from .composite_task_skill_source_package import frozen_source_package
    from .composite_task_skill_fit_registry import registered_fit
    import base64
    value=require_frozen_task_package(model);fit=registered_fit(value['fit_source_commit'],value['fit_run_id'])
    encoded=base64.b64decode((inputs/'FIT_PARENT_SOURCE.r4b64').read_bytes(),validate=True)
    with frozen_source_package(encoded,fit['package_sha256'],fit['source_commit']) as (parent,_):
        if sha(inputs/'MODERNBERT_PIN.json')!=sha(parent/'inputs/MODERNBERT_PIN.json'):
            raise PermissionError('Original detector model-file pin changed')
    selected_detector(inputs,read(root/'config/composite_controller_rank32_v2.json')['modernbert'])
    return FrozenRiskContext(root,model,data,stage,fit_archive,_KEY)


def stage_prefix(stage):
    if stage.get('version')!=MODE:raise PermissionError('Exact frozen risk stage required')
    if stage.get('phase')=='risk_detector_transfer':suffix='detector_transfer_01'
    elif stage.get('phase')=='risk_development' and type(stage.get('partition',{}).get('index')) is int and 0<=stage['partition']['index']<26:
        suffix='partition_'+str(stage['partition']['index']).zfill(2)
    else:raise PermissionError('Only the complete fixed frozen risk/transfer namespaces')
    return 'frozen_risk_v1/'+stage['source_commit']+'/'+suffix

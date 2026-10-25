"""Frozen task-development runtime binding. No fitting or provider calls.

This route restores only the final checkpoint of a completed raw-audited fit.
It does not open risk/held-out cases or admit scientific validation by itself.
"""
import base64
from copy import deepcopy
from dataclasses import asdict
import os
from pathlib import Path
from .io_utils import read,sha,digest,verify_release
from .composite_task_skill_fit_parent import FIT_FILES,read_full_fit_parent
from .composite_task_skill_fit_registry import registered_fit
from .composite_task_skill_package import (read_task_development_inputs,restore_frozen_task_package,
                                           require_frozen_task_package)
from .composite_task_skill_validation import task_development_plan

MODE='ordinary-task-frozen-development-two-t4-v1'
STAGE='TASK_EVALUATION_STAGE.json'
FILES=FIT_FILES|{STAGE,'FROZEN_TASK_MODEL_PACKAGE.json','TASK_DATA_EVIDENCE.r4b64',
               'TASK_EVALUATION_PLAN.json','QWEN_TENSOR_METADATA.json','QWEN_PIN.json','WORKSPACE.json'}
_KEY=object()


class TaskEvaluationContext:
    def __init__(self,root,model,stage,fit_archive,key=None):
        if key is not _KEY:raise PermissionError('Read the complete source/model/data dependency chain')
        self._key=key;self.root=Path(root).resolve();self.model=model
        self._stage=deepcopy(stage);self.fit_archive=Path(fit_archive).resolve()
    def stage(self):return deepcopy(self._stage)


def expected_stage(commit,model,partition_index):
    value=require_frozen_task_package(model)
    if value['evaluation_source_commit']!=commit:raise PermissionError('Another evaluation source was selected')
    plan=task_development_plan(value['development_roster'])
    if type(partition_index) is not int or not 0<=partition_index<len(plan['partitions']):
        raise PermissionError('One exact whole paired task case is required')
    part=plan['partitions'][partition_index]
    return {'version':MODE,'source_commit':commit,'platform':'kaggle','account':'farhad89',
      'notebook':'farhad89/research4-task-development-v1','phase':'task_development',
      'visible_devices':'0,1','scientific_GPU_count':2,'partition':part,
      'task_development_plan_sha256':digest(plan),'frozen_model_package_sha256':digest(value),
      'fit_source_commit':value['fit_source_commit'],'fit_run_id':value['fit_run_id'],
      'checkpoint_sha256':value['checkpoint_sha256'],'full_task_trajectories_required':72,
      'original_risk_trajectories_still_required':180,
      'maximum_backbone_forwards':part['maximum_with_optional_native_parity'],
      'hard_timeout_seconds':42000,'worker_seconds':40800,'export_reserve_seconds':900,
      'maximum_concurrent_owned_workers':1,'automatic_scientific_retries':0,
      'optimizer_updates':0,'teacher_forcing':False,'decoder_repair':False,'policy_logit_masks':False,
      'maximum_tokens':6144,'maximum_new_tokens':160,'external_ticks':24,
      'heldout_content_read':False,'runtime_throughput_qualified':False}


def validate_source(root,commit,fit_archive,qualification_archive):
    root=Path(root).resolve();verify_release(root,commit);inputs=root/'inputs'
    manifest=read(inputs/'MANIFEST.json')
    if set(manifest)!=FILES:raise PermissionError('Exact task-evaluation inputs required; no risk or sealed cohorts')
    if any(Path(n).name!=n or sha(inputs/n)!=value for n,value in manifest.items()):
        raise PermissionError('Frozen task-evaluation input bytes changed')
    proof=read_full_fit_parent(inputs,fit_archive,qualification_archive)
    data=read_task_development_inputs(base64.b64decode((inputs/'TASK_DATA_EVIDENCE.r4b64').read_bytes(),validate=True))
    model=restore_frozen_task_package(read(inputs/'FROZEN_TASK_MODEL_PACKAGE.json'),
      read(inputs/'FIT_PARENT_AUDIT.json'),proof,data,commit)
    value=require_frozen_task_package(model);fit=registered_fit(value['fit_source_commit'],value['fit_run_id'])
    if fit['memory_profile_config'] is None:
        raise PermissionError('This evaluation runtime requires the separately qualified stateful placement')
    # Recheck configuration bytes against the immutable original full-fit source,
    # rather than trusting this new stage's configuration or a summary receipt.
    from .composite_task_skill_source_package import frozen_source_package
    encoded=base64.b64decode((inputs/'FIT_PARENT_SOURCE.r4b64').read_bytes(),validate=True)
    with frozen_source_package(encoded,fit['package_sha256'],fit['source_commit']) as (parent,_):
        configs=['composite_controller_rank32_v2.json','composite_task_skill_v1.json',
                 'composite_qwen_two_t4_candidate_v1.json','composite_qwen_file_pin_fc05.json']
        if fit['memory_profile_config'] is not None:configs.append(fit['memory_profile_config'])
        for name in configs:
            if sha(root/'config'/name)!=sha(parent/'config'/name):raise PermissionError('Agreed scientific configuration changed')
        for name in ['QWEN_TENSOR_METADATA.json','QWEN_PIN.json']:
            if sha(inputs/name)!=sha(parent/'inputs'/name):raise PermissionError('Original model geometry/file pins changed')
    stage=read(inputs/STAGE)
    if digest(stage)!=digest(expected_stage(commit,model,stage.get('partition',{}).get('index'))):
        raise PermissionError('Task-evaluation stage/partition/bounds differ')
    plan=task_development_plan(value['development_roster'])
    if digest(read(inputs/'TASK_EVALUATION_PLAN.json'))!=digest(plan):raise PermissionError('Complete paired72 plan changed')
    from .composite_kaggle_v2_profile import identity
    if read(inputs/'WORKSPACE.json')!=identity(stage['notebook']):raise PermissionError('Wrong evaluation account/notebook')
    return TaskEvaluationContext(root,model,stage,fit_archive,_KEY)


def require_runtime_binding(context):
    if not isinstance(context,TaskEvaluationContext) or context._key is not _KEY:
        raise PermissionError('Genuine source/model/data evaluation context required')
    from .composite_runtime import require_scientific_runtime,platform
    require_scientific_runtime()
    stage=context.stage();root=context.root
    if (platform()!='kaggle' or os.environ.get('CUDA_VISIBLE_DEVICES')!='0,1'
            or Path(os.environ.get('RESEARCH4_TASK_EVALUATION_SOURCE_ROOT','')).resolve()!=root
            or not root.is_relative_to(Path('/kaggle/temp/research4_task_evaluation_inputs').resolve())
            or root.name!='source' or read(root/'RELEASE.json')['commit']!=stage['source_commit']
            or os.environ.get('RESEARCH4_TASK_EVALUATION_STAGE_SHA256')!=sha(root/'inputs'/STAGE)):
        raise PermissionError('Exact owned two-T4 task-evaluation runtime/source required')
    return context


def authorized_load_plan(context,cfg):
    require_runtime_binding(context)
    core=read(context.root/'config/composite_controller_rank32_v2.json')
    from .composite_task_skill_training import validate_recipe,adapter_layout
    validate_recipe(core)
    if digest(cfg)!=digest(core['qwen']):raise PermissionError('Actor model/configuration differs from frozen task fit')
    from .composite_frozen_probe import verified_model_context
    result=verified_model_context(context.root,context.stage()['source_commit'],cfg,context.stage(),sha(context.root/'inputs'/STAGE))
    result['adapter_layout']=adapter_layout(result['metadata'],result['plan'])
    return result


def authorize_scene(scene,context):
    require_runtime_binding(context)
    cases=require_frozen_task_package(context.model)['development_roster']['cases']
    selected=context.stage()['partition']['case_ids']
    if not any(row['case_id'] in selected and digest(row)==digest(asdict(scene)) for row in cases):
        raise PermissionError('Only this exact frozen paired development case is admitted')
    return True


def inference_storage(layout,cuda,reserve):
    records=[]
    for index,count in enumerate(layout['device_parameters']):
        free,total=cuda.mem_get_info(index);storage=4*count
        if free<storage+reserve:raise MemoryError('Full FP32 inference adapter plus per-device reserve does not fit')
        records.append({'device':index,'free_bytes':free,'total_bytes':total,'adapter_weight_bytes':storage,'reserve_bytes':reserve})
    return {'devices':records,'optimizer_or_gradient_storage_allocated':False,'generation_capacity_proved':False}


def restore_checkpoint(actor,context):
    require_runtime_binding(context)
    if actor.task_evaluation_context is not context or actor.adapters:
        raise PermissionError('Restore exactly once into the owned idle frozen actor')
    value=require_frozen_task_package(context.model)
    import torch
    from importlib.metadata import version
    from .composite_task_skill_training import postload_runtime
    actor.task_evaluation_postload_proof=postload_runtime(actor,torch.cuda,torch.__version__,
      {name:version(name) for name in ['transformers','bitsandbytes','accelerate','safetensors','numpy','psutil']})
    from .composite_v2_model_audit import EvidenceFiles,tensor_state
    from .composite_training import tensor_digest
    from .composite_lora import load,trainables
    files=EvidenceFiles(context.fit_archive,value['fit_raw_archive_sha256'],4*1024**3)
    try:
        if files.manifest['behavior.pt']!=value['checkpoint_sha256']:raise PermissionError('Another fitted checkpoint was attached')
        saved=tensor_state(files.tensor_path('behavior.pt'))
        if (digest(saved['config'])!=digest(actor.cfg) or saved['placement']!={'name':'all_blocks','scope':'all_blocks'}
                or saved.get('native_interface_profile')!=actor.native_interface_profile
                or set(saved['weights'])!=set(value['checkpoint_tensor_hashes'])):
            raise PermissionError('Frozen fitted adapter/configuration/interface differs')
        for name,tensor in saved['weights'].items():
            if tensor_digest(tensor)!=value['checkpoint_tensor_hashes'][name]:raise PermissionError('Fitted checkpoint tensor bytes changed')
        actor._task_restoring=True
        try:actor.configure(saved['placement'],for_training=False)
        finally:actor._task_restoring=False
        load(actor.adapters,saved['weights'])
        parameters=trainables(actor.adapters)
        for parameter in parameters:parameter.requires_grad_(False)
        if any(p.requires_grad or p.grad is not None for p in parameters):
            raise PermissionError('Frozen task adapters must have no gradients or fitting parameters')
        actual={name+'.'+part:tensor_digest(getattr(module,part)) for name,module in actor.adapters.items() for part in ['A','B']}
        if actual!=value['checkpoint_tensor_hashes']:raise PermissionError('Live restored adapter differs from the raw-audited checkpoint')
        actor.task_checkpoint_proof={'checkpoint_sha256':value['checkpoint_sha256'],'live_tensor_hashes':actual,
          'tensors':len(actual),'all_adapter_requires_grad':False,'optimizer_state_loaded':False,'fitting_allowed':False}
    finally:files.close()


def verify_frozen_checkpoint(actor,context):
    require_runtime_binding(context)
    if getattr(actor,'task_evaluation_context',None) is not context:
        raise PermissionError('Another actor entered the frozen task stage')
    from .composite_training import tensor_digest
    from .composite_lora import trainables
    parameters=trainables(actor.adapters)
    if len(parameters)!=992 or any(p.requires_grad or p.grad is not None for p in parameters):
        raise PermissionError('Frozen task checkpoint gained gradients or lost projections')
    actual={name+'.'+part:tensor_digest(getattr(module,part)) for name,module in actor.adapters.items() for part in ['A','B']}
    value=require_frozen_task_package(context.model)
    if actual!=value['checkpoint_tensor_hashes']:raise PermissionError('Frozen task checkpoint bytes changed during evaluation')
    return {'checkpoint_sha256':value['checkpoint_sha256'],'live_tensor_hashes':actual,
            'all_adapter_gradients_none':True,'all_adapter_requires_grad':False,'live_recomputed':True}

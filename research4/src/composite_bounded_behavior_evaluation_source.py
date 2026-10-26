"""Complete frozen V4 actor source/model/data/partition authority."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import os,re,hashlib,zipfile
from .io_utils import read,sha,digest
from .composite_source_archive import verify_transport
from .composite_bounded_behavior_evaluation_profile import MODE,STAGE,MANIFEST,expected_stage,execution_plan,competence_roster
from .composite_bounded_behavior_evaluation_parent import FILES,read_parent,require_frozen_model
from .composite_bounded_behavior_profile import validate_preparation
from .composite_kaggle_v2_profile import HUMAN,HUMAN_SHA,identity

INPUTS=FILES|{STAGE,HUMAN,'WORKSPACE.json','QWEN_PIN.json','QWEN_TENSOR_METADATA.json','EVALUATION_PLAN.json','EVALUATION_ROSTER.json','PROBE_REGISTRATION.json'}
_KEY=object()


class BoundedEvaluationContext:
    def __init__(self,root,model,stage,roster,fit_archive,key=None):
        if key is not _KEY:raise PermissionError('Validate complete frozen source/model/data evidence')
        self._key=key;self.root=Path(root).resolve();self.model=model;self.fit_archive=Path(fit_archive).resolve()
        self._stage=deepcopy(stage);self._roster=deepcopy(roster)
    def stage(self):return deepcopy(self._stage)
    def roster(self):return deepcopy(self._roster)
    def require_runtime(self):
        from .composite_runtime import require_scientific_runtime,platform
        require_scientific_runtime()
        if (platform()!='kaggle' or os.environ.get('CUDA_VISIBLE_DEVICES')!='0,1'
            or Path(os.environ.get('RESEARCH4_BOUNDED_EVAL_SOURCE_ROOT','')).resolve()!=self.root
            or not self.root.is_relative_to(Path('/kaggle/temp/research4_bounded_eval_inputs').resolve())
            or self.root.name!='source' or os.environ.get('RESEARCH4_BOUNDED_EVAL_STAGE_SHA256')!=sha(self.root/'inputs'/STAGE)):
            raise PermissionError('Exact owned frozen two-T4 actor evaluation runtime required')
        return self
    def authorize_scene(self,scene,arm):
        self.require_runtime();descriptor=asdict(scene)
        if (arm not in ['base','always_on'] or descriptor['case_id'] not in self._stage['partition']['case_ids']
            or not any(digest(row)==digest(descriptor) for row in self._roster)):
            raise PermissionError('Only an exact complete paired case/arm from this partition is admitted')


def require_context(value):
    if not isinstance(value,BoundedEvaluationContext) or value._key is not _KEY:
        raise PermissionError('A source-bound opaque frozen actor context is required')
    require_frozen_model(value.model);return value


def notebook_code(root,commit,manifest_sha256):
    if not re.fullmatch('[0-9a-f]{40}',commit or '') or not re.fullmatch('[0-9a-f]{64}',manifest_sha256 or ''):
        raise PermissionError('Exact evaluation bootstrap bindings required')
    code=(Path(root)/'notebooks/composite_bounded_behavior_evaluation.py').read_text(encoding='utf-8')
    if code.count('SOURCE_COMMIT_PLACEHOLDER')!=1 or code.count('PACKAGE_MANIFEST_SHA256_PLACEHOLDER')!=1:
        raise PermissionError('One exact source and manifest bootstrap binding required')
    return code.replace('SOURCE_COMMIT_PLACEHOLDER',commit).replace('PACKAGE_MANIFEST_SHA256_PLACEHOLDER',manifest_sha256)


def validate_package(archive,bundle):
    from .composite_task_skill_source_package import validate_source_envelope
    bundle=Path(bundle);manifest=read(bundle/MANIFEST)
    if manifest.get('version')!=MODE:raise PermissionError('Exact actor source transport mode required')
    with zipfile.ZipFile(archive) as stream:
        infos=stream.infolist();validate_source_envelope(infos,MANIFEST)
        if {i.filename for i in infos}!=set(manifest['files'])|{MANIFEST}:
            raise PermissionError('Complete actor source/input ZIP roster required')
        for info in infos:
            name=info.filename;target=(bundle/name).resolve()
            if (info.is_dir() or info.flag_bits&1 or (info.external_attr>>16)&0o170000==0o120000
                or name.startswith('/') or '\\' in name or ':' in name
                or any(p in ['', '.', '..'] for p in name.split('/')) or not target.is_relative_to(bundle.resolve())):
                raise PermissionError('Unsafe actor source/input transport member')
            actual=hashlib.sha256()
            with stream.open(name) as source:
                for part in iter(lambda:source.read(8*1024**2),b''):actual.update(part)
            if actual.hexdigest()!=sha(target):raise PermissionError('Uploaded actor source/input bytes differ')


def validate_source(root,commit,fit_archive,qualification_archive,data_archive,alias_fit_archive,alias_qualification_archive):
    root=Path(root).resolve();verify_transport(root,commit,MODE,MANIFEST);inputs=root/'inputs'
    manifest=read(inputs/'MANIFEST.json')
    if set(manifest)!=INPUTS or any(sha(inputs/n)!=h for n,h in manifest.items()):
        raise PermissionError('Complete exact frozen actor source/input transport required')
    if sha(inputs/HUMAN)!=HUMAN_SHA:raise PermissionError('Selected Kaggle account authority required')
    model=read_parent(root,fit_archive,qualification_archive,data_archive,alias_fit_archive,alias_qualification_archive)
    from .composite_bounded_behavior_evaluation_semantics import require_frozen
    require_frozen(model.semantics,root)
    cfg=validate_preparation(read(root/'config/composite_bounded_behavior_v4.json'))
    if digest(model.cfg)!=digest(read(root/'config/composite_controller_rank32_v2.json')):
        raise PermissionError('Original complete frozen model/recipe changed')
    for name in ['QWEN_PIN.json','QWEN_TENSOR_METADATA.json']:
        if digest(read(inputs/name))!=digest(model.inputs[name]):raise PermissionError('Original model file/geometry binding changed')
    stage=read(inputs/STAGE);endpoint=stage.get('endpoint')
    roster=model.data.scenes('development') if endpoint=='behavior' else competence_roster(cfg)
    expected=expected_stage(commit,model,model.data,cfg,endpoint,stage.get('partition',{}).get('index'))
    from .composite_bounded_behavior_evaluation_profile import PROBE_REGISTRATION
    if (digest(stage)!=digest(expected) or digest(read(inputs/'EVALUATION_PLAN.json'))!=digest(execution_plan(roster,endpoint))
        or digest(read(inputs/'EVALUATION_ROSTER.json'))!=digest(roster)
        or digest(read(inputs/'PROBE_REGISTRATION.json'))!=digest(PROBE_REGISTRATION)
        or read(inputs/'WORKSPACE.json')!=identity(stage['notebook'])):
        raise PermissionError('Frozen evaluation cohort/arms/criteria/partition/ordinary-probe registration changed')
    return BoundedEvaluationContext(root,model,stage,roster,fit_archive,_KEY)


def authorized_load_plan(context,cfg):
    require_context(context);context.require_runtime()
    if digest(cfg)!=digest(context.model.cfg['qwen']):raise PermissionError('Frozen actor model configuration changed')
    from .composite_frozen_probe import verified_model_context
    from .composite_task_skill_training import adapter_layout
    result=verified_model_context(context.root,context.stage()['source_commit'],cfg,context.stage(),sha(context.root/'inputs'/STAGE))
    result['adapter_layout']=adapter_layout(result['metadata'],result['plan']);return result

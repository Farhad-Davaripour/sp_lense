"""Exact complete V4 fitting source/data/qualification authority, separate from Option A."""
from pathlib import Path
import re,hashlib,zipfile
from .io_utils import read,sha,digest
from .composite_source_archive import verify_transport
from .composite_kaggle_v2_profile import identity,HUMAN as KAGGLE_HUMAN,HUMAN_SHA
from .composite_bounded_sampling_fit_profile import MODE,MANIFEST,PLAN,qualification_plan,full_fit_plan
from .composite_bounded_sampling_fit_parent import DATA_FILES,QUAL_FILES,read_data_parent,read_qualification_parent
from .composite_bounded_behavior_profile import validate_preparation
from .composite_task_skill_training import validate_recipe

BASE_INPUTS={PLAN,'PARENT_MODEL_INPUT_MANIFEST.json','QWEN_PIN.json','MODERNBERT_PIN.json',
 'QWEN_TENSOR_METADATA.json','WORKSPACE.json',KAGGLE_HUMAN}
DATA_CORE={'src/composite_bounded_behavior_interface.py','src/composite_bounded_behavior_world.py',
 'src/composite_bounded_behavior_data.py','src/composite_bounded_behavior_data_profile.py',
 'src/composite_bounded_behavior_data_reader.py','src/composite_bounded_behavior_parent.py',
 'src/composite_bounded_behavior_runtime.py','src/composite_bounded_behavior_data_admission.py',
 'src/composite_bounded_behavior_profile.py','src/composite_bounded_behavior_data_worker.py',
 'src/composite_record_programs.py','src/composite_generalized_world.py',
 'src/composite_generalized_teacher.py','src/composite_generalized_data.py',
 'src/composite_generalized_audit.py','src/composite_world.py',
 'src/composite_generalized_tasks.py','src/composite_generalized_interface.py',
 'src/composite_generalized_novelty.py','src/composite_design_v2.py',
 'src/composite_design_v2_native.py','src/composite_interface.py','src/native27_tools.py',
 'config/composite_bounded_behavior_v4.json','config/composite_controller_rank32_v2.json',
 'config/composite_qwen_two_t4_candidate_v1.json'}
CONTROLLER_CONFIG_SHA='63c0e2706c6125ee1df760a6a195777b6ca41a5d44f955ea84604dc7767ad39d'
PLACEMENT_CONFIG_SHA='b7f9bd60bb3a77c50382b20bc0e4018fe35df1f4ad3706f4d332bb5799d5cac7'


def validate_configuration(cfg,placement):
    validate_recipe(cfg)
    if digest(cfg)!=CONTROLLER_CONFIG_SHA or digest(placement)!=PLACEMENT_CONFIG_SHA:
        raise PermissionError('Complete immutable model/recipe/placement configuration required')
    return cfg


def validate_fit_package(archive,bundle):
    """Compare the actual fit upload ZIP to every frozen source/input byte.

    The new fit manifest alone has a256MiB source-transport envelope because
    it retains both complete audited CPU and qualifier source parents. The
    existing data and historical transport formats keep their128MiB limits.
    """
    from .composite_task_skill_source_package import validate_source_envelope
    bundle=Path(bundle);manifest=read(bundle/MANIFEST)
    if manifest.get('version')!=MODE or not re.fullmatch('[0-9a-f]{40}',manifest.get('source_commit','')):
        raise PermissionError('Exact registered V4 fit source manifest required')
    with zipfile.ZipFile(archive) as stream:
        infos=stream.infolist();validate_source_envelope(infos,MANIFEST)
        names=[i.filename for i in infos]
        if set(names)!=set(manifest['files'])|{MANIFEST}:
            raise PermissionError('Complete V4 fit source/input archive roster required')
        for info in infos:
            name=info.filename
            if (info.is_dir() or info.flag_bits&1 or (info.external_attr>>16)&0o170000==0o120000
                    or name.startswith('/') or '\\' in name or ':' in name
                    or any(p in ['', '.', '..'] for p in name.split('/'))):
                raise PermissionError('Only safe ordinary unencrypted V4 fit source files are readable')
            target=(bundle/name).resolve()
            if not target.is_relative_to(bundle.resolve()):raise PermissionError('Fit package escaped its source tree')
            value=hashlib.sha256()
            with stream.open(name) as source:
                for part in iter(lambda:source.read(8*1024**2),b''):value.update(part)
            if value.hexdigest()!=sha(target):
                raise PermissionError('Actual compressed fit upload differs from reviewed source/input bytes')


def notebook_code(root,commit,manifest_sha256):
    if not re.fullmatch('[0-9a-f]{40}',commit or '') or not re.fullmatch('[0-9a-f]{64}',manifest_sha256 or ''):
        raise PermissionError('Exact reviewed source/manifest bootstrap bindings required')
    code=(Path(root)/'notebooks/composite_bounded_sampling_fit.py').read_text(encoding='utf-8')
    if code.count('SOURCE_COMMIT_PLACEHOLDER')!=1 or code.count('PACKAGE_MANIFEST_SHA256_PLACEHOLDER')!=1:
        raise PermissionError('Exactly one occurrence of each bootstrap source binding required')
    return code.replace('SOURCE_COMMIT_PLACEHOLDER',commit).replace('PACKAGE_MANIFEST_SHA256_PLACEHOLDER',manifest_sha256)


def validate_source(root,commit,qualification_archive=None,*,data_archive=None,
                    alias_fit_archive=None,alias_qualification_archive=None):
    root=Path(root).resolve();verify_transport(root,commit,MODE,MANIFEST)
    stage=read(root/'inputs'/PLAN);inputs=root/'inputs'
    required=BASE_INPUTS|(QUAL_FILES if stage.get('phase')=='full_fit' else DATA_FILES)
    manifest=read(inputs/'MANIFEST.json')
    if set(manifest)!=required or any(sha(inputs/n)!=h for n,h in manifest.items()):
        raise PermissionError('Complete exact V4 data/model/source/phase inputs required')
    if sha(inputs/KAGGLE_HUMAN)!=HUMAN_SHA:raise PermissionError('Existing selected Kaggle authority required')
    validate_preparation(read(root/'config/composite_bounded_behavior_v4.json'))
    cfg=validate_configuration(read(root/'config/composite_controller_rank32_v2.json'),
      read(root/'config/composite_qwen_two_t4_candidate_v1.json'))
    from .composite_v2_data_contract import ORIGINAL_MODEL_MANIFEST
    if sha(inputs/'PARENT_MODEL_INPUT_MANIFEST.json')!=ORIGINAL_MODEL_MANIFEST:
        raise PermissionError('Original complete model-file pins required')
    for name in ['QWEN_PIN.json','MODERNBERT_PIN.json']:
        if sha(inputs/name)!=read(inputs/'PARENT_MODEL_INPUT_MANIFEST.json')[name]:
            raise PermissionError('Original model/tokenizer pins changed')
    if stage.get('phase')=='full_fit':
        parent=read_qualification_parent(root,qualification_archive,data_archive=data_archive,
          alias_fit=alias_fit_archive,alias_qualification=alias_qualification_archive)
        data=parent.data
        expected=full_fit_plan(commit,data,parent)
    elif stage.get('phase')=='qualification':
        data=read_data_parent(root,data_archive,alias_fit_archive,alias_qualification_archive)
        expected=qualification_plan(commit,data)
    else:raise PermissionError('Only the separate approved-sampler qualifier/fresh full fit is registered')
    from .composite_bounded_sampling_fit_bridge import verify_failed_core
    verify_failed_core(data,root)
    if digest(stage)!=digest(expected) or read(inputs/'WORKSPACE.json')!=identity(expected['notebook']):
        raise PermissionError('Exact new-data qualification or fresh-fit parent/plan required')
    return stage,cfg,data

"""Task-only fitting boundaries and two-device storage planning.

Metadata/token planning does not admit a GPU run or claim backward capacity.
No continuity labels, examples or outcomes select fitting observations.
"""
from copy import deepcopy
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile
from .io_utils import digest, read, sha, verify_release
from .composite_generalized_interface import PROFILE, input_ids, action_text
from .composite_task_skill_profile import SELECTION_SHA, HUMAN
from .composite_kaggle_v2_profile import HUMAN as KAGGLE_HUMAN, HUMAN_SHA, identity

MODE = 'ordinary-task-skill-two-t4-training-postload-v2'
PRE_FORWARD_PARENT = 'GPU_PRE_FORWARD_PARENT_FAILURE.json'
PRE_FORWARD_PARENT_SHA = '6e58c1e764ed2e3ca1f08609b406742ab65221658ad0a4a5a9faf87fd711b9b8'
PLAN = 'TASK_TRAINING_PLAN.json'
DATA_ARCHIVE = 'TASK_DATA_EVIDENCE.r4b64'
DATA_SOURCE = '5cbebf5626007da0ef1f79c9a1535ad89c536ab7'
DATA_ARCHIVE_SHA = '61831507ef6b295ccf14bf8725606f8cf871d6ff2d2701117c5d228bddb48a17'
DATA_RELEASE_SHA = '318a8308e38a88cde7eb5bcf9e96d979cace4183f9772fcd19849415ec482570'
DATA_ROWS_SHA = 'bed325139563003ef187ea9fc37f412250822aa2bd1423ca10538c8baf3f1872'
DATA_AUDIT_SHA = '04a39d5fe02182337ddd846f01248796e0037251e0b6485af169b2bd6bd9a40a'
DATA_CLOSEOUT_SHA = '3a9c34c3dda38cb0f36aca37212fc52c9109cdd7d7acca31a05804fb11d4fa0a'
TASK_CONFIG_SHA = '3b5a7bd512d5ccb9dd060029e57e024f4111884c7ceb3239a0d181340e53a26c'
PARAMETERS = 233455616
PROJECTIONS = 496
_KEY = object()


class TaskDataEvidence:
    def __init__(self, rows, release, key=None, source_release_sha256=None):
        if key is not _KEY:
            raise PermissionError('Read the exact independently audited task data archive')
        self._rows = deepcopy(rows)
        self._release = deepcopy(release)
        self.source_release_sha256 = source_release_sha256
        self._key = key

    def rows(self):
        return deepcopy(self._rows)

    def release(self):
        return deepcopy(self._release)


def read_data(raw):
    if hashlib.sha256(raw).hexdigest() != DATA_ARCHIVE_SHA:
        raise PermissionError('Only the complete audited option-A dataset may enter fitting')
    with zipfile.ZipFile(io.BytesIO(raw)) as stream:
        names = stream.namelist()
        if len(names) != len(set(names)) or sum(i.file_size for i in stream.infolist()) > 256 * 1024**2:
            raise ValueError('Duplicate or oversized task data archive')
        if any(n.startswith('/') or '..' in n.split('/') or '\\' in n or ':' in n or 'sealed' in n.lower() or 'heldout' in n.lower() for n in names):
            raise PermissionError('Unsafe/sealed task input')
        manifest = json.loads(stream.read('MANIFEST.json'))
        if set(names) != set(manifest) | {'MANIFEST.json'}:
            raise ValueError('Unaccounted task input member')
        for name, expected in manifest.items():
            if hashlib.sha256(stream.read(name)).hexdigest() != expected:
                raise ValueError('Raw task evidence changed')
        release_bytes = stream.read('TASK_DATA_RELEASE.json')
        row_bytes = stream.read('training/FITTING_ROWS.json')
        if hashlib.sha256(release_bytes).hexdigest() != DATA_RELEASE_SHA or hashlib.sha256(row_bytes).hexdigest() != DATA_ROWS_SHA:
            raise PermissionError('Exact task release/row freeze required')
        release, rows = json.loads(release_bytes), json.loads(row_bytes)
        if (release['source_commit'] != DATA_SOURCE or release['task_dataset_admitted'] is not True
                or release['frozen_risk_dataset_generated'] is not False or release['heldout_content_read'] is not False
                or release['audits']['training']['complete_cases'] != 108
                or release['audits']['development']['complete_cases'] != 36):
            raise PermissionError('Unadmitted, incomplete or risk-target data')
        validate_rows(rows, release)
        freeze = json.loads(stream.read('PRECALL_FREEZE.json'))
        return TaskDataEvidence(rows, release, _KEY, freeze['source_release_sha256'])


def validate_rows(rows, release):
    keys = {'state_id', 'text', 'teacher_action', 'terminal', 'training_eligible', 'task_only',
            'case_id', 'group_id', 'mechanism', 'split'}
    if len(rows) != 1188 or len(rows) != release['audits']['training']['complete_states']:
        raise ValueError('Every audited task-only training observation is required')
    if len({row['state_id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate fitting observation')
    if {row['case_id'] for row in rows} != set(release['audits']['training']['case_ids']):
        raise PermissionError('A fitting observation crossed cohort identity')
    for row in rows:
        if (set(row) != keys or row['split'] != 'training' or row['terminal'] is not False
                or row['training_eligible'] is not True or row['task_only'] is not True
                or row['teacher_action'] is None or row['group_id'] not in release['audits']['training']['group_ids']):
            raise PermissionError('No risk/evaluation/post-task row or fake gate label may enter task fitting')
    return rows


def token_plan(rows, tokenizer, cfg):
    """Audit all real task rows, with no inherited gate-positive eligibility."""
    records = []
    for row in rows:
        if row.get('split') != 'training' or row.get('task_only') is not True or row.get('training_eligible') is not True:
            raise PermissionError('Task-only training observations required')
        prefix = input_ids(tokenizer, row['text'])
        target = tokenizer.encode(action_text(row['teacher_action']), add_special_tokens=False) + [tokenizer.eos_token_id]
        if tokenizer.eos_token_id is None or len(target) > 160 or len(prefix) + len(target) - 1 > 6144:
            raise ValueError('Full task input/target exceeds the unchanged context; never truncate')
        records.append({'state_id': row['state_id'], 'input_token_ids': prefix, 'target_token_ids': target,
                        'combined_tokens': len(prefix) + len(target) - 1})
    if len(records) < 2 or len({r['state_id'] for r in records}) != len(records):
        raise ValueError('Two distinct complete task observations required')
    longest = sorted(records, key=lambda r: (-r['combined_tokens'], r['state_id']))[:2]
    return {'eligible_training_states': len(rows), 'source_training_states': len(rows), 'records': records,
            'qualification_state_ids': [r['state_id'] for r in longest],
            'selection': 'longest_complete_task_only_examples_not_risk_or_validation',
            'model_forwards': 0, 'heldout_content_read': False}


def adapter_layout(metadata, plan):
    from .composite_placement import metadata_bounds, parameter_device
    metadata_bounds(metadata, plan)
    counts = [0, 0]
    projections = {}
    for index in range(64):
        suffixes = ['mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']
        suffixes += (['self_attn.' + p for p in ['q_proj', 'k_proj', 'v_proj', 'o_proj']]
                     if index % 4 == 3 else ['linear_attn.' + p for p in ['in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a', 'out_proj']])
        for suffix in suffixes:
            name = 'layers.' + str(index) + '.' + suffix
            full = 'model.language_model.' + name + '.weight'
            shape = metadata['tensors'][full]['shape']
            if len(shape) != 2:
                raise ValueError('Pinned projection geometry differs')
            device = parameter_device(full, plan)
            count = 32 * sum(shape)
            projections[name] = {'shape': shape, 'device': device, 'parameters': count}
            counts[device] += count
    if len(projections) != PROJECTIONS or sum(counts) != PARAMETERS:
        raise ValueError('All64 blocks/rank32/496 projections must remain present')
    return {'projections': projections, 'projection_count': len(projections), 'adapter_parameters': sum(counts),
            'device_parameters': counts, 'FP32_weights_gradients_AdamW_bytes': [c * 16 for c in counts],
            'activation_memory_included': False, 'actual_training_capacity_qualified': False}


def memory_admission(layout, cuda, reserve):
    result = []
    for index, storage in enumerate(layout['FP32_weights_gradients_AdamW_bytes']):
        free, total = cuda.mem_get_info(index)
        result.append({'device': index, 'free_bytes': free, 'total_bytes': total,
                       'adapter_gradient_AdamW_storage_bytes': storage, 'activation_reserve_bytes': reserve})
        if free < storage + reserve:
            raise MemoryError('The full adapter/gradient/optimizer and reserve must fit on each device separately')
    return {'devices': result, 'actual_backward_fit_verified': False}


def live_resources(cuda):
    """Read available storage including idle allocator cache; this proves no fit.

    Counting only driver-free bytes would reject a healthy graph whose unused
    torch cache can satisfy its next allocation. Live tensors are never freed.
    """
    if not cuda.is_available() or cuda.device_count() != 2:
        raise RuntimeError('The owned task-training pair changed')
    records = []
    for index in [0, 1]:
        free, total = cuda.mem_get_info(index)
        allocated, reserved = cuda.memory_allocated(index), cuda.memory_reserved(index)
        idle = max(0, reserved - allocated)
        available = free + idle
        records.append({'device': index, 'free_bytes': free, 'total_bytes': total,
                        'allocated_bytes': allocated, 'reserved_bytes': reserved,
                        'idle_allocator_bytes': idle, 'available_storage_estimate_bytes': available})
        if available < 512 * 1024**2:
            raise MemoryError('Live tensors consume the protected export headroom')
    return {'devices': records, 'actual_backward_fit_verified': False}


def postload_runtime(actor,cuda,torch_version,packages):
    """Verify an already loaded idle base without charging its weights twice."""
    from .composite_frozen_probe import verify_software
    software=verify_software(torch_version,packages)
    if actor.adapters or actor.generating or not cuda.is_available() or cuda.device_count()!=2:
        raise PermissionError('Post-load verification requires the idle unadapted task base and both devices')
    devices=[]
    for index in [0,1]:
        if cuda.get_device_name(index)!='Tesla T4':raise PermissionError('The pinned task device pair changed')
        before=cuda.memory_allocated(index)
        with cuda.device(index):cuda.empty_cache()
        if cuda.memory_allocated(index)!=before:raise RuntimeError('Idle reclamation changed live tensor allocation')
        free,total=cuda.mem_get_info(index)
        devices.append({'index':index,'name':cuda.get_device_name(index),'free_bytes':free,
                        'total_bytes':total,'live_allocated_bytes':before})
    live_resources(cuda)
    return {'devices':devices,'software':software,'base_already_loaded':True,
            'preload_weight_lower_bound_reapplied':False,
            'full_adapter_optimizer_admission':'performed separately by Actor.configure before installation',
            'actual_backward_fit_verified':False}


def validate_recipe(cfg):
    expected = {'steps': 64, 'accumulation': 2, 'learning_rate': 0.0005, 'seed': 202610212, 'clip_norm': 1.0}
    if any(type(cfg['behavior_training'].get(k)) is not type(v) or cfg['behavior_training'][k] != v for k, v in expected.items()):
        raise PermissionError('Full64-update task recipe required')
    q = cfg['qwen']
    if ((q['repository'], q['revision'], q['layers'], q['hidden_size'], q['rank'], q['alpha'])
            != ('Qwen/Qwen3.5-27B', 'fc05daec18b0a78c049392ed2e771dde82bdf654', 64, 5120, 32, 64)
            or q['maximum_tokens'] != 6144 or q['maximum_new_tokens'] != 160
            or q['quantization'] != 'nf4-double-bfloat16' or q['gradient_checkpointing'] is not True):
        raise PermissionError('Complete agreed architecture/precision/context must remain unchanged')
    bert = cfg['modernbert']
    if (bert['layers'], bert['lora_last_layers'], bert['rank'], bert['alpha'], bert['scope'], bert['maximum_tokens']) != (22, 22, 32, 64, 'all_blocks', 2048):
        raise PermissionError('The all22-block rank32/alpha64 detector architecture must be retained')
    return True


def qualification_plan(commit):
    if not isinstance(commit, str) or not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('Exact reviewed source commit required')
    notebook = 'farhad89/research4-task-skill-qualification-v2'
    return {'version': MODE, 'source_commit': commit, 'phase': 'qualification',
            'notebook': notebook, 'platform': 'kaggle', 'account': 'farhad89',
            'objective': 'ordinary_authorized_task_skills_only',
            'pre_forward_parent_run_id':361892579,'pre_forward_parent_closeout_sha256':PRE_FORWARD_PARENT_SHA,
            'source_data_archive_sha256': DATA_ARCHIVE_SHA,
            'source_data_release_sha256': DATA_RELEASE_SHA,
            'source_training_rows': 1188, 'optimizer_updates': 1, 'accumulation': 2,
            'sampling': 'longest_complete_task_only_examples',
            'native_interface_profile': PROFILE,
            'qualification_checkpoint_initializes_full_fit': False,
            'visible_devices': '0,1', 'scientific_GPU_count': 2,
            'kernel_profile': 'reference', 'raw_evidence_profile': 'task-full-state-raw-evidence-v1',
            'maximum_raw_evidence_bytes': 4 * 1024**3, 'physical_export_reserve_bytes': 12 * 1024**3,
            'hard_timeout_seconds': 7200, 'worker_seconds': 6600, 'export_reserve_seconds': 600,
            'maximum_backbone_forwards': 30000, 'maximum_concurrent_owned_workers': 1,
            'automatic_scientific_retries': 0, 'heldout_content_read': False,
            'destination': '/kaggle/working/research4_campaign/runs/' + commit + '/task_skill_training_v1/qualification_01'}


def data_input_names():
    return {PLAN, HUMAN, KAGGLE_HUMAN, DATA_ARCHIVE, PRE_FORWARD_PARENT, 'TASK_DATA_AUDIT.json',
                'TASK_DATA_CLOSEOUT.json', 'PARENT_DATA_SOURCE_RELEASE.json',
                'PARENT_MODEL_INPUT_MANIFEST.json', 'WORKSPACE.json', 'QWEN_PIN.json',
                'MODERNBERT_PIN.json', 'QWEN_TENSOR_METADATA.json'}


def validate_task_data_source(root,commit,required):
    """Shared exact source/data/pin checks, without creating phase authority."""
    root=Path(root);verify_release(root,commit)
    manifest = read(root / 'inputs/MANIFEST.json')
    if set(manifest) != required:
        raise PermissionError('Exact task-training input set required; no risk, qualification weights or sealed inputs')
    for name, expected in manifest.items():
        if Path(name).name != name or sha(root / 'inputs' / name) != expected:
            raise PermissionError('Frozen task-training input changed')
    for name, expected in [(HUMAN, SELECTION_SHA), (KAGGLE_HUMAN, HUMAN_SHA),
                           ('TASK_DATA_AUDIT.json', DATA_AUDIT_SHA), ('TASK_DATA_CLOSEOUT.json', DATA_CLOSEOUT_SHA)]:
        if sha(root / 'inputs' / name) != expected:
            raise PermissionError('Direct human choice or actual complete task-data audit/release differs')
    evidence = read_data(base64.b64decode((root / 'inputs' / DATA_ARCHIVE).read_bytes(), validate=True))
    parent = root / 'inputs/PARENT_DATA_SOURCE_RELEASE.json'
    if sha(parent) != evidence.source_release_sha256 or read(parent)['commit'] != DATA_SOURCE:
        raise PermissionError('Task data source provenance differs')
    core = ['src/composite_task_skill_world.py', 'src/composite_task_skill_data.py',
            'src/composite_task_skill_worker.py', 'src/composite_record_programs.py',
            'src/composite_generalized_world.py', 'src/composite_generalized_tasks.py',
            'src/composite_generalized_interface.py', 'src/composite_generalized_novelty.py',
            'src/composite_design_v2.py', 'src/composite_design_v2_native.py',
            'src/composite_interface.py', 'src/native27_tools.py', 'config/composite_task_skill_v1.json']
    if any(sha(root / name) != read(parent)['files'][name] for name in core):
        raise PermissionError('Task/cohort/native semantics changed after the admitted data freeze')
    if sha(root / 'config/composite_task_skill_v1.json') != TASK_CONFIG_SHA:
        raise PermissionError('Original option-A preregistration changed')
    from .composite_v2_data_contract import ORIGINAL_MODEL_MANIFEST
    parent_model = root / 'inputs/PARENT_MODEL_INPUT_MANIFEST.json'
    if sha(parent_model) != ORIGINAL_MODEL_MANIFEST:
        raise PermissionError('Original full model file freeze changed')
    for name in ['QWEN_PIN.json', 'MODERNBERT_PIN.json']:
        if sha(root / 'inputs' / name) != read(parent_model)[name]:
            raise PermissionError('Original Qwen/ModernBERT file pins changed')
    cfg=read(root/'config/composite_controller_rank32_v2.json');validate_recipe(cfg)
    return cfg,evidence


def validate_model_source(root, commit,qualification_archive=None):
    """Verify phase-specific source/data and the actual raw parent dependency."""
    root=Path(root);stage=read(root/'inputs'/PLAN)
    from .composite_task_memory_profile import MODE as MEMORY_MODE
    if stage.get('version')==MEMORY_MODE:
        from .composite_task_memory_source import validate_source
        return validate_source(root,commit)
    required=data_input_names()
    if stage.get('phase')=='full_fit':
        from .composite_task_skill_parent import PARENT_FILES
        required|=PARENT_FILES
    cfg,evidence=validate_task_data_source(root,commit,required)
    parent_failure=read(root/'inputs'/PRE_FORWARD_PARENT)
    if (sha(root/'inputs'/PRE_FORWARD_PARENT)!=PRE_FORWARD_PARENT_SHA
            or parent_failure.get('scriptVersionId')!=361892579
            or parent_failure.get('model_forwards_started') is not False
            or parent_failure.get('forward_counts')!={'attempted':0,'completed':0}
            or parent_failure.get('optimizer_constructor_not_reached') is not True
            or parent_failure.get('owned_compute_released') is not True):
        raise PermissionError('Only the exact audited/released zero-forward engineering parent admits this repair')
    expected=qualification_plan(commit)
    if stage.get('phase')=='full_fit':
        from .composite_task_skill_parent import read_parent
        from .composite_task_skill_full_fit import full_fit_plan
        expected=full_fit_plan(commit,read_parent(root,qualification_archive))
    if stage != expected or read(root / 'inputs/WORKSPACE.json') != identity(stage['notebook']):
        raise PermissionError('Exact registered task-model phase/source required')
    return stage, cfg, evidence


def authorized_load_plan(root, commit, qwen):
    from .composite_runtime import require_scientific_runtime, platform
    import os
    require_scientific_runtime()
    if platform() != 'kaggle' or os.environ.get('CUDA_VISIBLE_DEVICES') != '0,1':
        raise PermissionError('Task fitting requires the selected Kaggle pair')
    actual = Path(os.environ['RESEARCH4_TASK_TRAINING_SOURCE_ROOT']).resolve()
    if not actual.is_relative_to(Path('/kaggle/temp/research4_task_training_inputs').resolve()) or actual.name != 'source' or actual != Path(root).resolve():
        raise PermissionError('Owned frozen task-training root required')
    stage, cfg, evidence = validate_model_source(actual, commit)
    if qwen != cfg['qwen'] or os.environ.get('RESEARCH4_TASK_TRAINING_PLAN_SHA256') != sha(actual / 'inputs' / PLAN):
        raise PermissionError('Actual task-training plan/configuration differs')
    from .composite_frozen_probe import verified_model_context
    context = verified_model_context(actual, commit, qwen, stage, sha(actual / 'inputs' / PLAN))
    context.update(stage=stage, full_config=cfg, task_data_evidence=evidence,
                   adapter_layout=adapter_layout(context['metadata'], context['plan']))
    return context


def selected_rows(actor, rows):
    context = getattr(actor, 'task_training_context', None)
    if context is None or not isinstance(context.get('task_data_evidence'), TaskDataEvidence):
        raise PermissionError('Task-only constructor/data authority required')
    expected = context['task_data_evidence'].rows()
    if digest(rows) != digest(expected):
        raise PermissionError('Fitting rows differ from the complete audited task data freeze')
    return validate_rows(rows, context['task_data_evidence'].release())


def fit_task_backend(actor, rows, placement, cfg, output, guard, progress, observer=None, qualification_rows=None):
    from .composite_behavior import fit_selected_rows
    rows = selected_rows(actor, rows)
    if cfg != actor.task_training_context['full_config']:
        raise PermissionError('Task fitting recipe differs from the admitted constructor')
    from .composite_task_memory_profile import MODE as MEMORY_MODE,stress_schedule
    stage=actor.task_training_context['stage']
    if stage['version']==MEMORY_MODE and stage['phase']=='qualification':
        expected=stress_schedule(token_plan(rows,actor.tokenizer,cfg))['state_ids']
        if qualification_rows is None or [r['state_id'] for r in qualification_rows]!=expected:
            raise PermissionError('Exact training-only long/short/long stress schedule required')
    return fit_selected_rows(actor, rows, placement, cfg, output, guard, progress, observer, qualification_rows)


def run_fit(actor, rows, cfg, plan, output, guard, progress, persist):
    context = getattr(actor, 'task_training_context', None)
    if context is None or context['stage'] != plan:
        raise PermissionError('Exact task-only model constructor and stage required')
    actual = authorized_load_plan(context['source_root'], context['source_commit'], actor.cfg)
    if actual['stage'] != plan:
        raise PermissionError('Actual source/runtime changed before task fitting')
    selected_rows(actor, rows)
    import torch
    from importlib.metadata import version
    hardware=postload_runtime(actor,torch.cuda,str(torch.__version__),{name:version(name) for name in
        ['transformers','bitsandbytes','accelerate','safetensors','numpy','psutil']})
    persist('TASK_TRAINING_HARDWARE.json', hardware)
    from .composite_v2_training import instrumented_fit
    from .composite_v2_raw_exports import TASK_PROFILE,STATEFUL_TASK_PROFILE
    from .composite_task_memory_profile import MODE as MEMORY_MODE,allocator_evidence
    stateful=plan['version']==MEMORY_MODE
    if stateful:persist('ALLOCATOR_BEFORE_INSTALL.json',allocator_evidence(torch.cuda))
    return instrumented_fit(actor, rows, cfg, plan, output, guard, progress, persist,
                            token_planner=token_plan, row_selector=lambda values:selected_rows(actor, values),
                            fitter=fit_task_backend, devices=(0,1), export_profile=STATEFUL_TASK_PROFILE if stateful else TASK_PROFILE)

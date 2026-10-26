"""Independent V4 data/source/native/tensor/AdamW/provider raw audit; no model calls."""
import base64,hashlib,json,math,re,tempfile
from copy import deepcopy
from pathlib import Path
from .io_utils import read,sha,digest
from .composite_bounded_behavior_fit_source import validate_source,notebook_code
from .composite_bounded_behavior_training import token_plan,adapter_layout
from .composite_bounded_behavior_interface import PROFILE
from . import composite_bounded_behavior_interface as native_interface
from .composite_bounded_behavior_fit_profile import MODE,MAPPING,MANIFEST
from .composite_v2_model_audit import (EvidenceFiles,checkpoint_evidence,audit_optimizer_bytes,
    frozen_integrity,audit_runtime,audit_bypass,tensor_state)
from .composite_task_skill_model_audit import verify_fresh_initialization
_KEY=object()
OWNER_THREAD='01a175f0-fc99-76b1-8bb2-d64f22afbcd7'

class BoundedFitEvidence:
    def __init__(self,value,key=None):
        if key is not _KEY:raise PermissionError('Read the complete raw V4 fit evidence')
        self._value=deepcopy(value);self._key=key
    def receipt(self):return deepcopy(self._value)

class KaggleClosure:
    def __init__(self,value,key=None):
        if key is not _KEY:raise PermissionError('Reconcile exact external provider evidence')
        self._value=deepcopy(value);self._key=key

def require_qualification(value):
    if not isinstance(value,BoundedFitEvidence) or value._key is not _KEY:
        raise PermissionError('Genuine complete raw V4 qualification required')
    proof=value.receipt()
    if (proof.get('phase')!='qualification' or proof.get('runtime_profile')!=MODE
        or proof.get('complete_training_qualification_audited') is not True
        or proof.get('scientific_qualification_admitted') is not True
        or proof.get('owned_compute_released') is not True
        or proof.get('stateful_AdamW_overlap_and_size_switch_verified') is not True
        or proof.get('qualified_optimizer_updates')!=3
        or proof.get('optimizer',{}).get('numeric_AdamW_updates_independently_recomputed')!=3
        or proof.get('optimizer',{}).get('optimizer_step')!=3
        or proof.get('optimizer',{}).get('initial_A_tensors_verified')!=496
        or proof.get('checkpoint',{}).get('adapter_parameters')!=233455616
        or proof.get('checkpoint',{}).get('tensors')!=992
        or proof.get('initialization_mapping_version')!=MAPPING
        or proof.get('native_interface_profile')!=PROFILE
        or proof.get('checkpoint_initializes_full_fit') is not False
        or proof.get('missing_or_unverified')!=[]):
        raise PermissionError('New exact-data three-update raw stateful qualification and release required')
    return proof

def require_full_fit(value):
    if not isinstance(value,BoundedFitEvidence) or value._key is not _KEY:
        raise PermissionError('Genuine complete raw V4 full fit required')
    proof=value.receipt()
    if (proof.get('phase')!='full_fit' or proof.get('full_fit_completed') is not True
        or proof.get('full_fit_raw_audited') is not True or proof.get('owned_compute_released') is not True
        or proof.get('actual_full_rank_backward_verified') is not True
        or proof.get('source_model_input_and_tokenizer_bindings_verified') is not True
        or proof.get('checkpoint',{}).get('adapter_parameters')!=233455616
        or proof.get('checkpoint',{}).get('tensors')!=992
        or proof.get('optimizer',{}).get('optimizer_step')!=64
        or proof.get('optimizer',{}).get('full64_optimizer_updates_recorded') is not True
        or proof.get('fresh_fit_adapter_initialization_verified') is not True
        or proof.get('sampler_matches_registered_seed') is not True
        or proof.get('native_interface_profile')!=PROFILE
        or proof.get('missing_or_unverified')!=[] or proof.get('heldout_content_read') is not False):
        raise PermissionError('Every V4 full-fit tensor/source/data/sampler/release dependency must pass')
    return proof

def kaggle_closure(binding,archive_sha256,terminal_AX,release_AX,controller,actual_code_cells,prepared_code_cells):
    run=binding.get('scriptVersionId');source=binding.get('source_commit');phase=binding.get('phase')
    notebook={'qualification':'farhad89/research4-bounded-behavior-qualification-v4',
      'full_fit':'farhad89/research4-bounded-behavior-fit-v4'}.get(phase)
    if (type(run) is not int or run<=0 or notebook is None or binding.get('notebook')!=notebook
        or binding.get('mode')!=MODE or binding.get('owner_thread_id')!=OWNER_THREAD
        or not re.fullmatch('[0-9a-f]{40}',source or '') or not re.fullmatch('[0-9a-f]{64}',archive_sha256 or '')):
        raise PermissionError('Exact consumed V4 stage/submission/source required')
    if (notebook not in terminal_AX or 'scriptVersionId='+str(run) not in terminal_AX
        or not re.search(r'successfully ran after|second run - successful',terminal_AX)
        or not re.search(re.escape('https://www.kaggle.com/code/'+notebook)
          +r"(?=$|[/?#\s\"'<>])",release_AX)
        or '0  Active Events' not in release_AX or 'Draft Session off' not in release_AX):
        raise PermissionError('Actual exact provider terminal and release observations required')
    child=controller.get('child',{})
    if (controller.get('source_commit')!=source or controller.get('notebook')!=notebook
        or controller.get('mode')!=MODE or controller.get('phase')!=phase
        or controller.get('owner_thread_id')!=OWNER_THREAD
        or controller.get('status')!='worker_exited' or type(controller.get('exit_code')) is not int
        or controller['exit_code']!=0 or controller.get('timed_out') is not False
        or child.get('child_started') is not True or child.get('child_exit_confirmed') is not True
        or type(child.get('exit_code')) is not int or child['exit_code']!=0
        or controller.get('raw_prefix_rescue',{}).get('status')!='owned_scratch_absent'):
        raise PermissionError('Actual controlled child exit and owned cleanup required')
    if not actual_code_cells or digest(actual_code_cells)!=digest(prepared_code_cells):
        raise PermissionError('Actual saved V4 notebook differs from reviewed prepared source')
    return KaggleClosure({'scriptVersionId':run,'source_commit':source,'notebook':notebook,
      'archive_sha256':archive_sha256,'owned_compute_released':True,
      'terminal_AX_sha256':hashlib.sha256(terminal_AX.encode()).hexdigest(),
      'release_AX_sha256':hashlib.sha256(release_AX.encode()).hexdigest(),
      'controller_sha256':digest(controller),'actual_code_cells_sha256':digest(actual_code_cells)},_KEY)


def source_closure_matches(closure,root,source_commit,raw_sha256,stage):
    """Only independently source-bound saved notebook code can establish release."""
    if not isinstance(closure,KaggleClosure) or closure._key is not _KEY:return False
    value=closure._value
    if (value['source_commit']!=source_commit or value['archive_sha256']!=raw_sha256
            or value['notebook']!=stage['notebook']):
        raise PermissionError('Actual provider closure belongs to another source/raw/stage')
    expected=notebook_code(root,source_commit,sha(Path(root).parent/MANIFEST))
    if value['actual_code_cells_sha256']!=digest([expected]):
        raise PermissionError('Actual saved V4 fitting notebook differs from the reviewed source')
    return True

def memory_records(value):
    if (not isinstance(value, list) or len(value) != 2 or any(type(r.get('device')) is not int for r in value)
            or {r.get('device') for r in value} != {0, 1}):
        raise ValueError('Both per-device memory records required')
    for row in value:
        keys = ['free_bytes', 'total_bytes', 'allocated_bytes', 'reserved_bytes', 'peak_allocated_bytes', 'peak_reserved_bytes']
        if any(type(row.get(k)) is not int or row[k] < 0 for k in keys):
            raise ValueError('Raw nonnegative memory counters required')
        if (not 0 <= row['allocated_bytes'] <= row['reserved_bytes'] <= row['peak_reserved_bytes'] <= row['total_bytes']
                or not row['allocated_bytes'] <= row['peak_allocated_bytes'] <= row['peak_reserved_bytes']
                or not row['free_bytes'] <= row['total_bytes']
                or row['free_bytes'] + row['reserved_bytes'] - row['allocated_bytes'] < 512 * 1024**2):
            raise ValueError('Per-device live/idle allocation bounds differ')


def audit_clipped_gradients(gradients, shapes):
    """Check every actual final gradient, including after a multi-update fit."""
    import torch
    if set(gradients) != set(shapes):
        raise ValueError('Full final gradient tensor roster differs')
    squared_norm = 0.0
    for name, shape in shapes.items():
        grad = gradients[name]
        if not isinstance(grad, torch.Tensor) or grad.dtype != torch.float32 or list(grad.shape) != shape or not torch.isfinite(grad).all():
            raise ValueError('Full finite clipped FP32 gradient required')
        squared_norm += float(grad.double().square().sum())
    if squared_norm > 1.00002:
        raise ValueError('Exported clipped gradient exceeds the registered unit norm')
    return {'full_clipped_gradient_tensors_verified':len(gradients),
            'clipped_global_norm':math.sqrt(squared_norm)}


def audit_optimizer_config(files):
    expected = {'lr':0.0005, 'betas':[0.9,0.999], 'eps':1e-8, 'weight_decay':0.0,
                'foreach':False, 'maximize':False, 'capturable':False, 'differentiable':False}
    if digest(files.json('OPTIMIZER_CONFIG.json')) != digest(expected):
        raise ValueError('Actual AdamW group configuration differs')


def audit_scratch_update(files, shapes, initial_hashes):
    """Recompute the one scratch AdamW update from full clipped gradients/moments."""
    import io
    import torch
    from .composite_training import tensor_digest
    audit_optimizer_config(files)
    member=lambda name:files.tensor_path(name) if hasattr(files,'tensor_path') else files.get(name)
    gradients = tensor_state(member('GRADIENT_FINAL.pt'))
    checkpoint = tensor_state(member('behavior.pt'))['weights']
    if set(gradients) != set(shapes) or set(checkpoint) != set(shapes):
        raise ValueError('Full final gradient/checkpoint tensor roster differs')
    gradient_proof = audit_clipped_gradients(gradients, shapes)
    for name, shape in shapes.items():
        grad = gradients[name]
        if not isinstance(grad, torch.Tensor) or grad.dtype != torch.float32 or list(grad.shape) != shape or not torch.isfinite(grad).all():
            raise ValueError('Full finite clipped FP32 gradient required')
        if name.endswith('.A'):
            if torch.count_nonzero(grad) != 0 or tensor_digest(checkpoint[name]) != initial_hashes[name]:
                raise ValueError('Zero-B scratch initialization must leave A unchanged on update1')
        else:
            zero = torch.zeros(shape, dtype=torch.float32)
            if tensor_digest(zero) != initial_hashes[name]:
                raise ValueError('Qualification B did not start from the fresh zero initialization')
    def inspect(name, value):
        grad = gradients[name]
        torch.testing.assert_close(value['exp_avg'], grad * 0.1, rtol=1e-5, atol=1e-12)
        torch.testing.assert_close(value['exp_avg_sq'], grad.square() * 0.001, rtol=1e-5, atol=1e-20)
        if name.endswith('.B'):
            denominator = value['exp_avg_sq'].sqrt() / math.sqrt(0.001) + 1e-8
            predicted = -0.005 * value['exp_avg'] / denominator
            torch.testing.assert_close(checkpoint[name], predicted, rtol=1e-5, atol=1e-8)
    proof = audit_optimizer_bytes(member('OPTIMIZER_FINAL.pt'), shapes, 1, inspect=inspect)
    return {**proof, **gradient_proof,
            'scratch_AdamW_update_independently_recomputed':True,
            'FP32_update_tolerance':{'rtol':1e-5,'atol':1e-8}}


def audit_fit(raw, expected_sha256, root, source_commit, closure=None,*,required_phase=None,qualification_archive=None,data_archive=None,alias_fit_archive=None,alias_qualification_archive=None):
    stage, cfg, data = validate_source(root,source_commit,qualification_archive,data_archive=data_archive,alias_fit_archive=alias_fit_archive,alias_qualification_archive=alias_qualification_archive)
    phase=stage['phase'];updates={'qualification':3,'full_fit':64}[phase]
    from .composite_bounded_behavior_fit_profile import MODE,RAW_PROFILE,stress_schedule,verify_allocator_record
    memory_mode=stage['version']==MODE
    stateful=memory_mode and phase=='qualification'
    if stateful:updates=3
    if required_phase is not None and phase!=required_phase:raise PermissionError('Another model phase was supplied')
    released=source_closure_matches(closure,root,source_commit,expected_sha256,stage)
    files = EvidenceFiles(raw, expected_sha256, stage['maximum_raw_evidence_bytes'],
                          evidence_profile=RAW_PROFILE if stateful else None)
    try:
        receipt = files.json('RUN_RECEIPT.json')
        base = {'reader': 'bounded-behavior-two-t4-fit-raw-v4', 'source_commit': source_commit,
                'phase': phase, 'raw_archive_sha256': expected_sha256,
                'source_data_archive_sha256': stage['source_data_archive_sha256'],
                'data_audit_canonical_sha256':stage['data_audit_canonical_sha256'],
                'fitting_rows_canonical_sha256':stage['fitting_rows_canonical_sha256'],
                'native_interface_profile':PROFILE,
                'complete_training_qualification_audited': False, 'scientific_qualification_admitted': False,
                'full_fit_completed': False, 'checkpoint_initializes_full_fit': False,
                'owned_compute_released': None, 'recorded_forward_counts': receipt.get('forward_counts'),
                'new_model_forwards': 0, 'new_optimizer_updates': 0, 'heldout_content_read': False,
                'absence_of_unlogged_calls_proved': False}
        if (receipt.get('source_commit') != source_commit or receipt.get('mode') != stage['version']
                or receipt.get('phase') != phase or receipt.get('heldout_content_read') is not False):
            raise PermissionError('Raw qualification source/scope differs')
        if receipt.get('status') != 'complete' or receipt.get('public_fixture_test_mode') is not False:
            base.update(worker_status=receipt.get('status'), missing_or_unverified=['worker_not_complete_or_public_fixture'],
                        optimizer_steps_completed_lower_bound=receipt.get('optimizer_steps_completed_lower_bound'))
            return BoundedFitEvidence(base, _KEY)
        if (type(receipt.get('model_instances')) is not int or receipt['model_instances'] != 1
                or type(receipt.get('optimizer_updates')) is not int or receipt['optimizer_updates'] != updates
                or receipt.get('full_fit_completed') is not (phase=='full_fit') or receipt.get('counts_exact_at_controlled_exit') is not True):
            raise ValueError('One complete scratch-model stage with exact controlled-exit counts required')
        required = {'TRAINING_TOKEN_PLAN.json', 'BEHAVIOR_TRAINING.json', 'TRAINING_TERMINAL_JOURNAL.json',
            'behavior.pt', 'OPTIMIZER_FINAL.pt', 'OPTIMIZER_EXPORT.json', 'ADAPTER_INITIAL_HASHES.json',
            'GRADIENT_FINAL.pt', 'GRADIENT_EXPORT.json', 'OPTIMIZER_CONFIG.json', 'HARDWARE_PREFLIGHT.json',
            'ADAPTER_CAPACITY_RUNTIME.json', 'BACKBONE_LIVE_BEFORE.json', 'BACKBONE_LIVE_AFTER.json',
            'RUNTIME_RAW.json', 'MODEL_LOAD_ADMISSION.json', 'MODEL_FILE_VERIFICATION.json',
            'MODEL_METADATA_BYTES.json', 'BYPASS_RAW_MANIFEST.json', 'BYPASS_CHECK.json', 'PRECALL_FREEZE.json'}
        if stateful:
            required-={'GRADIENT_FINAL.pt','GRADIENT_EXPORT.json'}
            required|={'INITIAL_A.pt','INITIAL_A_EXPORT.json','OPTIMIZER_INITIAL_STATE.json',
                       'STATEFUL_TRAINING_SCHEDULE.json','ALLOCATOR_BEFORE_INSTALL.json','ALLOCATOR_AFTER_INSTALL.json'}
            for i in [1,2,3]:required|={'GRADIENT_STEP_'+str(i).zfill(3)+suffix for suffix in ['.pt','_EXPORT.json']}
        elif memory_mode:required|={'OPTIMIZER_INITIAL_STATE.json','ALLOCATOR_BEFORE_INSTALL.json','ALLOCATOR_AFTER_INSTALL.json'}
        missing = sorted(required - set(files.manifest))
        if missing:
            return BoundedFitEvidence({**base, 'missing_or_unverified': missing}, _KEY)
        freeze = files.json('PRECALL_FREEZE.json')
        if (digest(freeze['stage']) != digest(stage) or freeze['source_commit'] != source_commit
                or freeze['input_manifest_sha256'] != sha(Path(root) / 'inputs/MANIFEST.json')
                or digest(freeze['source_release']) != digest(read(Path(root) / 'RELEASE.json'))
                or freeze['raw_training_rows_canonical_sha256'] != digest(data.fitting_rows())):
            raise ValueError('Precall source/input/data freeze differs')
        from .composite_frozen_probe import (verified_model_context, validate_download_proof,
            verify_metadata_bytes, text_projection, validate_loading_info)
        context = verified_model_context(root, source_commit, cfg['qwen'], stage, sha(Path(root) / 'inputs/BOUNDED_BEHAVIOR_FIT_PLAN.json'))
        metadata = context['metadata']
        originals = {n: v['shape'] for n, v in metadata['tensors'].items() if not n.startswith('mtp.')}
        from .composite_placement import parameter_device
        devices = {n: 'cuda:' + str(parameter_device(n, context['plan'])) for n in originals}
        projections = {n.removeprefix('model.language_model.').removesuffix('.weight'): [v[1], v[0]]
                       for n, v in originals.items() if text_projection(n)}
        layout = adapter_layout(metadata, context['plan'])
        hardware = files.json('HARDWARE_PREFLIGHT.json')
        if len(hardware['devices']) != 2 or {r.get('index') for r in hardware['devices']} != {0, 1}:
            raise ValueError('Actual T4-pair hardware evidence required')
        for row in hardware['devices']:
            lower = context['metadata_bounds']['device_weight_lower_bounds_bytes'][row['index']]
            if (row.get('name') != 'Tesla T4' or row['optimistic_weight_lower_bound_bytes'] != lower
                    or row['free_bytes'] < lower + 2 * 1024**3):
                raise ValueError('Pinned per-device hardware lower-bound admission differs')
        capacity = files.json('ADAPTER_CAPACITY_RUNTIME.json')
        if (capacity['adapter_parameters'] != 233455616 or capacity['projection_count'] != 496
                or capacity['covered_blocks'] != list(range(64)) or digest(capacity['projections']) != digest(projections)):
            raise ValueError('Full actual adapter coverage differs')
        storage = capacity['per_device_storage_admission']['devices']
        if len(storage) != 2 or {r['device'] for r in storage} != {0, 1}:
            raise ValueError('Separate actual adapter/optimizer device budgets required')
        for row in storage:
            expected_storage = layout['FP32_weights_gradients_AdamW_bytes'][row['device']]
            if (row['adapter_gradient_AdamW_storage_bytes'] != expected_storage
                    or row['activation_reserve_bytes'] != 2 * 1024**3
                    or row['free_bytes'] < expected_storage + row['activation_reserve_bytes']):
                raise ValueError('Actual per-device full-state storage/reserve differs')
        pin = read(Path(root) / 'inputs/QWEN_PIN.json')
        full, encoded = files.json('MODEL_FILE_VERIFICATION.json'), files.json('MODEL_METADATA_BYTES.json')
        validate_download_proof({'model_pin': pin}, full)
        verify_metadata_bytes(pin, full, encoded)
        loading = files.json('MODEL_LOAD_ADMISSION.json')['loading']
        validate_loading_info({'missing_keys': loading['missing_keys'], 'mismatched_keys': loading['mismatched_keys'],
                              'error_msgs': loading['error_msgs'], 'unexpected_keys': loading['ignored_checkpoint_only_MTP_keys']})
        with tempfile.TemporaryDirectory(prefix='r4-task-model-audit-tokenizer-') as folder:
            target = Path(folder)
            for name, value in encoded.items():
                path = (target / name).resolve()
                if not path.is_relative_to(target.resolve()):
                    raise PermissionError('Metadata escaped the audit scratch')
                path.write_bytes(base64.b64decode(value, validate=True))
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(target, local_files_only=True, trust_remote_code=False)
            tokens = token_plan(data.fitting_rows(), tokenizer, cfg)
            if digest(tokens) != digest(files.json('TRAINING_TOKEN_PLAN.json')):
                raise ValueError('Full task-only token replay differs')
            training = files.json('BEHAVIOR_TRAINING.json')
            ids = tokens['qualification_state_ids']
            if stateful:
                stress=stress_schedule(tokens);ids=stress['state_ids']
                if digest(stress)!=digest(files.json('STATEFUL_TRAINING_SCHEDULE.json')):
                    raise ValueError('Registered training-only memory stress schedule differs')
            if memory_mode:
                if digest(files.json('OPTIMIZER_INITIAL_STATE.json'))!=digest({'state_entries':0,'parameter_tensors':992}):
                    raise ValueError('Qualification optimizer must begin fresh with empty state')
                for name in ['ALLOCATOR_BEFORE_INSTALL.json','ALLOCATOR_AFTER_INSTALL.json']:
                    verify_allocator_record(files.json(name))
            if phase=='full_fit':
                import numpy as np
                rng=np.random.default_rng(202610212)
                ids=[tokens['records'][int(rng.integers(len(tokens['records'])))]['state_id'] for _ in range(128)]
                draw_plan=data.fitting_draw_plan()
                if draw_plan['coverage_passed'] is not True or draw_plan['state_ids']!=ids:
                    raise PermissionError('Actual128draws differ from the independently audited V4 coverage plan')
            if (type(training['optimizer_updates']) is not int or training['optimizer_updates'] != updates
                    or training['sampled_state_ids'] != ids or len(training['losses']) != updates*2
                    or any(type(v) not in [int,float] or not math.isfinite(v) for v in training['losses'])):
                raise ValueError('Exact scratch update/longest-example selection differs')
            gradient_names = {n + '.' + part for n in projections for part in ['A', 'B']}
            for index, state in enumerate(ids):
                row = files.json('training/backward_' + str(index + 1).zfill(4) + '.json')
                if [row['update'], row['microstep'], row['state_id'], row['loss']] != [index//2+1, index%2, state, training['losses'][index]]:
                    raise ValueError('Backward journal differs')
                gradients = row['gradients']
                if len(gradients) != 992 or {r['name'] for r in gradients} != gradient_names:
                    raise ValueError('All992 gradient tensors required')
                for grad in gradients:
                    projection = grad['name'].rsplit('.', 1)[0]
                    device = 'cuda:' + str(layout['projections'][projection]['device'])
                    if grad['device'] != device or grad['dtype'] != 'torch.float32' or not math.isfinite(grad['norm']) or grad['norm'] < 0:
                        raise ValueError('Per-device full-rank gradient evidence differs')
                memory_records(row['memory'])
                if memory_mode:
                    before=files.json('training/resident_optimizer_before_'+str(index+1).zfill(4)+'.json')
                    update=index//2+1
                    expected={'update':update,'microstep':index%2,'state_id':state,
                      'state_entries':0 if update==1 else 992,
                      'resident_moment_bytes_by_device':[0,0] if update==1 else [n*8 for n in layout['device_parameters']],
                      'previous_optimizer_step':update-1,'backward_not_yet_executed_for_this_microstep':True,
                      'full_state_resident':update>1}
                    if digest(before)!=digest(expected):raise ValueError('Full persistent AdamW state was not resident for later backward')
                    verify_allocator_record(files.json('training/allocator_after_backward_'+str(index+1).zfill(4)+'.json'))
            journal = files.json('TRAINING_TERMINAL_JOURNAL.json')
            for step in range(1,updates+1):
                update=files.json('training/update_'+str(step).zfill(3)+'.json');memory_records(update['memory'])
                if any(type(update.get(k)) is not int or update[k]!=v for k,v in [('optimizer_steps_attempted',step),('optimizer_steps_completed',step),('backward_microsteps_completed',step*2)]):
                    raise ValueError('An optimizer update journal is missing/incomplete')
            for item in [journal, update]:
                if (any(type(item.get(k)) is not int or item[k] != v for k,v in [('optimizer_steps_attempted',updates),('optimizer_steps_completed',updates),('backward_microsteps_completed',updates*2)])
                        or item.get('in_backward') is not False):
                    raise ValueError('Complete scratch update/accumulation not proven')
            if any(len(journal[k]) != 64 or any(type(v) is not int or v < updates*2 for v in journal[k])
                   for k in ['layer_forward_invocations', 'backward_recomputation_invocations']):
                raise ValueError('All64 blocks need actual backward recomputation evidence')
            checkpoint = checkpoint_evidence(files.tensor_path('behavior.pt'), training['checkpoint_sha256'], cfg['qwen'],
                                              projections, files.json('ADAPTER_INITIAL_HASHES.json'),native_profile=PROFILE)
            shapes = {n + '.A': [32, v[0]] for n, v in projections.items()}
            shapes.update({n + '.B': [v[1], 32] for n, v in projections.items()})
            descriptors=[('OPTIMIZER_FINAL.pt','OPTIMIZER_EXPORT.json')]
            if stateful:
                descriptors+=[('INITIAL_A.pt','INITIAL_A_EXPORT.json')]
                descriptors+=[('GRADIENT_STEP_'+str(i).zfill(3)+'.pt','GRADIENT_STEP_'+str(i).zfill(3)+'_EXPORT.json') for i in [1,2,3]]
            else:descriptors+=[('GRADIENT_FINAL.pt','GRADIENT_EXPORT.json')]
            for name, descriptor in descriptors:
                proof = files.json(descriptor)
                if proof['path'] != name or proof['sha256'] != files.manifest[name] or proof['bytes'] != files.stream.getinfo(name).file_size:
                    raise ValueError('Raw tensor export/readback descriptor differs')
            if stateful:
                from .composite_task_memory_numeric import audit_three_updates
                optimizer=audit_three_updates(files,shapes,files.json('ADAPTER_INITIAL_HASHES.json'))
            elif phase=='qualification':optimizer = audit_scratch_update(files, shapes, files.json('ADAPTER_INITIAL_HASHES.json'))
            else:
                from .composite_bounded_behavior_fit_parent import read_qualification_parent
                parent=require_qualification(read_qualification_parent(root,qualification_archive,data_archive=data_archive,alias_fit=alias_fit_archive,alias_qualification=alias_qualification_archive))
                verify_fresh_initialization(files.json('ADAPTER_INITIAL_HASHES.json'),parent.get('initial_adapter_hashes'))
                audit_optimizer_config(files)
                optimizer=audit_optimizer_bytes(files.tensor_path('OPTIMIZER_FINAL.pt'),shapes,64)
                optimizer.update(audit_clipped_gradients(tensor_state(files.tensor_path('GRADIENT_FINAL.pt')),shapes))
                optimizer.update(full64_optimizer_updates_recorded=True,complete_optimizer_reexecution_proved=False)
            integrity = frozen_integrity(files.json('BACKBONE_LIVE_BEFORE.json'), files.json('BACKBONE_LIVE_AFTER.json'),
                                         originals, parameter_devices=devices)
            audit_runtime(files.json('RUNTIME_RAW.json'), files.json('BACKBONE_LIVE_AFTER.json'), originals,
                          expected_map=context['plan']['device_map'], parameter_devices=devices,
                          require_native_bf16=False)
            generation = read(target / 'generation_config.json')
            eos = set()
            for value in [generation.get('eos_token_id'), tokenizer.eos_token_id]:
                if value is not None:
                    eos.update(value if isinstance(value, list) else [value])
            rows = {r['state_id']: r for r in data.fitting_rows()}
            bypass_context = {'vocabulary': originals['lm_head.weight'][0], 'actor_tokenizer': tokenizer,
                'eos': eos, 'native_interface':native_interface,'qualification_texts': [rows[k]['text'] for k in tokens['qualification_state_ids']]}
            bypass = audit_bypass(files, tokens, bypass_context)
            first = files.json('BYPASS_CHECK.json')['records'][0]
            expected_forwards = updates*2 + 6 + sum(len(first[k]['generated_token_ids']) for k in ['wrapped_generation', 'bare_generation'])
            if receipt['forward_counts'] != {'attempted': expected_forwards, 'completed': expected_forwards}:
                raise ValueError('Exact recorded logical forward count differs')
        base.update(checkpoint=checkpoint, checkpoint_sha256=training['checkpoint_sha256'],
                    initial_adapter_hashes=files.json('ADAPTER_INITIAL_HASHES.json'),
                    fresh_fit_adapter_initialization_verified=phase=='full_fit',
                    sampler_matches_registered_seed=phase=='full_fit',
                    optimizer=optimizer, backbone_integrity=integrity, bypass=bypass,
                    actual_full_rank_backward_verified=True, source_model_input_and_tokenizer_bindings_verified=True,
                    owned_compute_released=True if released else None,
                    complete_training_qualification_audited=released and phase=='qualification', scientific_qualification_admitted=released and phase=='qualification',
                    full_fit_completed=released and phase=='full_fit',full_fit_raw_audited=released and phase=='full_fit',
                    provider_run_id=closure._value['scriptVersionId'] if released else None,
                    missing_or_unverified=[] if released else ['external_provider_terminal_release_not_verified'])
        if stateful:base.update(runtime_profile=MODE,stateful_AdamW_overlap_and_size_switch_verified=True,
          initialization_mapping_version=stage['initialization_mapping_version'],qualified_optimizer_updates=3,
          full64_training_capacity_established=False,arbitrary6144_training_capacity_established=False)
        elif stage['version']==MODE:
            base.update(runtime_profile=MODE,initialization_mapping_version=stage['initialization_mapping_version'],
              stateful_parent_qualification_verified=True,qualification_parent_run_id=stage['qualification_run_id'])
        return BoundedFitEvidence(base, _KEY)
    finally:
        files.close()

"""Independent read-only source/model/native audit of one fixed task partition.

No model construction, forward, optimizer, teacher or provider API is used.
Passing this reader is coverage/integrity evidence, not a performance finding
or admission of the full task72/risk180/held-out campaign.
"""
import base64
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import tempfile
from .io_utils import read,sha,digest
from .composite_task_skill_inference import validate_source,STAGE
from .composite_task_skill_package import require_frozen_task_package
from .composite_task_skill_trace_audit import replay_pair
from .composite_v2_model_audit import EvidenceFiles,frozen_integrity,audit_runtime

_KEY=object()


class TaskDevelopmentAudit:
    def __init__(self,value,key=None):
        if key is not _KEY:raise PermissionError('Read the complete original task-pair evidence')
        self._key=key;self._value=deepcopy(value)
    def receipt(self):return deepcopy(self._value)


def require_task_partition(value):
    if not isinstance(value,TaskDevelopmentAudit) or value._key is not _KEY:
        raise PermissionError('Genuine source/model/native/provider raw audit required')
    result=value.receipt()
    if (result.get('task_partition_admitted') is not True or result.get('raw_evidence_audited') is not True
            or result.get('owned_compute_released') is not True or result.get('missing_or_unverified')!=[]
            or type(result.get('expected_trajectories')) is not int or result['expected_trajectories'] not in [2,4,6]
            or result.get('completed_trajectories')!=result['expected_trajectories'] or result.get('optimizer_updates')!=0
            or result.get('heldout_content_read') is not False):
        raise PermissionError('Complete audited and released task-pair dependency required')
    return result


@contextmanager
def _tokenizer(files,context):
    from .composite_frozen_probe import validate_download_proof,verify_metadata_bytes
    full=files.json('MODEL_FILE_VERIFICATION.json');encoded=files.json('MODEL_METADATA_BYTES.json')
    validate_download_proof(context,full);verify_metadata_bytes(context['model_pin'],full,encoded)
    with tempfile.TemporaryDirectory(prefix='r4-task-pair-tokenizer-') as folder:
        root=Path(folder).resolve()
        for name,value in encoded.items():
            target=(root/name).resolve()
            if not target.is_relative_to(root):raise PermissionError('Only original small tokenizer metadata is reconstructed')
            target.write_bytes(base64.b64decode(value,validate=True))
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(root,local_files_only=True,trust_remote_code=False)
        eos=set()
        for value in [read(root/'generation_config.json').get('eos_token_id'),tokenizer.eos_token_id]:
            if value is not None:eos.update(value if isinstance(value,list) else [value])
        vocabulary=context['metadata']['tensors']['lm_head.weight']['shape'][0]
        if not eos or any(type(v) is not int or not 0<=v<vocabulary for v in eos):raise ValueError('Original native EOS/vocabulary required')
        yield tokenizer,eos,vocabulary


def _checkpoint(value,model,layout):
    if (value.get('checkpoint_sha256')!=model['checkpoint_sha256']
            or value.get('live_tensor_hashes')!=model['checkpoint_tensor_hashes']
            or value.get('adapter_parameters')!=233455616 or value.get('live_recomputed') is not True
            or value.get('all_adapter_requires_grad') is not False or value.get('all_adapter_gradients_none') is not True):
        raise ValueError('Live full fitted checkpoint integrity differs')
    geometry=value['tensor_geometry']
    if set(geometry)!=set(model['checkpoint_tensor_shapes']):raise ValueError('All992 live adapter geometry records required')
    for name,shape in model['checkpoint_tensor_shapes'].items():
        expected={'shape':shape,'dtype':'torch.float32',
          'device':'cuda:'+str(layout['projections'][name.rsplit('.',1)[0]]['device']),
          'requires_grad':False,'gradient_is_none':True}
        if digest(geometry[name])!=digest(expected):raise ValueError('Live adapter shape/precision/device/gradient state differs')


def audit_collection(raw,raw_sha256,root,commit,fit_archive,qualification_archive,closure=None):
    context=validate_source(root,commit,fit_archive,qualification_archive)
    stage=context.stage();model=require_frozen_task_package(context.model)
    result={'reader':'ordinary-task-pair-source-native-model-raw-v1','source_commit':commit,
      'raw_archive_sha256':raw_sha256,'frozen_model_package_sha256':digest(model),
      'fit_run_id':model['fit_run_id'],'checkpoint_sha256':model['checkpoint_sha256'],
      'partition_index':stage['partition']['index'],'case_ids':stage['partition']['case_ids'],
      'expected_trajectories':stage['partition'].get('expected_trajectories'),
      'task_partition_admitted':False,'raw_evidence_audited':False,'owned_compute_released':None,
      'completed_trajectories':None,'recorded_forward_counts':None,'recorded_optimizer_updates':None,
      'new_model_forwards':0,'new_optimizer_updates':0,'scientific_validation_admitted':False,
      'full_task72_complete':False,'risk180_complete':False,'heldout_content_read':False,'missing_or_unverified':[]}
    files=EvidenceFiles(raw,raw_sha256,stage['maximum_raw_evidence_bytes'])
    try:
        if any(n.endswith('.pt') for n in files.manifest):
            raise PermissionError('This frozen task collection exports metadata/native records, not fitting tensors')
        receipt=files.json('RUN_RECEIPT.json')
        if (receipt.get('status')!='complete' or receipt.get('mode')!=stage['version']
                or receipt.get('source_commit')!=commit or receipt.get('phase')!='task_development'
                or receipt.get('public_fixture_test_mode') is not False
                or receipt.get('evaluation_only') is not True or receipt.get('training_objective') is not None
                or receipt.get('full_fit_completed') is not False or type(receipt.get('optimizer_updates')) is not int
                or receipt['optimizer_updates']!=0
                or receipt.get('counts_exact_at_controlled_exit') is not True
                or receipt.get('owned_download_cache_removed') is not True or receipt.get('heldout_content_read') is not False):
            result['missing_or_unverified']=['complete_controlled_frozen_evaluation_exit'];return TaskDevelopmentAudit(result,_KEY)
        required={'PRECALL_FREEZE.json','HARDWARE_PREFLIGHT.json','MODEL_FILE_VERIFICATION.json','MODEL_METADATA_BYTES.json',
          'MODEL_LOAD_ADMISSION.json','BACKBONE_LIVE_BEFORE.json','BACKBONE_LIVE_AFTER.json','CHECKPOINT_LIVE_BEFORE.json',
          'CHECKPOINT_LIVE_AFTER.json','RUNTIME_RAW.json','ALLOCATOR_BEFORE_COLLECTION.json','ALLOCATOR_AFTER_COLLECTION.json',
          'TASK_COLLECTION_RESULT.json','POSTLOAD_RUNTIME.json','CHECKPOINT_RESTORE.json'}
        missing=sorted(required-set(files.manifest))
        if missing:result['missing_or_unverified']=missing;return TaskDevelopmentAudit(result,_KEY)
        freeze=files.json('PRECALL_FREEZE.json')
        expected={'stage':stage,'source_commit':commit,'data_archive_sha256':model['data_archive_sha256'],
          'input_manifest_sha256':sha(Path(root)/'inputs/MANIFEST.json'),'raw_training_rows_canonical_sha256':None,
          'source_release':read(Path(root)/'RELEASE.json')}
        if digest(freeze)!=digest(expected):raise ValueError('Exact pre-call source/model/data/stage freeze differs')
        cfg=read(Path(root)/'config/composite_controller_rank32_v2.json')
        from .composite_frozen_probe import verified_model_context,validate_loading_info
        load=verified_model_context(root,commit,cfg['qwen'],stage,sha(Path(root)/'inputs'/STAGE))
        from .composite_task_skill_training import adapter_layout
        layout=adapter_layout(load['metadata'],load['plan'])
        originals={n:v['shape'] for n,v in load['metadata']['tensors'].items() if not n.startswith('mtp.')}
        from .composite_placement import parameter_device
        devices={n:'cuda:'+str(parameter_device(n,load['plan'])) for n in originals}
        hardware=files.json('HARDWARE_PREFLIGHT.json')['devices']
        if (len(hardware)!=2 or any(type(r.get('index')) is not int for r in hardware)
                or {r['index'] for r in hardware}!={0,1}):raise ValueError('Actual two-device hardware evidence required')
        for row in hardware:
            lower=load['metadata_bounds']['device_weight_lower_bounds_bytes'][row['index']]
            if (row.get('name')!='Tesla T4' or row.get('optimistic_weight_lower_bound_bytes')!=lower
                    or row['free_bytes']<lower+2*1024**3):raise ValueError('Actual per-device model lower-bound admission differs')
        frozen_integrity(files.json('BACKBONE_LIVE_BEFORE.json'),files.json('BACKBONE_LIVE_AFTER.json'),originals,parameter_devices=devices)
        audit_runtime(files.json('RUNTIME_RAW.json'),files.json('BACKBONE_LIVE_AFTER.json'),originals,
                      expected_map=load['plan']['device_map'],parameter_devices=devices,require_native_bf16=False)
        from .composite_task_memory_profile import verify_allocator_record
        for name in ['ALLOCATOR_BEFORE_COLLECTION.json','ALLOCATOR_AFTER_COLLECTION.json']:verify_allocator_record(files.json(name))
        for name in ['CHECKPOINT_LIVE_BEFORE.json','CHECKPOINT_LIVE_AFTER.json']:_checkpoint(files.json(name),model,layout)
        loading=files.json('MODEL_LOAD_ADMISSION.json')['loading']
        validate_loading_info({'missing_keys':loading['missing_keys'],'mismatched_keys':loading['mismatched_keys'],
          'error_msgs':loading['error_msgs'],'unexpected_keys':loading['ignored_checkpoint_only_MTP_keys']})
        capacity=files.json('MODEL_LOAD_ADMISSION.json')['adapter_capacity']
        if (capacity['adapter_parameters']!=233455616 or capacity['projection_count']!=496
                or capacity['covered_blocks']!=list(range(64))):raise ValueError('Full all64-block rank32 adapter capacity required')
        storage=capacity['per_device_storage_admission']['devices']
        if len(storage)!=2 or {r['device'] for r in storage}!={0,1}:raise ValueError('Per-device frozen inference storage required')
        for row in storage:
            needed=4*layout['device_parameters'][row['device']]
            if row['adapter_weight_bytes']!=needed or row['reserve_bytes']!=2*1024**3 or row['free_bytes']<needed+row['reserve_bytes']:
                raise ValueError('Full FP32 inference adapter/reserve admission differs')
        pairs=[]
        with _tokenizer(files,load) as (tokenizer,eos,vocabulary):
            previous=None
            for case_id in stage['partition']['case_ids']:
                part,previous=replay_pair(files,context,tokenizer,eos,vocabulary,case_id,previous)
                pairs.extend(part)
        counts=receipt.get('forward_counts');total=sum(r['logical_forwards'] for r in pairs)
        if (type(total) is not int or not 0<total<=stage['maximum_backbone_forwards']
                or digest(counts)!=digest({'attempted':total,'completed':total})):
            raise ValueError('Actual controlled hook counts differ from the complete native response count')
        for index in range(10,total+1,10):
            if digest(files.json('forward_progress/'+str(index).zfill(6)+'.json'))!=digest({
              'counts_lower_bound':{'attempted':index,'completed':index},'exact_at_exit':False}):
                raise ValueError('Forward callback prefix differs')
        collected=files.json('TASK_COLLECTION_RESULT.json')
        expected_pairs=[{'partition':stage['partition'],'case_id':case_id,'outcomes':[
          files.json('trajectories/'+case_id+'/'+arm+'/TERMINAL.json') for arm in ['base','always_on']],
          'checkpoint_sha256':model['checkpoint_sha256'],'live_checkpoint_before':files.json('CHECKPOINT_LIVE_BEFORE.json'),
          'live_checkpoint_after':files.json('CHECKPOINT_LIVE_AFTER.json'),'completed_trajectories':2,'optimizer_updates':0,
          'teacher_forcing':False,'decoder_repair':False,'source_raw_audit_still_required':True,
          'scientific_validation_admitted':False,'heldout_content_read':False} for case_id in stage['partition']['case_ids']]
        expected_collection={'partition':stage['partition'],'pairs':expected_pairs,
          'checkpoint_sha256':model['checkpoint_sha256'],'live_checkpoint_before':files.json('CHECKPOINT_LIVE_BEFORE.json'),
          'live_checkpoint_after':files.json('CHECKPOINT_LIVE_AFTER.json'),
          'completed_trajectories':stage['partition']['expected_trajectories'],'optimizer_updates':0,
          'teacher_forcing':False,'decoder_repair':False,'source_raw_audit_still_required':True,
          'scientific_validation_admitted':False,'heldout_content_read':False}
        if digest(collected)!=digest(expected_collection):
            raise ValueError('Complete collection result differs from the bound native pair')
        from .composite_task_skill_model_audit import KaggleClosure,_KEY as CLOSURE_KEY
        closed=(isinstance(closure,KaggleClosure) and closure._key is CLOSURE_KEY
          and closure._value.get('source_commit')==commit and closure._value.get('notebook')==stage['notebook']
          and closure._value.get('archive_sha256')==raw_sha256 and closure._value.get('owned_compute_released') is True)
        expected_code=(Path(root)/'notebooks/composite_task_skill_qualification.py').read_text().replace(
          'SOURCE_COMMIT_PLACEHOLDER',commit).replace('PACKAGE_MANIFEST_SHA256_PLACEHOLDER',sha(Path(root).parent/'TASK_TRAINING_PACKAGE_FILES.json'))
        if closed and closure._value['actual_code_cells_sha256']!=digest([expected_code]):
            raise PermissionError('External submitted notebook belongs to another exact source/input/partition package')
        result.update(completed_trajectories=stage['partition']['expected_trajectories'],recorded_forward_counts=counts,recorded_optimizer_updates=0,
          optimizer_updates=0,case_arm_outcomes=pairs,owned_compute_released=True if closed else None,
          provider_run_id=closure._value['scriptVersionId'] if closed else None,
          missing_or_unverified=[] if closed else ['independent_exact_provider_release_and_readback'],
          task_partition_admitted=closed,raw_evidence_audited=closed,
          full_checkpoint_base_and_native_bindings_verified=True)
        return TaskDevelopmentAudit(result,_KEY)
    finally:files.close()

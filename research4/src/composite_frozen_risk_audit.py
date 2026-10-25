"""Independent frozen risk/transfer raw audit; no model, fit or provider call."""
from copy import deepcopy
from pathlib import Path
from .io_utils import read,sha,digest
from .composite_frozen_risk_profile import validate_source,STAGE
from .composite_task_skill_package import require_frozen_task_package
from .composite_frozen_risk_data_reader import require_data
from .composite_v2_model_audit import EvidenceFiles,detector_integrity,frozen_integrity,audit_runtime

_KEY=object();_ARCHIVE_KEY=object()


class RiskArchive:
    def __init__(self,context,records,key=None):
        if key is not _ARCHIVE_KEY:raise PermissionError('Read the complete source/model/provider risk archive')
        self._key=key;self.context=context;self.records={(r['scene']['case_id'],r['condition']):r for r in records}


class FrozenRiskEvidence:
    def __init__(self,value,key=None):
        if key is not _KEY:raise PermissionError('Read actual complete frozen risk/transfer raw evidence')
        self._key=key;self._value=deepcopy(value)
    def receipt(self):return deepcopy(self._value)


def require_stage(value,phase=None):
    if not isinstance(value,FrozenRiskEvidence) or value._key is not _KEY:raise PermissionError('Genuine complete frozen-risk raw reader required')
    report=value.receipt()
    if (report.get('stage_integrity_admitted') is not True or report.get('raw_evidence_audited') is not True
            or report.get('owned_compute_released') is not True or report.get('missing_or_unverified')!=[]
            or (phase is not None and report.get('phase')!=phase)):
        raise PermissionError('Complete exact source/model/native/provider stage admission required')
    return report


def audit_stage(raw,raw_sha256,root,commit,fit_archive,qualification_archive,risk_data_archive,closure=None):
    context=validate_source(root,commit,fit_archive,qualification_archive,risk_data_archive)
    stage=context.stage();transfer=stage['phase']=='risk_detector_transfer'
    model=require_frozen_task_package(context.model);data=require_data(context.data)
    result={'reader':'option-a-frozen-risk-raw-v1','source_commit':commit,'phase':stage['phase'],
      'raw_archive_sha256':raw_sha256,'frozen_model_package_sha256':digest(model),
      'risk_data_release_sha256':data['release_sha256'],'checkpoint_sha256':model['checkpoint_sha256'],
      'detector_checkpoint_sha256':stage.get('detector_checkpoint_sha256'),
      'partition_index':None if transfer else stage['partition']['index'],
      'stage_integrity_admitted':False,'raw_evidence_audited':False,'owned_compute_released':None,
      'recorded_forward_counts':None,'completed_trajectories':None,'actual_transfer_report':None,
      'scientific_validation_admitted':False,'full_risk180_complete':False,'heldout_content_read':False,
      'new_model_forwards':0,'new_optimizer_updates':0,'missing_or_unverified':[]}
    files=EvidenceFiles(raw,raw_sha256,stage['maximum_raw_evidence_bytes'])
    try:
        run=files.json('RUN_RECEIPT.json')
        expected={'mode':stage['version'],'source_commit':commit,'phase':stage['phase'],'status':'complete',
          'evaluation_only':True,'training_objective':None,'model_instances':1 if transfer else 2,'optimizer_updates':0,
          'public_fixture_test_mode':False,'heldout_content_read':False,'owned_download_cache_removed':True,
          'counts_exact_at_controlled_exit':True}
        if any(digest(run.get(k))!=digest(v) for k,v in expected.items()):
            result['missing_or_unverified']=['complete_controlled_frozen_stage_exit'];return FrozenRiskEvidence(result,_KEY)
        freeze=files.json('PRECALL_FREEZE.json')
        if digest(freeze)!=digest({'stage':stage,'source_commit':commit,'data_archive_sha256':model['data_archive_sha256'],
          'input_manifest_sha256':sha(Path(root)/'inputs/MANIFEST.json'),'raw_training_rows_canonical_sha256':None,
          'source_release':read(Path(root)/'RELEASE.json')}):raise PermissionError('Risk pre-call source/input/model/data freeze differs')
        from .composite_frozen_risk_detector import selected_detector
        cfg=read(Path(root)/'config/composite_controller_rank32_v2.json');selected_detector(Path(root)/'inputs',cfg['modernbert'])
        detector_integrity(files,read(Path(root)/'inputs/MODERNBERT_PIN.json'),
          (Path(root)/'inputs/DETECTOR_SELECTED.pt').read_bytes(),stage['detector_checkpoint_sha256'])
        from .composite_frozen_probe import verify_software
        hardware=files.json('HARDWARE_PREFLIGHT.json')
        verify_software(hardware['software']['torch'],hardware['software']['packages'])
        if transfer and (hardware.get('visible_GPUs')!=1 or hardware.get('TF32_enabled') is not False
                         or len(hardware.get('devices',[]))!=1 or hardware['devices'][0].get('name')!='Tesla T4'):
            raise PermissionError('Actual fixed-detector hardware/software profile differs')
        collected=files.json('RISK_COLLECTION_RESULT.json')
        if collected.get('optimizer_updates')!=0 or collected.get('heldout_content_read') is not False:
            raise PermissionError('Frozen risk collection must not fit or open sealed data')
        if transfer:
            from .composite_v2_detector_transfer import verified_detector_tokenizer,transfer_report
            rows=context.data.reference_rows();wanted={'detector_predictions/'+r['state_id']+'.json' for r in rows}
            if {n for n in files.manifest if n.startswith('detector_predictions/')}!=wanted:
                raise PermissionError('Every exact fixed-detector reference prediction is required')
            records=[files.json('detector_predictions/'+r['state_id']+'.json') for r in rows]
            if digest(records)!=digest(collected.get('records')):raise PermissionError('Transfer collection and raw prediction rows differ')
            with verified_detector_tokenizer(files,root) as tokens:
                report=transfer_report(records,rows,cfg,commit,data['release_sha256'],stage['detector_checkpoint_sha256'],tokens)
            forwards=len(rows);actor_count=0;detector_count=forwards
            result['actual_transfer_report']=report
        else:
            from .composite_frozen_probe import verified_model_context,text_projection
            from .composite_task_skill_training import adapter_layout
            load=verified_model_context(root,commit,cfg['qwen'],stage,sha(Path(root)/'inputs'/STAGE))
            metadata=load['metadata'];originals={n:v['shape'] for n,v in metadata['tensors'].items() if not n.startswith('mtp.')}
            from .composite_placement import parameter_device
            devices={n:'cuda:'+str(parameter_device(n,load['plan'])) for n in originals}
            frozen_integrity(files.json('BACKBONE_LIVE_BEFORE.json'),files.json('BACKBONE_LIVE_AFTER.json'),originals,parameter_devices=devices)
            audit_runtime(files.json('RUNTIME_RAW.json'),files.json('BACKBONE_LIVE_AFTER.json'),originals,
                          expected_map=load['plan']['device_map'],parameter_devices=devices,require_native_bf16=False)
            from .composite_task_skill_evaluation_audit import _checkpoint
            layout=adapter_layout(metadata,load['plan'])
            for name in ['CHECKPOINT_LIVE_BEFORE.json','CHECKPOINT_LIVE_AFTER.json']:_checkpoint(files.json(name),model,layout)
            units=stage['partition']['units'];wanted={'traces/'+u['case_id']+'/'+u['arm']+'.json' for u in units}
            if {n for n in files.manifest if n.startswith('traces/')}!=wanted:raise PermissionError('Complete exact risk case/control coverage required')
            records=[files.json('traces/'+u['case_id']+'/'+u['arm']+'.json') for u in units]
            lookup={s['case_id']:s for s in context.data.scenes()}
            for unit,record in zip(units,records):
                if record.get('condition')!=unit['arm'] or digest(record.get('scene'))!=digest(lookup[unit['case_id']]):
                    raise PermissionError('Recorded risk scene/control differs from the exact fixed dataset unit')
            if digest(records)!=digest(collected.get('records')) or collected.get('completed_trajectories')!=len(units):
                raise PermissionError('Raw traces and completed collection differ')
            archive=RiskArchive(context,records,_ARCHIVE_KEY)
            from .composite_v2_trace_audit import verified_trace_tokenizers,audit_trace
            from .composite_v2_helper_audit import ordinary_task_training_exposure,read_helper_evidence
            from .composite_frozen_risk_runtime import trace_replay_scope
            from .composite_generalized_world import GeneralizedScene
            from .composite_native_capture import verify_feature_bytes
            design=read(Path(root)/'config/composite_generalization_amendment_v2.json')
            exposure=ordinary_task_training_exposure(context.model);raw_proofs={};helper_proofs={};features=[]
            with verified_trace_tokenizers(files,load,{'detector_files':files.json('DETECTOR_FILE_VERIFICATION.json')},root) as tokens:
                for record in records:
                    key=record['scene']['case_id']+'/'+record['condition'];scene=GeneralizedScene(**record['scene'])
                    with trace_replay_scope(archive,record):raw_proofs[key]=audit_trace(record,scene,commit,tokens)
                    with trace_replay_scope(archive,record,helper_pass=True):
                        helper_proofs[key]=read_helper_evidence(record,scene,commit,design,exposure,tokens).receipt()
                    for index,step in enumerate(record['trace']):
                        response=step['response'];descriptor=response['native_features']
                        name='native_features/'+key+'/response_'+str(index).zfill(3)+'.npz'
                        if descriptor.get('path')!=name or digest(files.json(name.removesuffix('.npz')+'.json'))!=digest(descriptor):
                            raise PermissionError('Actual native feature identity/metadata differs')
                        values,logits=verify_feature_bytes(files.get(name),descriptor,vocabulary=tokens.vocabulary)
                        if (descriptor.get('additional_capture_forwards')!=0 or descriptor.get('case_id')!=scene.case_id
                                or descriptor.get('arm')!=record['condition'] or descriptor.get('index')!=index
                                or descriptor.get('adapter_enabled') is not response['selected_gate']
                                or descriptor.get('input_token_ids')!=response['input_token_ids']
                                or descriptor.get('native_prefill_tokens')!=len(response['input_token_ids'])
                                or int(logits.argmax())!=response['generated_token_ids'][0]):
                            raise PermissionError('Captured prefill is not the actual native model response')
                        if len(descriptor['layers'])!=64 or [r['layer'] for r in descriptor['layers']]!=list(range(64)):
                            raise PermissionError('All64 activation layers are required')
                        for layer in descriptor['layers']:
                            device=parameter_device('model.language_model.layers.'+str(layer['layer'])+'.input_layernorm.weight',load['plan'])
                            if layer.get('device')!='cuda:'+str(device) or layer.get('source_dtype') not in ['bfloat16','float32']:
                                raise PermissionError('Activation layer device/precision differs from the frozen full placement')
                        features.append(name)
            if {n for n in files.manifest if n.endswith('.npz')}!=set(features):raise PermissionError('Missing or extra native activation arrays')
            actor_count=sum(len(s['response']['generated_token_ids']) for r in records for s in r['trace'])
            detector_count=sum(o['memory']['prediction']['raw'].get('context_budget_failure') is not True
              for r in records for o in [r['initial_observation']]+[s['observation_after'] for s in r['trace']])
            forwards=actor_count+detector_count
            result.update(completed_trajectories=len(records),raw_trace_audits=raw_proofs,helper_audits=helper_proofs,
                          trace_manifest={r['scene']['case_id']+'/'+r['condition']:digest(r) for r in records},
                          activation_files_verified=len(features),records=records)
        if (digest(run.get('forward_counts'))!=digest({'attempted':forwards,'completed':forwards})
                or not 0<forwards<=stage['maximum_backbone_forwards']
                or digest(collected.get('detector_forward_counts'))!=digest({'attempted':detector_count,'completed':detector_count})
                or (not transfer and digest(collected.get('actor_forward_counts'))!=digest({'attempted':actor_count,'completed':actor_count}))):
            raise PermissionError('Actual exact joint forward counts differ from recorded native/recognition evidence')
        for index in range(10,forwards+1,10):
            if digest(files.json('forward_progress/'+str(index).zfill(6)+'.json'))!=digest({
              'counts_lower_bound':{'attempted':index,'completed':index},'exact_at_exit':False}):raise PermissionError('Joint forward journal differs')
        from .composite_task_skill_model_audit import KaggleClosure,_KEY as CLOSURE_KEY
        closed=(isinstance(closure,KaggleClosure) and closure._key is CLOSURE_KEY
          and closure._value['source_commit']==commit and closure._value['notebook']==stage['notebook']
          and closure._value['archive_sha256']==raw_sha256 and closure._value['owned_compute_released'] is True)
        code=(Path(root)/'notebooks/composite_task_skill_qualification.py').read_text(encoding='utf-8-sig').replace(
          'SOURCE_COMMIT_PLACEHOLDER',commit).replace('PACKAGE_MANIFEST_SHA256_PLACEHOLDER',sha(Path(root).parent/'TASK_TRAINING_PACKAGE_FILES.json'))
        if closed and closure._value['actual_code_cells_sha256']!=digest([code]):raise PermissionError('Exact submitted risk source/input code differs')
        if transfer:
            result['actual_transfer_report'].update(model_integrity_audited=True,raw_archive_sha256=raw_sha256,
                                                    owned_compute_released=True if closed else None)
        result.update(stage_integrity_admitted=closed,raw_evidence_audited=closed,
          owned_compute_released=True if closed else None,provider_run_id=closure._value['scriptVersionId'] if closed else None,
          missing_or_unverified=[] if closed else ['independent_exact_provider_release_and_code_readback'],
          recorded_forward_counts={'attempted':forwards,'completed':forwards},recorded_optimizer_updates=0)
        return FrozenRiskEvidence(result,_KEY)
    finally:files.close()

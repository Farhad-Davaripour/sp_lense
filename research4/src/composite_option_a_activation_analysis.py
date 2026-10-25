"""Descriptive layer contrasts from already audited native-prefill bytes.

No model is constructed or called. Only equal initial input-token sequences
are contrasted with BASE; divergent prompts stay incomparable, not zero.
"""
from copy import deepcopy
import numpy as np
from .io_utils import digest
from .composite_task_skill_package import require_frozen_task_package
from .composite_frozen_risk_data_reader import require_data
from .composite_frozen_risk_audit import require_stage
from .composite_v2_controller import CONDITIONS
from .composite_v2_model_audit import EvidenceFiles
from .composite_native_capture import verify_feature_bytes
from .composite_option_a_reporting import _binding


def _contrast(left,right):
    left=np.asarray(left,dtype=np.float64);right=np.asarray(right,dtype=np.float64)
    if left.shape!=right.shape or left.ndim!=1 or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('Equal complete finite vector geometry required')
    delta=right-left;a=float(np.linalg.norm(left));b=float(np.linalg.norm(right))
    cosine=None if a==0 or b==0 else float(np.dot(left,right)/(a*b))
    return {'BASE_norm':a,'arm_norm':b,'L2_difference':float(np.linalg.norm(delta)),
      'maximum_absolute_difference':float(np.max(np.abs(delta))),
      'cosine_similarity':None if cosine is None else max(-1.0,min(1.0,cosine)),
      'exact_array_equal':bool(np.array_equal(left,right))}


def activation_report(model_evidence,data_evidence,partition_evidence,archives):
    model=require_frozen_task_package(model_evidence);data=require_data(data_evidence)
    scenes=data_evidence.scenes();scene_lookup={s['case_id']:s for s in scenes}
    expected={(s['case_id'],arm) for s in scenes for arm in CONDITIONS}
    proofs=[];run_ids=set();raw_ids=set()
    for evidence in partition_evidence:
        proof=require_stage(evidence,'risk_development');transport=_binding(proof,model)
        run=transport['provider_run_id'];raw=transport['raw_archive_sha256']
        if (run in run_ids or raw in raw_ids or proof.get('risk_data_release_sha256')!=data['release_sha256']):
            raise PermissionError('Unique exact source/model/data/provider risk partitions required')
        run_ids.add(run);raw_ids.add(raw);proofs.append((proof,transport))
    if not isinstance(archives,dict) or set(archives)!=run_ids:
        raise PermissionError('Provide exactly the original archive path for every audited provider result')
    lookup={};manifest=[]
    for proof,transport in proofs:
        files=EvidenceFiles(archives[transport['provider_run_id']],transport['raw_archive_sha256'],4*1024**3)
        try:
            for record in proof['records']:
                case=record['scene']['case_id'];arm=record['condition'];unit=(case,arm)
                if unit not in expected or unit in lookup or digest(record['scene'])!=digest(scene_lookup[case]):
                    raise PermissionError('Duplicate or foreign actual frozen risk unit')
                key=case+'/'+arm
                if proof['trace_manifest'].get(key)!=digest(record) or not record['trace']:
                    raise PermissionError('Actual complete native trace manifest required')
                response=record['trace'][0]['response'];descriptor=response['native_features']
                path='native_features/'+key+'/response_000.npz'
                if (descriptor.get('path')!=path or descriptor.get('case_id')!=case or descriptor.get('arm')!=arm
                        or descriptor.get('index')!=0 or descriptor.get('additional_capture_forwards')!=0
                        or descriptor.get('input_token_ids')!=response['input_token_ids']
                        or descriptor.get('adapter_enabled') is not response['selected_gate']
                        or digest(files.json(path.removesuffix('.npz')+'.json'))!=digest(descriptor)):
                    raise PermissionError('First native prefill identity/descriptor differs from the audited trace')
                features,logits=verify_feature_bytes(files.get(path),descriptor)
                if int(logits.argmax())!=response['generated_token_ids'][0]:
                    raise PermissionError('Captured native logits do not predict the observed first generated token')
                lookup[unit]={'features':features,'logits':logits,'input_token_ids':response['input_token_ids'],
                  'adapter_enabled':response['selected_gate'],'feature_sha256':descriptor['sha256'],
                  'provider_run_id':transport['provider_run_id']}
            manifest.append(deepcopy(transport))
        finally:files.close()
    contrasts=[]
    for scene in sorted(scenes,key=lambda s:s['case_id']):
        case=scene['case_id'];base=lookup.get((case,'base'))
        for arm in CONDITIONS[1:]:
            value=lookup.get((case,arm));reason=None
            if base is None or value is None:reason='missing_audited_case_arm'
            elif base['input_token_ids']!=value['input_token_ids']:reason='different_initial_native_prompt_tokens'
            row={'case_id':case,'group_id':scene['group_id'],'mechanism':scene['mechanism'],
              'scenario':scene['condition'],'arm':arm,'BASE_arm':'base','comparison_available':reason is None,
              'unavailable_reason':reason,'layer_contrasts':None,'prefill_logit_contrast':None,
              'adapter_state_differs':None,'exact_initial_prompt_match':None}
            if reason is None:
                row.update(exact_initial_prompt_match=True,adapter_state_differs=base['adapter_enabled']!=value['adapter_enabled'],
                  BASE_adapter_enabled=base['adapter_enabled'],arm_adapter_enabled=value['adapter_enabled'],
                  BASE_feature_sha256=base['feature_sha256'],arm_feature_sha256=value['feature_sha256'],
                  BASE_provider_run_id=base['provider_run_id'],arm_provider_run_id=value['provider_run_id'],
                  layer_contrasts=[{'layer':index,**_contrast(a,b)} for index,(a,b) in enumerate(zip(base['features'],value['features']))],
                  prefill_logit_contrast=_contrast(base['logits'],value['logits']))
            contrasts.append(row)
    complete=set(lookup)==expected
    return {'version':'option-a-recorded-native-initial-prefill-layer-contrasts-v1',
      'source_commit':model['evaluation_source_commit'],'frozen_model_package_sha256':digest(model),
      'checkpoint_sha256':model['checkpoint_sha256'],'risk_data_release_sha256':data['release_sha256'],
      'expected_native_case_arm_prefills':180,'observed_audited_native_prefills':len(lookup),
      'complete_raw_prefill_coverage':complete,'missing_case_arm_prefills':sorted(expected-set(lookup)),
      'expected_BASE_arm_contrasts':144,'available_equal_prompt_contrasts':sum(r['comparison_available'] for r in contrasts),
      'contrasts':contrasts,'raw_partition_manifest':manifest,
      'capture_location':'each block output at the final input token of the first native prefill',
      'layers':64,'hidden_size':5120,'additional_capture_forwards':0,'new_model_forwards':0,'new_optimizer_updates':0,
      'divergent_later_prompts_compared_as_interventions':False,'adapter_state_equal_is_distinct_intervention':False,
      'activation_to_shutdown_policy_fit_or_selection':False,'mechanistic_causality_or_internal_gate_localization_proved':False,
      'population_inference_admitted':False,'scientific_validation_admitted':False,'heldout_opening_admitted':False,
      'heldout_content_read':False}

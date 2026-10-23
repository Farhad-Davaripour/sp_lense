"""Read-only audit of durable native steps before an interrupted base episode."""
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

from .io_utils import read,digest
from .composite_frozen_base import authorized_load_plan
from .composite_frozen_base_audit import (audit_archive,ArchiveFiles,parameter_audit,
    native_response,audit_features,audit_instrumentation)
from .composite_frozen_metadata import verified_tokenizers
from .composite_frozen_base_eval import selected_detector
from .composite_frozen_audit import audit_runtime_metadata
from .composite_audit import audit_trajectory_prefix
from .composite_runtime import FrozenInputs
from .composite_environment import ENVIRONMENT


def memory_diagnosis(responses):
    if not responses:raise ValueError('A memory diagnosis requires actual response samples')
    devices=[]
    for device in [0,1]:
        rows=[response['joint_device_memory'][device] for response in responses]
        if any(row['device']!=device for row in rows):raise ValueError('Memory sample device order differs')
        devices.append({'device':device,'samples':len(rows),
            'post_response_allocated_constant':len({row['allocated_bytes'] for row in rows})==1,
            'first_allocated_bytes':rows[0]['allocated_bytes'],'last_allocated_bytes':rows[-1]['allocated_bytes'],
            'first_reserved_bytes':rows[0]['reserved_bytes'],'last_reserved_bytes':rows[-1]['reserved_bytes'],
            'last_free_bytes':rows[-1]['free_bytes'],
            'last_reserved_minus_allocated_bytes':rows[-1]['reserved_bytes']-rows[-1]['allocated_bytes']})
    return {'devices':devices,'minimum_driver_free_bytes':512*1024**2,
            'allocator_retention_supported':all(row['post_response_allocated_constant'] and row['last_reserved_bytes']>row['first_reserved_bytes'] for row in devices),
            'successful_reclaimability_measured':False,'CUDA_OOM_demonstrated':False,
            'failing_device_and_exact_failing_free_bytes':None}


def replay_prefix(files,context,tokenizers,eos,response_count):
    if type(response_count) is not int or not 1<=response_count<=24:
        raise ValueError('A durable incomplete scene prefix must obey the original hard stop')
    case_id=context['evaluation_plan']['case_ids'][0]
    scene=next(scene for scene in context['scenes'] if scene['case_id']==case_id)
    steps=[files.json('steps/'+case_id+'_'+str(i+1).zfill(3)+'.json') for i in range(response_count)]
    initial=files.json('observations/'+case_id+'_000.json')
    final=steps[-1]['observation_after']
    record={'scene':scene,'condition':'base','environment':ENVIRONMENT,'initial_observation':initial,
            'trace':steps,'outcome':steps[-1]['outcome'],'terminal_memory':final['memory']}
    bt,qt=tokenizers
    replayed=audit_trajectory_prefix(record,0.9,bt,qt,eos,'reference',{'detector':2048,'actor':6144})
    if replayed['trajectory_complete'] is not False:
        raise ValueError('This incomplete-prefix profile cannot replace a completed trajectory')
    observations=[initial]+[step['observation_after'] for step in steps]
    responses=[];detector_forwards=0;vocabulary=context['metadata']['tensors']['lm_head.weight']['shape'][0]
    for observation in observations:
        if files.json('observations/'+case_id+'_'+str(observation['sequence']).zfill(3)+'.json')!=observation:
            raise ValueError('Durable prefix observation differs from the simulator-step replay')
        detector_forwards+=int(observation['memory']['prediction']['raw'].get('context_budget_failure') is not True)
    for index,step in enumerate(steps):
        response=step['response'];native_response(response,eos,vocabulary)
        audit_features(files,response,index,context,observations[index]['text']);responses.append(response)
    baseline_forwards=audit_instrumentation(files,responses[0],eos,vocabulary)
    counts=detector_forwards+baseline_forwards+sum(len(response['generated_token_ids']) for response in responses)
    boundary={'case_id':case_id,'responses':response_count,'sequence':final['sequence'],
        'world_outcome':record['outcome'],'predicted_memory':final['memory'],
        'visible_text_sha256':hashlib.sha256(final['text'].encode()).hexdigest(),
        'durable_prefix_sha256':digest(record),'source_record_role':'incomplete_prefix_not_completed_trajectory'}
    return {'prefix_replay':replayed,'last_complete_tool_boundary':boundary,
        'fully_accounted_prefix_forwards':counts,'detector_prefix_forwards':detector_forwards,
        'native_complete_responses':response_count,'baseline_instrumentation_forwards':baseline_forwards,
        'completed_trajectories':0,'memory_diagnosis':memory_diagnosis(responses),
        'full_validation_admitted':False,'heldout_content_read':False}


def audit_partial_archive(archive,source,commit,transport,controller,export):
    basic=audit_archive(archive,source,commit,transport,controller,export)
    if basic['worker_status']!='failed' or basic['complete_base_partition_audited'] is not False:
        raise ValueError('A partial recovery parent must remain a failed original partition')
    source=Path(source);context=authorized_load_plan(source,commit,read(source/'config/composite_controller_rank32_v2.json')['qwen'],transport)
    with zipfile.ZipFile(archive) as stream:
        files=ArchiveFiles(stream);receipt=files.json('RUN_RECEIPT.json');journal=files.json('FORWARD_COUNTS.json')
        if (receipt['error_type']!='MemoryError' or receipt['error']!='Frozen probe lost its per-device export reserve'
            or controller['child_state']['child_exit_confirmed'] is not True
            or controller['cleanup']['owned_cache_removed'] is not True
            or controller['evidence']['evidence_kind']!='original_worker_export'):
            raise ValueError('This prefix audit is limited to a released controlled reserve refusal')
        counts=receipt['forward_counts']
        if (controller['counts']['counts_exact_at_exit'] is not True or receipt['counts_exact_at_controlled_exit'] is not True
            or any(type(counts[key]) is not int for key in ['attempted','completed'])
            or not 0<counts['attempted']==counts['completed']<=30000
            or counts!=controller['counts']['forward_counts'] or counts!=receipt['independent_joint_counts']
            or journal['sealed_for_exit'] is not True or any(journal[key]!=counts[key] for key in counts)
            or export is None or any(export.get(key)!=value for key,value in receipt.items())):
            raise ValueError('Partial native forward counts are not sealed and mutually reconciled')
        names=stream.namelist();responses=sorted(name for name in names if re.fullmatch(r'responses/response_[0-9]{5}\.json',name))
        n=len(responses)
        if responses!=['responses/response_'+str(i).zfill(5)+'.json' for i in range(n)]:
            raise ValueError('Partial native responses contain a missing or duplicate index')
        if any(name.startswith('trajectories/') for name in names):
            raise ValueError('This prefix-only profile cannot hide a completed trajectory')
        first=context['evaluation_plan']['case_ids'][0]
        if {name for name in names if name.startswith('steps/')}!={
            'steps/'+first+'_'+str(i+1).zfill(3)+'.json' for i in range(n)}:
            raise ValueError('A partial prefix cannot hide extra or missing simulator steps')
        if {name for name in names if name.startswith('features/')}!={
            'features/response_'+str(i).zfill(5)+'.npz' for i in range(n)}:
            raise ValueError('A partial prefix cannot hide extra or missing feature records')
        report=files.json('FROZEN_BASE_REPORT.json')
        if report['completed_case_ids']!=[] or report['missing_case_ids']!=context['evaluation_plan']['case_ids']:
            raise ValueError('All original partition cases must remain pending')
        package=files.json('FROZEN_BASE_PACKAGE.json')
        if package['plan']!=context['evaluation_plan'] or package['backbone_instances']!={'ModernBERT':1,'Qwen':1} or package['behavior_adapter'] is not None:
            raise ValueError('Partial package belongs to another frozen control/model plan')
        loaded=files.json('FROZEN_LOADED_LAYOUT.json')
        if loaded!=package['actor'] or loaded['model_files']!=files.json('MODEL_FILE_VERIFICATION.json'):
            raise ValueError('Partial native load and file-verification provenance differ')
        loading=loaded['loading_info']
        if (loading['missing_keys'] or loading['mismatched_keys'] or loading['error_msgs']
            or any(not key.startswith('mtp.') for key in loading['ignored_checkpoint_only_MTP_keys'])):
            raise ValueError('Partial native model loading was not the complete original reference model')
        audit_runtime_metadata(files.json('HARDWARE_PREFLIGHT.json'),context)
        saved=selected_detector(FrozenInputs(transport,context['pin']['transport_manifest_sha256']),read(source/'config/composite_controller_rank32_v2.json'))
        parameters=parameter_audit(files,context,package,saved)
        with verified_tokenizers(files,context,package,source) as (bt,qt,eos):
            prefix=replay_prefix(files,context,(bt,qt),eos,n)
        extra=counts['completed']-prefix['fully_accounted_prefix_forwards']
        if not 0<=extra<=160:raise ValueError('Unaccounted forwards cannot belong to one interrupted native response')
        basic.update(partial_prefix_audited=True,parameters=parameters,**prefix,
            sealed_forward_counts=counts,additional_interrupted_response_forwards=extra,
            interrupted_response_output_durable=False,interrupted_KV_cache_durable=False,
            exact_interrupted_generation_resume_supported=False,
            all_original_partition_cases_pending=True,
            recovery_requires_explicit_versioned_source_and_lineage=True)
    return basic

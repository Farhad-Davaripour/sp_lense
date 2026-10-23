"""Offline replay of original base trajectories and lossless native features."""
import hashlib
import io
import json
import re
from pathlib import Path
import zipfile

import numpy as np

from .io_utils import read, sha
from .composite_frozen_base import MODE, BOUNDS, authorized_load_plan, base_report
from .composite_frozen_base_worker import DISK_POLICY
from .composite_exports import verify_evidence_archive
from .composite_runtime import FrozenInputs
from .composite_frozen_probe import (validate_download_proof, verify_metadata_bytes, audit_loaded_layout,
    parameter_device)
from .composite_native_capture import verify_feature_bytes
from .composite_frozen_audit import audit_runtime_metadata
from .composite_audit import audit_trajectory
from .composite_environment import ENVIRONMENT
from .composite_frozen_metadata import verified_tokenizers


def native_response(response, eos, vocabulary):
    tokens = response['generated_token_ids']; cache = response['cache']
    if (response['adapter_enabled'] is not False or response['kernel_profile'] != 'reference'
        or response['generation_parameters'] != {'max_new_tokens':160,'do_sample':False,'num_beams':1,'num_return_sequences':1,'use_cache':True}
        or not 0 < len(tokens) <= 160 or any(type(token) is not int or not 0 <= token < vocabulary for token in tokens)
        or any(token in eos for token in tokens[:-1])
        or (tokens[-1] not in eos and len(tokens) != 160)):
        raise ValueError('Frozen native generation budget/decoding contract differs')
    if (cache['cache_object_present'] is not True or cache['prefill_checked'] is not True
        or cache['initial_cache_length'] != 0 or cache['previous_cache_reused'] is not False
        or type(cache['cache_identity']) is not int or cache['cache_identity'] <= 0
        or type(cache['previous_cache_alive']) is not bool
        or cache['previous_cache_alive'] != (cache['previous_cache_identity'] is not None)
        or (cache['previous_cache_alive'] and (type(cache['previous_cache_identity']) is not int or cache['previous_cache_identity'] <= 0))
        or (cache['previous_cache_alive'] and cache['cache_identity'] == cache['previous_cache_identity'])):
        raise ValueError('Fresh native response-cache provenance differs')


def parameter_audit(files, context, package, saved_detector):
    from .composite_training import tensor_digest
    before = files.json('QWEN_PARAMETERS_BEFORE.json'); after = files.json('QWEN_PARAMETERS_AFTER.json')
    if before != after or set(before) != set(context['metadata_bounds']['runtime_parameter_names']):
        raise ValueError('Original Qwen parameter coverage or frozen fingerprints differ')
    for name, value in before.items():
        if (value['requires_grad'] is not False or value['logical_shape'] != context['metadata']['tensors'][name]['shape']
            or value['device'] != 'cuda:'+str(parameter_device(name, context['plan']))):
            raise ValueError('Original Qwen parameter geometry/device/freeze differs')
    if files.json('QWEN_AUXILIARY_BEFORE.json') != files.json('QWEN_AUXILIARY_AFTER.json'):
        raise ValueError('Qwen buffers or quantization metadata changed')
    if not re.fullmatch('[0-9a-f]{64}',files.json('QWEN_AUXILIARY_AFTER.json')['sha256']):
        raise ValueError('Missing complete Qwen auxiliary fingerprint')
    audit_loaded_layout(package['actor']['loaded_layout'], context, before)
    detector = files.json('DETECTOR_PARAMETERS_BEFORE.json')
    if detector != files.json('DETECTOR_PARAMETERS_AFTER.json') or len({item['name'] for item in detector}) != len(detector):
        raise ValueError('Frozen detector parameter coverage/fingerprints differ')
    lookup = {item['name']:item for item in detector}
    if any(item['requires_grad'] is not False or item['device'] != 'cuda:0' for item in detector):
        raise ValueError('All detector parameters, adapters and heads must be frozen on device0')
    expected = {**{'encoder.'+key:value for key,value in saved_detector['lora'].items()},
                **{'heads.'+key:value for key,value in saved_detector['heads'].items()}}
    for name, tensor in expected.items():
        value = lookup.get(name)
        if value is None or value['sha256'] != tensor_digest(tensor) or value['shape'] != list(tensor.shape):
            raise ValueError('Frozen detector adapter/head bytes differ from the admitted checkpoint')
    return {'Qwen_parameters':len(before),'detector_parameters':len(detector),
            'restored_detector_tensors':len(expected),'optimizer_updates':0}


class ArchiveFiles:
    def __init__(self, stream): self.stream = stream
    def get(self, name): return self.stream.read(name)
    def json(self, name): return json.loads(self.get(name))


def audit_features(files,response,number,context,prompt):
    vocabulary=context['metadata']['tensors']['lm_head.weight']['shape'][0]
    memory=response['joint_device_memory']
    if len(memory)!=2 or [item['device'] for item in memory]!=[0,1]:
        raise ValueError('Both devices require joint-model runtime memory evidence')
    for item in memory:
        if (item['free_bytes']<512*1024**2
            or not 0<=item['allocated_bytes']<=item['reserved_bytes']<=item['total_bytes']
            or not item['allocated_bytes']<=item['peak_allocated_bytes']<=item['peak_reserved_bytes']<=item['total_bytes']
            or item['reserved_bytes']>item['peak_reserved_bytes']):
            raise ValueError('Actual joint-model memory bounds differ')
    descriptor=response['frozen_prefill_features'];path=descriptor['path']
    if path!='features/response_'+str(number).zfill(5)+'.npz':
        raise ValueError('Native feature response order or uniqueness differs')
    if (descriptor['prompt_sha256']!=hashlib.sha256(prompt.encode()).hexdigest()
        or descriptor['native_prefill_tokens']!=len(response['input_token_ids'])
        or descriptor['additional_capture_forwards']!=0):
        raise ValueError('Native feature prompt/call provenance differs')
    _,logits=verify_feature_bytes(files.get(path),descriptor,vocabulary=vocabulary)
    expected_layers=[{'layer':i,'device':'cuda:'+str(0 if i<18 else 1),'source_dtype':'bfloat16'} for i in range(64)]
    if descriptor['layers']!=expected_layers:raise ValueError('Actual native feature layer/device coverage differs')
    saved=files.json('responses/response_'+str(number).zfill(5)+'.json')
    if any(response.get(name)!=value for name,value in saved.items()):
        raise ValueError('Native response changed after pre-execution persistence')
    return logits


def audit_instrumentation(files,first_response,eos,vocabulary,prefix=''):
    if prefix not in ['', 'continuation_parity/']:raise ValueError('Unregistered instrumentation-proof namespace')
    parity=files.json(prefix+'INSTRUMENTATION_PARITY.json');baseline=files.json(prefix+'BASELINE_NATIVE_RESPONSE.json')
    native_response(baseline,eos,vocabulary)
    fields=['input_token_ids','generated_token_ids','raw_text','actions','finished_with_eos']
    if (parity['baseline_generation']!=baseline or any(baseline[key]!=first_response[key] for key in fields)
        or any(parity['captured_generation'][key]!=first_response[key] for key in fields)
        or parity['exact_native_outputs_equal'] is not True or parity['exact_prefill_logits_equal'] is not True
        or parity['adapter_OFF_bypass'] is not None or parity['adapter_ON_effect'] is not None):
        raise ValueError('Raw matched-prefix instrumentation comparison differs')
    raw=files.get(prefix+'BASELINE_PREFILL_LOGITS.npy')
    if hashlib.sha256(raw).hexdigest()!=parity['baseline_logit_file_sha256']:
        raise ValueError('Baseline raw logit bytes changed')
    logits=np.load(io.BytesIO(raw),allow_pickle=False)
    _,captured=verify_feature_bytes(files.get(first_response['frozen_prefill_features']['path']),first_response['frozen_prefill_features'],vocabulary=vocabulary)
    if (logits.dtype!=np.float32 or not np.array_equal(logits,captured)
        or hashlib.sha256(logits.tobytes()).hexdigest()!=parity['baseline_logit_sha256']
        or parity['baseline_logit_sha256']!=parity['captured_logit_sha256']):
        raise ValueError('Layer capture changed the exact raw prefill logits')
    return len(baseline['generated_token_ids'])


def replay(files, context, package, tokenizers, eos):
    bert, qwen = tokenizers
    plan = context['evaluation_plan']; lookup = {scene['case_id']:scene for scene in context['scenes']}
    records = []; responses = []; feature_paths = set(); detector_forwards = 0
    vocabulary = context['metadata']['tensors']['lm_head.weight']['shape'][0]
    for key in plan['case_ids']:
        record = files.json('trajectories/'+key+'.json')
        if record['scene'] != lookup[key] or record['condition'] != 'base' or record['environment'] != ENVIRONMENT:
            raise ValueError('Another original scene, control or environment entered the base endpoint')
        expected_epoch = 2 if not records else records[-1]['terminal_memory']['cache_epoch']+2
        if record['initial_observation']['memory']['cache_epoch'] != expected_epoch:
            raise ValueError('Episode reset/observation epoch did not advance from a cleared state')
        audit_trajectory(record, 0.9, bert, qwen, eos, 'reference', {'detector':2048,'actor':6144})
        observations = [record['initial_observation']]+[step['observation_after'] for step in record['trace']]
        for observation in observations:
            name = 'observations/'+key+'_'+str(observation['sequence']).zfill(3)+'.json'
            if files.json(name) != observation: raise ValueError('Pre-response observation changed in the final trace')
            detector_forwards += int(observation['memory']['prediction']['raw'].get('context_budget_failure') is not True)
        for index, step in enumerate(record['trace']):
            if files.json('steps/'+key+'_'+str(index+1).zfill(3)+'.json') != step:
                raise ValueError('Durable simulator step changed in the final trace')
            response = step['response']; native_response(response, eos, vocabulary)
            number = len(responses);path=response['frozen_prefill_features']['path']
            if path in feature_paths:raise ValueError('Native feature response order or uniqueness differs')
            prompt = observations[index]['text']
            audit_features(files,response,number,context,prompt)
            feature_paths.add(path); responses.append(response)
        records.append(record)
    report = base_report(records, plan['case_ids'])
    if files.json('FROZEN_BASE_REPORT.json') != report: raise ValueError('Raw base report differs from complete trajectory replay')
    if not responses: raise ValueError('A complete base partition must contain a native response')
    baseline_forwards=audit_instrumentation(files,responses[0],eos,vocabulary)
    expected_forwards = detector_forwards+sum(len(response['generated_token_ids']) for response in responses)+baseline_forwards
    return records, {'trajectories':len(records),'native_responses':len(responses),
        'detector_forwards':detector_forwards,'expected_joint_forwards':expected_forwards,
        'activation_layers_per_response':64,'instrumentation_parity_verified':True,
        'full_validation_admitted':False,'heldout_content_read':False}


def audit_archive(archive, source, commit, transport, controller, export):
    source = Path(source); context = authorized_load_plan(source, commit,
        read(source/'config/composite_controller_rank32_v2.json')['qwen'], transport)
    raw = Path(archive).read_bytes()
    with zipfile.ZipFile(io.BytesIO(raw)) as stream:
        if sum(item.file_size for item in stream.infolist()) > DISK_POLICY['maximum_raw_evidence_bytes']:
            raise ValueError('Base raw evidence exceeds its complete export bound')
    integrity = verify_evidence_archive(raw)
    if (controller['source_commit'] != commit or controller['mode'] != MODE
        or controller['notebook'] != context['pin']['notebook']
        or controller['child_state']['child_exit_confirmed'] is not True
        or controller['evidence']['archive_sha256'] != hashlib.sha256(raw).hexdigest()):
        raise ValueError('Base archive/controller exact source/run identity differs')
    report = {'archive_sha256':hashlib.sha256(raw).hexdigest(),**integrity,'source_commit':commit,
        'partition':context['evaluation_plan']['partition'],'case_ids':context['evaluation_plan']['case_ids'],
        'transport_manifest_sha256':context['pin']['transport_manifest_sha256'],
        'complete_base_partition_audited':False,'full_system_validation_complete':False,
        'full_validation_admitted':False,'heldout_confirmation':'not_performed','heldout_content_read':False,
        'audit_model_forwards':0,'optimizer_updates':0,'cross_device_numerical_parity':'not_established'}
    with zipfile.ZipFile(io.BytesIO(raw)) as stream:
        files = ArchiveFiles(stream); names=set(stream.namelist())
        receipt = files.json('RUN_RECEIPT.json') if 'RUN_RECEIPT.json' in names else None
        if receipt is not None and (receipt['source_commit'] != commit or receipt['stage'] != MODE or receipt['heldout_content_read'] is not False
            or receipt['behavior_adapter_fitting'] is not False or type(receipt['optimizer_updates']) is not int or receipt['optimizer_updates'] != 0):
            raise ValueError('Base worker terminal scope differs')
        report['worker_status'] = receipt['status'] if receipt is not None else 'unknown_abrupt_exit'
        counts_observed=controller['counts']
        if counts_observed['counts_exact_at_exit'] is not True and counts_observed['forward_counts'] != {'attempted':None,'completed':None}:
            raise ValueError('Unknown terminal counts must remain unknown')
        report.update(counts_exact_at_exit=counts_observed['counts_exact_at_exit'],forward_counts=counts_observed['forward_counts'])
        freeze = files.json('PRECALL_FREEZE.json')
        if freeze != {'source_commit':commit,'mode':MODE,'pin':context['pin'],
            'input_manifest_sha256':sha(source/'inputs/MANIFEST.json'),'bounds':BOUNDS,'disk_policy':DISK_POLICY}:
            raise ValueError('Raw frozen-base source/input/envelope changed')
        if receipt is None or receipt['status'] != 'complete':
            if controller['evidence']['evidence_kind']=='controller_post_exit_rescue':
                observed=files.json('CONTROLLER_TERMINAL_OBSERVATION.json')
                if any(observed.get(key)!=controller.get(key) for key in ['source_commit','mode','notebook','child_state','counts']):
                    raise ValueError('Post-exit partial rescue identity differs')
            return report
        if (export is None or export['archive_sha256'] != report['archive_sha256'] or export['archive_bytes'] != len(raw)
            or any(export.get(key) != value for key,value in receipt.items())
            or controller['evidence']['evidence_kind'] != 'original_worker_export'
            or controller['child_state']['exit_code'] != 0 or controller['complete_collection_exported'] is not True
            or controller['cleanup']['owned_cache_removed'] is not True
            or controller['counts']['counts_exact_at_exit'] is not True):
            raise ValueError('Complete base collection lacks exact terminal export/cleanup proof')
        counts = receipt['forward_counts']
        if (any(type(counts[key]) is not int for key in ['attempted','completed'])
            or not 0 < counts['completed'] == counts['attempted'] <= 30000
            or controller['counts']['forward_counts'] != counts or receipt['independent_joint_counts'] != counts):
            raise ValueError('Exact joint backbone forward accounting differs')
        journal = files.json('FORWARD_COUNTS.json')
        if journal.get('sealed_for_exit') is not True or any(journal[key] != counts[key] for key in counts):
            raise ValueError('Terminal joint counter journal differs')
        package = files.json('FROZEN_BASE_PACKAGE.json')
        if (package['plan'] != context['evaluation_plan'] or package['backbone_instances'] != {'ModernBERT':1,'Qwen':1}
            or package['behavior_adapter'] is not None or package['trainable_parameters'] != 0
            or package['full_validation_admitted'] is not False):
            raise ValueError('Base package identity, frozen model count or claims differ')
        loaded=files.json('FROZEN_LOADED_LAYOUT.json')
        if loaded!=package['actor']:raise ValueError('Raw native load proof changed in the packaged record')
        loading=loaded['loading_info']
        if loading['missing_keys'] or loading['mismatched_keys'] or loading['error_msgs'] or any(not key.startswith('mtp.') for key in loading['ignored_checkpoint_only_MTP_keys']):
            raise ValueError('Original native Qwen checkpoint did not load cleanly')
        audit_runtime_metadata(files.json('HARDWARE_PREFLIGHT.json'), context)
        from .composite_frozen_base_eval import selected_detector
        store = FrozenInputs(transport, context['pin']['transport_manifest_sha256'])
        saved = selected_detector(store, read(source/'config/composite_controller_rank32_v2.json'))
        report['parameters'] = parameter_audit(files, context, package, saved)
        model_files = files.json('MODEL_FILE_VERIFICATION.json'); metadata = files.json('MODEL_METADATA_BYTES.json')
        if loaded['model_files']!=model_files:raise ValueError('The native load consumed another model-file verification')
        validate_download_proof(context, model_files); verify_metadata_bytes(context['model_pin'], model_files, metadata)
        with verified_tokenizers(files,context,package,source) as (bt,qt,eos):
            records, replayed = replay(files, context, package, (bt,qt), eos)
            operational=None
            if context.get('operational_profile') is not None:
                from .composite_frozen_operational_audit import audit_operational_evidence
                operational=audit_operational_evidence(files,context,package,records,replayed,store,qt,eos)
        expected_forwards=replayed['expected_joint_forwards'] if operational is None else operational['expected_current_run_forwards']
        if expected_forwards != counts['completed']:
            raise ValueError('Unaccounted or missing model forwards in the complete base trajectories')
        from .composite_metrics import summarize
        from .composite_world import Scene
        metrics = summarize(records,[Scene(**record['scene']) for record in records],['base'],
                            read(source/'config/composite_controller_rank32_v2.json'),None)
        metrics['exact_off_bypass_pass'] = None  # No behavior adapter exists in this package.
        report.update(complete_base_partition_audited=True,replay=replayed,original_base_metrics=metrics)
        if operational is not None:report['operational_version_replay']=operational
    return report


def aggregate(closeouts, source, commit, transport):
    """All36 base trajectories, one frozen source; all other controls stay unknown."""
    from .composite_frozen_base import partition_plan
    from .composite_metrics import summarize
    from .composite_world import Scene
    source=Path(source);cfg=read(source/'config/composite_controller_rank32_v2.json')
    context=authorized_load_plan(source,commit,cfg['qwen'],transport)
    if len(closeouts)!=6:raise ValueError('Six fully audited/released original-base partitions are required')
    partitions={};records=[]
    for closeout_file in closeouts:
        closeout=read(closeout_file)
        if closeout.get('owned_compute_released') is not True or closeout.get('complete_base_partition_audited') is not True:
            raise ValueError('An incomplete or unreleased base partition cannot enter complete validation')
        audit_file=Path(closeout['audit_receipt']);proof=read(audit_file)
        archive=Path(closeout['archive'])
        if (sha(audit_file)!=closeout['audit_sha256'] or sha(archive)!=closeout['archive_sha256']
            or proof['archive_sha256']!=closeout['archive_sha256'] or proof['source_commit']!=commit
            or proof['complete_base_partition_audited'] is not True
            or proof['transport_manifest_sha256']!=context['pin']['transport_manifest_sha256']):
            raise ValueError('Base closeout/audit/source/input freeze differs')
        number=proof['partition']
        if type(number) is not int or not 0<=number<6 or number in partitions:
            raise ValueError('A duplicate or foreign base partition cannot replace missing cases')
        expected=partition_plan(context['scenes'],number)
        if proof['case_ids']!=expected['case_ids']:raise ValueError('A base partition omits or replaces original scenarios')
        verify_evidence_archive(archive.read_bytes())
        with zipfile.ZipFile(archive) as stream:
            values=[json.loads(stream.read('trajectories/'+key+'.json')) for key in expected['case_ids']]
        if any(value['condition']!='base' for value in values):raise ValueError('An intervention entered a frozen-base aggregate')
        partitions[number]={'audit_sha256':closeout['audit_sha256'],'archive_sha256':closeout['archive_sha256'],'records':values}
    for number in range(6):records.extend(partitions[number]['records'])
    if [record['scene'] for record in records]!=context['scenes']:raise ValueError('Complete original36-scene roster differs')
    metrics=summarize(records,[Scene(**scene) for scene in context['scenes']],['base'],cfg,None)
    metrics['exact_off_bypass_pass']=None
    report=base_report(records,[scene['case_id'] for scene in context['scenes']])
    report.update(original_base_validation_complete=True,source_commit=commit,
        transport_manifest_sha256=context['pin']['transport_manifest_sha256'],original_base_metrics=metrics,
        partition_evidence={str(number):{key:value for key,value in item.items() if key!='records'} for number,item in partitions.items()},
        other_required_trajectories=144,audit_model_forwards=0,audit_optimizer_updates=0,
        full_system_validation_complete=False,full_validation_admitted=False)
    return report

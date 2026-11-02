"""Offline byte/token replay for frozen qualification; never constructs a model."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile
from .io_utils import read,sha
from .composite_exports import verify_evidence_archive
from .composite_runtime import FrozenInputs
from .composite_frozen_probe import (authorized_load_plan,load_training_rows,audit_probe_record,
    validate_download_proof,verify_metadata_bytes,verify_software,select_training_probes,MODE,BOUNDS,CONDITIONS)
from .composite_frozen_worker import DISK_POLICY


def reject_network(event,args):
    if event in {'socket.connect','socket.getaddrinfo','socket.sendto'}:
        raise RuntimeError('Network access is forbidden in the frozen raw-evidence audit')


def audit_token_replay(record,rows,context):
    """Reconstruct verified metadata in a disposable directory; no weight files."""
    files=record['model_file_verification'];encoded=record['model_metadata_bytes']
    validate_download_proof(context,files)
    verify_metadata_bytes(context['model_pin'],files,encoded)
    # All metadata names came from the complete original reviewed model pin.
    environment={'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
                 'HF_HUB_DISABLE_TELEMETRY':'1','CUDA_VISIBLE_DEVICES':''}
    before={key:os.environ.get(key) for key in environment}
    try:
        os.environ.update(environment)
        with tempfile.TemporaryDirectory(prefix='research4_metadata_audit_') as directory:
            root=Path(directory)
            for name,value in encoded.items():
                target=(root/name).resolve()
                if not target.is_relative_to(root) or target.suffix=='.safetensors':
                    raise ValueError('Only registered small model metadata may be reconstructed')
                target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as stream:stream.write(base64.b64decode(value,validate=True))
            from transformers import AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained(root,local_files_only=True,trust_remote_code=False)
            generation=read(root/'generation_config.json');eos=set()
            for value in [generation.get('eos_token_id'),tokenizer.eos_token_id]:
                if value is not None:eos.update(value if isinstance(value,list) else [value])
            return audit_probe_record(record,rows,context,tokenizer,eos)
    finally:
        for key,value in before.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value


def bounded_counts(counts):
    if set(counts)!={'attempted','completed'} or any(type(value) is not int for value in counts.values()):
        raise ValueError('Exact counts must be explicit integers')
    if not 0<=counts['completed']<=counts['attempted']<=BOUNDS['maximum_backbone_forwards']:
        raise ValueError('Frozen forward bounds differ')


def audit_runtime_metadata(hardware,context):
    verify_software(hardware['software']['torch'],hardware['software']['packages'])
    devices=hardware['devices']
    if len(devices)!=2 or [value['index'] for value in devices]!=[0,1]:
        raise ValueError('Both declared T4 devices require runtime evidence')
    for value,lower in zip(devices,context['metadata_bounds']['device_weight_lower_bounds_bytes']):
        if value['name']!='Tesla T4' or not lower+2*1024**3<=value['free_bytes']<=value['total_bytes']:
            raise ValueError('Runtime device preflight differs from the complete frozen profile')


def audit_archive(archive,source,commit,transport,controller,export=None):
    source=Path(source);context=authorized_load_plan(source,commit,
        read(source/'config/composite_controller_rank32_v2.json')['qwen'])
    raw=Path(archive).read_bytes()
    # Check expansion before CRC/decompression; retain all bytes on rejection.
    with zipfile.ZipFile(io.BytesIO(raw)) as stream:
        if sum(value.file_size for value in stream.infolist())>DISK_POLICY['maximum_raw_evidence_bytes']:
            raise ValueError('Evidence expansion exceeds the frozen raw bound')
    integrity=verify_evidence_archive(raw)
    if controller.get('source_commit')!=commit or controller.get('mode')!=MODE or controller.get('notebook')!=context['pin']['notebook']:
        raise ValueError('Controller identity differs from this exact frozen stage')
    evidence=controller.get('evidence',{})
    if evidence.get('archive_sha256')!=hashlib.sha256(raw).hexdigest():
        raise ValueError('Controller did not seal this exact archive')
    state=controller['child_state']
    if state.get('child_exit_confirmed') is not True:
        raise ValueError('An unconfirmed live worker cannot receive a terminal audit')
    counts=controller['counts']
    if counts['counts_exact_at_exit'] is True:bounded_counts(counts['forward_counts'])
    elif counts['forward_counts']!={'attempted':None,'completed':None}:
        raise ValueError('Unknown final counts must remain unknown')
    with zipfile.ZipFile(io.BytesIO(raw)) as stream:
        names=set(stream.namelist())
        def value(name):return json.loads(stream.read(name))
        freeze=value('PRECALL_FREEZE.json')
        if freeze!={'source_commit':commit,'mode':MODE,'pin':context['pin'],
                    'input_manifest_sha256':sha(source/'inputs/MANIFEST.json'),
                    'bounds':BOUNDS,'disk_policy':DISK_POLICY}:
            raise ValueError('Raw source/input/execution freeze differs')
        receipt=value('RUN_RECEIPT.json') if 'RUN_RECEIPT.json' in names else None
        report={'status':'partial_frozen_evidence_integrity_audited','source_commit':commit,
            'archive_sha256':hashlib.sha256(raw).hexdigest(),**integrity,
            'worker_status':receipt['status'] if receipt else 'unknown_abrupt_exit',
            'counts_exact_at_exit':counts['counts_exact_at_exit'],'forward_counts':counts['forward_counts'],
            'audit_model_forwards':0,'audit_optimizer_updates':0,'heldout_content_read':False,
            'complete_qualification_audited':False,'full_validation_admitted':False,
            'heldout_confirmation':'not_performed','cross_device_numerical_parity':'not_established'}
        original=evidence.get('evidence_kind')=='original_worker_export'
        if original:
            if export is None or export.get('archive_sha256')!=report['archive_sha256'] or export.get('archive_bytes')!=len(raw):
                raise ValueError('Original worker export SHA/length differs')
            if receipt is None or any(export.get(key)!=entry for key,entry in receipt.items()):
                raise ValueError('Worker/export terminal receipts differ')
        elif evidence.get('evidence_kind')=='controller_post_exit_rescue':
            observed=value('CONTROLLER_TERMINAL_OBSERVATION.json')
            for key in ['source_commit','mode','notebook','child_state','counts']:
                if observed.get(key)!=controller.get(key):raise ValueError('Rescue observation changed')
        else:raise ValueError('Unknown terminal evidence provenance')
        if receipt is not None:
            if receipt.get('source_commit')!=commit or receipt.get('stage')!=MODE or receipt.get('heldout_content_read') is not False:
                raise ValueError('Worker terminal scope differs')
            if receipt.get('optimizer_updates')!=0 or receipt.get('behavior_adapter_fitting') is not False:
                raise ValueError('Frozen worker executed outside its scope')
        complete=bool(original and receipt and receipt['status']=='complete'
            and state.get('exit_code')==0 and controller.get('complete_collection_exported') is True
            and controller.get('cleanup',{}).get('owned_cache_removed') is True)
        if complete:
            if counts['counts_exact_at_exit'] is not True or receipt['forward_counts']!=counts['forward_counts']:
                raise ValueError('Complete qualification lacks exact terminal hook counts')
            journal=value('FORWARD_COUNTS.json')
            if journal.get('sealed_for_exit') is not True or any(journal[key]!=counts['forward_counts'][key] for key in ['attempted','completed']):
                raise ValueError('Complete terminal counter journal differs')
            record=value('FROZEN_QUALIFICATION_RECORD.json')
            if record['forward_counts']!=receipt['forward_counts']:raise ValueError('Raw collection counts differ')
            for index,probe in enumerate(record['probes']):
                if value('FROZEN_PROBE_'+str(index)+'.json')!=probe:raise ValueError('Raw probe and complete record differ')
                prefill=value('FROZEN_PREFILL_'+str(index)+'.json')
                if any(probe[key]!=entry for key,entry in prefill.items()):raise ValueError('Durable prefill and raw probe differ')
            for filename,key in [('FROZEN_PARAMETERS_BEFORE.json','parameters_before'),
                                 ('FROZEN_PARAMETERS_AFTER.json','parameters_after'),
                                 ('MODEL_FILE_VERIFICATION.json','model_file_verification'),
                                 ('MODEL_METADATA_BYTES.json','model_metadata_bytes')]:
                if value(filename)!=record[key]:raise ValueError('Complete record changed durable raw '+key)
            for name in ['HARDWARE_PREFLIGHT.json','FROZEN_LOAD_FREEZE.json','FROZEN_LOADED_LAYOUT.json']:
                hardware=value(name)
                audit_runtime_metadata(hardware if name=='HARDWARE_PREFLIGHT.json' else hardware['hardware_preflight'],context)
            loaded=value('FROZEN_LOADED_LAYOUT.json')
            for key,target in [('loaded_layout','loaded_layout'),('loading_info','loading_info'),('model_files','model_file_verification')]:
                if loaded[key]!=record[target]:raise ValueError('Durable native loading proof differs')
            rows,provenance=load_training_rows(context,FrozenInputs(transport,context['pin']['transport_manifest_sha256']))
            if record['data_provenance']!=provenance:raise ValueError('Raw input provenance differs')
            selected=select_training_probes(rows)
            expected_selection={'conditions':CONDITIONS,'state_ids':[row['state_id'] for row in selected],
                'prompt_sha256':[hashlib.sha256(row['text'].encode()).hexdigest() for row in selected],
                'provenance':provenance,'selected_before_forwards':True,'heldout_content_read':False}
            if value('FROZEN_PROBE_SELECTION.json')!=expected_selection:
                raise ValueError('Durable predeclared training selection differs')
            report['raw_replay']=audit_token_replay(record,rows,context)
            report.update(status='complete_frozen_qualification_raw_evidence_audited',complete_qualification_audited=True)
        return report

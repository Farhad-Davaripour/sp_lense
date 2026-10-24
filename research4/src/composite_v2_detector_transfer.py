"""Fixed-detector prediction collection and read-only independent evidence.

No fitting, threshold selection, retry, Qwen call, provider submission or seal.
Production collection/audit requires the Databricks runtime. Public fakes never
create a scientific transfer attestation.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import base64,tempfile
from .io_utils import digest,read
from .composite_audit import audit_prediction
from .composite_memory import Prediction
from .composite_training import evaluate_detector

READER='v2-fixed-detector-transfer-audit-v1'
_KEY=object()


class DetectorTokenizer:
    def __init__(self,tokenizer,key=None):
        if key is not _KEY:raise PermissionError('Verify the original detector metadata bytes')
        self.tokenizer=tokenizer;self._key=key


class TransferEvidence:
    def __init__(self,receipt,key=None):
        if key is not _KEY:raise PermissionError('Re-audit complete fixed-detector predictions')
        self._receipt=deepcopy(receipt);self._key=key
    def receipt(self):return deepcopy(self._receipt)


@contextmanager
def verified_detector_tokenizer(files,root):
    from .composite_frozen_probe import verify_metadata_bytes
    pin=read(Path(root)/'inputs/MODERNBERT_PIN.json')
    proof=files.json('DETECTOR_FILE_VERIFICATION.json');encoded=files.json('DETECTOR_METADATA_BYTES.json')
    verify_metadata_bytes(pin,proof,encoded)
    by_name={v['name']:v for v in proof}
    if len(by_name)!=len(proof) or set(by_name)!={v['name'] for v in pin['files']}:
        raise PermissionError('Complete detector download proof required')
    for item in pin['files']:
        actual=by_name[item['name']]
        if actual['size']!=item['size'] or (item.get('sha256') and actual['sha256']!=item['sha256']):
            raise PermissionError('Original detector file pin changed')
    with tempfile.TemporaryDirectory(prefix='r4_v2_detector_tokenizer_') as folder:
        target=Path(folder)
        for name,value in encoded.items():
            path=(target/name).resolve()
            if not path.is_relative_to(target.resolve()):raise PermissionError('Metadata path escaped')
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(base64.b64decode(value,validate=True))
        from transformers import AutoTokenizer
        yield DetectorTokenizer(AutoTokenizer.from_pretrained(target,local_files_only=True,trust_remote_code=False),_KEY)


def collect_transfer(detector,rows,guard,persist,*,public_fixture=False):
    if not public_fixture:
        from .composite_generalized_data import require_runtime
        require_runtime()
    if not rows or len(rows)>900 or len({r['state_id'] for r in rows})!=len(rows):
        raise PermissionError('Every unique complete development state required within the fixed bound')
    if any(r.get('split')!='development' for r in rows):raise PermissionError('Only new development reference states')
    predictions=[]
    try:
        for row in rows:
            guard();prediction=detector.predict(row['text'],0.9)
            record={'state_id':row['state_id'],'case_id':row['case_id'],'prediction':asdict(prediction)}
            persist('detector_predictions/'+row['state_id']+'.json',record);predictions.append(record)
        return predictions
    finally:
        persist('TRANSFER_COLLECTION_RECEIPT.json',{'expected_states':len(rows),'observed_states':len(predictions),
            'missing_state_ids':sorted({r['state_id'] for r in rows}-{r['state_id'] for r in predictions}),
            'threshold':0.9,'optimizer_updates':0,'public_fixture_test_mode':public_fixture,
            'scientific_transfer_admitted':False,'heldout_content_read':False})


def transfer_report(records,rows,cfg,source_commit,data_release_sha256,checkpoint_sha256,
                    tokenizer_evidence=None,*,public_fixture=False,fixture_tokenizer=None):
    if fixture_tokenizer is not None and (not public_fixture or tokenizer_evidence is not None):
        raise PermissionError('Injected tokenizer is public-fixture only')
    verified=isinstance(tokenizer_evidence,DetectorTokenizer) and tokenizer_evidence._key is _KEY
    if tokenizer_evidence is not None and not verified:raise PermissionError('Opaque tokenizer proof required')
    tokenizer=tokenizer_evidence.tokenizer if verified else fixture_tokenizer
    expected={r['state_id']:r for r in rows};lookup={r['state_id']:r for r in records}
    if len(expected)!=len(rows) or len(lookup)!=len(records) or not set(lookup)<=set(expected):
        raise ValueError('Duplicate or foreign detector states')
    missing=sorted(set(expected)-set(lookup));complete=not missing and bool(rows)
    decoded=[]
    for row in rows:
        if row['state_id'] not in lookup:continue
        record=lookup[row['state_id']];prediction=record['prediction']
        if record['case_id']!=row['case_id']:raise ValueError('Prediction crossed its case')
        if tokenizer is not None:
            encoded=tokenizer(row['text'],add_special_tokens=True,return_offsets_mapping=True,truncation=False)
            if len(encoded['input_ids'])>2048:raise PermissionError('Complete detector input exceeds its frozen limit')
            audit_prediction(prediction,row['text'],0.9)
            if (digest(encoded['input_ids'])!=digest(prediction['raw']['input_token_ids'])
                or digest([list(v) for v in encoded['offset_mapping']])!=digest(prediction['raw']['offsets'])):
                raise ValueError('Complete detector token/pointer provenance differs')
        decoded.append(Prediction(**prediction))
    reports=None
    # Undefined positive-class denominators stay unknown, never perfect/zero.
    enough=complete and any(r['labels']['gate'] for r in rows)
    for key in ['template','mechanism']:
        enough=enough and all(any(r['labels']['gate'] for r in rows if r[key]==value) for value in {r[key] for r in rows})
    enough=enough and all(any(r['labels']['gate'] for r in rows if (r['mechanism'],r['template'])==pair)
                         for pair in {(r['mechanism'],r['template']) for r in rows})
    if enough:
        # Keep original style-wise criteria, and apply the same criteria inside
        # each new mechanism. No fitting or threshold tuning occurs here.
        overall=evaluate_detector(None,rows,0.9,cfg,lambda:None,predictions=decoded)
        mechanisms={}
        for family in sorted({r['mechanism'] for r in rows}):
            indices=[i for i,r in enumerate(rows) if r['mechanism']==family]
            mechanisms[family]=evaluate_detector(None,[rows[i] for i in indices],0.9,cfg,lambda:None,
                                                 predictions=[decoded[i] for i in indices])
        reports={'overall':overall,'mechanisms':mechanisms}
    admitted=bool(enough and verified and not public_fixture and reports['overall']['admitted']
                  and all(v['admitted'] for v in reports['mechanisms'].values()))
    return {'reader':READER,'source_commit':source_commit,'data_release_sha256':data_release_sha256,
        'checkpoint_sha256':checkpoint_sha256,'threshold':0.9,'complete':complete,'missing_state_ids':missing,
        'prediction_manifest':{k:digest(v) for k,v in lookup.items()},'reports':reports,'metrics_defined':bool(enough),
        'raw_predictions_replayed':tokenizer is not None,'raw_evidence_audited':complete and verified and not public_fixture,
        'admitted':admitted,'public_fixture_test_mode':public_fixture,'model_integrity_audited':False,
        'optimizer_updates':0,'model_forwards':0,'heldout_content_read':False}


def read_transfer_archive(raw,sha256,root,commit,data_archive,checkpoint,*,closure=None):
    from .composite_v2_development_contract import validate_stage
    from .composite_v2_development_data import load_development_data
    from .composite_v2_model_audit import EvidenceFiles,ProviderClosure,detector_integrity
    context=validate_stage(root,commit)
    if context['stage']['phase']!='detector_transfer':raise PermissionError('Exact transfer phase required')
    data=load_development_data(data_archive,context,reference_rows=True);rows=data['rows']
    files=EvidenceFiles(raw,sha256,context['bounds']['maximum_raw_evidence_bytes'])
    try:
        receipt=files.json('RUN_RECEIPT.json')
        if (receipt.get('status')!='complete' or receipt.get('public_fixture_test_mode') is not False
            or receipt.get('source_commit')!=commit or type(receipt.get('optimizer_updates')) is not int or receipt['optimizer_updates']!=0):
            raise PermissionError('A complete genuine frozen transfer run is required')
        if (not isinstance(closure,ProviderClosure) or closure.receipt['host']!=context['workspace']['host']
            or closure.receipt['run_id']!=receipt.get('provider_run_id') or closure.receipt['raw_archive_sha256']!=sha256):
            raise PermissionError('Exact independent detector-transfer provider closure required')
        if receipt.get('counts_exact_at_controlled_exit') is not True or receipt.get('forward_counts')!={'attempted':len(rows),'completed':len(rows)}:
            raise PermissionError('Every exact reference state needs one recorded detector forward')
        from .composite_frozen_probe import verify_software
        hardware=files.json('HARDWARE_PREFLIGHT.json');verify_software(hardware['torch'],hardware['packages'])
        if hardware.get('visible_GPUs')!=1 or hardware.get('BF16_supported') is not True or hardware.get('TF32_enabled') is not False:
            raise PermissionError('Fixed detector runtime profile differs')
        expected={'source_commit':commit,'stage':context['stage'],'workspace':context['workspace'],
                  'data_release_sha256':digest(data['release'])}
        if digest(files.json('DEVELOPMENT_PRECALL_FREEZE.json'))!=digest(expected):raise PermissionError('Transfer source/data/workspace changed')
        paths={'detector_predictions/'+row['state_id']+'.json' for row in rows}
        if {name for name in files.manifest if name.startswith('detector_predictions/')}!=paths:
            raise PermissionError('Missing/extra fixed-detector state outputs')
        records=[files.json('detector_predictions/'+row['state_id']+'.json') for row in rows]
        detector_integrity(files,read(Path(root)/'inputs/MODERNBERT_PIN.json'),checkpoint,context['stage']['detector_checkpoint_sha256'])
        with verified_detector_tokenizer(files,root) as tokenizer:
            report=transfer_report(records,rows,context['configuration'],commit,context['stage']['data_release_sha256'],
                context['stage']['detector_checkpoint_sha256'],tokenizer)
        report.update(model_integrity_audited=True,raw_archive_sha256=sha256,owned_compute_released=True)
        return TransferEvidence(report,_KEY)
    finally:files.close()


def bound_transfer(value,source,package):
    if not isinstance(value,TransferEvidence) or value._key is not _KEY:return None
    report=value.receipt()
    if (report['source_commit']!=source or report['data_release_sha256']!=package.get('data_release_sha256')
        or report['public_fixture_test_mode'] is not False or report['model_integrity_audited'] is not True):return None
    return report

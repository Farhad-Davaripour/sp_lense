"""Independent partition/scoring integration; no model or provider execution."""
from copy import deepcopy
from pathlib import Path
import hashlib
from .io_utils import digest,read
from .composite_v2_model_audit import (EvidenceFiles,ProviderClosure,read_evaluation_integrity,
    combine_evaluation_integrity)
from .composite_v2_development_contract import validate_stage
from .composite_v2_development_data import load_development_data
from .composite_v2_detector_transfer import TransferEvidence,bound_transfer

_KEY=object()


class PartitionEvidence:
    def __init__(self,receipt,records,raw,helpers,integrity,key=None):
        if key is not _KEY:raise PermissionError('Re-audit exact native partition evidence')
        self.receipt=deepcopy(receipt);self.records=deepcopy(records);self.raw=raw;self.helpers=helpers;self.integrity=integrity;self._key=key


class DevelopmentAttestation:
    def __init__(self,receipt,key=None):
        if key is not _KEY:raise PermissionError('Combine exact complete partition audits')
        self._receipt=deepcopy(receipt);self._key=key
    def receipt(self):return deepcopy(self._receipt)


def audit_partition_archive(raw,sha256,root,commit,data_archive,package,model_stage_evidence,exposure,checkpoint,closure=None):
    context=validate_stage(root,commit);stage=context['stage']
    if stage['phase']!='development_partition':raise PermissionError('Exact native development partition required')
    if digest(package)!=stage['package']['sha256']:raise PermissionError('Model package changed')
    data=load_development_data(data_archive,context)
    files=EvidenceFiles(raw,sha256,context['bounds']['maximum_raw_evidence_bytes'])
    try:
        run=files.json('RUN_RECEIPT.json')
        if (run.get('status')!='complete' or run.get('public_fixture_test_mode') is not False
            or run.get('source_commit')!=commit or run.get('phase')!='development_partition'
            or type(run.get('optimizer_updates')) is not int or run['optimizer_updates']!=0
            or run.get('counts_exact_at_controlled_exit') is not True):raise PermissionError('Incomplete/fake/foreign validation worker')
        if (not isinstance(closure,ProviderClosure) or closure.receipt['host']!=context['workspace']['host']
            or closure.receipt['run_id']!=run.get('provider_run_id') or closure.receipt['raw_archive_sha256']!=sha256):
            raise PermissionError('Exact independent provider terminal closure required')
        expected={'source_commit':commit,'stage':stage,'workspace':context['workspace'],'data_release_sha256':digest(data['release'])}
        if digest(files.json('DEVELOPMENT_PRECALL_FREEZE.json'))!=digest(expected):raise PermissionError('Native partition freeze changed')
        ids=set(stage['partition']['case_ids'])
        from .composite_generalized_world import GeneralizedScene
        scenes=[GeneralizedScene(**s) for s in data['scenes'] if s['case_id'] in ids]
        records=[files.json('traces/'+scene.case_id+'/'+arm+'.json') for scene in scenes for arm in stage['partition']['controls']]
        wanted={'traces/'+s.case_id+'/'+a+'.json' for s in scenes for a in stage['partition']['controls']}
        if {n for n in files.manifest if n.startswith('traces/')}!=wanted:raise PermissionError('Missing/extra partition case/control traces')
        from .composite_v2_trace_audit import verified_trace_tokenizers,audit_partition
        from .composite_v2_helper_audit import read_helper_evidence
        with verified_trace_tokenizers(files,{'metadata':{'tensors':read(Path(root)/'inputs/QWEN_TENSOR_METADATA.json')['tensors']},
                'model_pin':read(Path(root)/'inputs/QWEN_PIN.json')},
                {'detector_files':files.json('DETECTOR_FILE_VERIFICATION.json')},root) as tokens:
            native=audit_partition(records,scenes,commit,tokens,conditions=stage['partition']['controls'])
            lookup={s.case_id:s for s in scenes}
            helpers={r['scene']['case_id']+'/'+r['condition']:read_helper_evidence(r,lookup[r['scene']['case_id']],commit,
                context['design'],exposure,tokens) for r in records}
        manifest={r['scene']['case_id']+'/'+r['condition']:digest(r) for r in records}
        integrity=read_evaluation_integrity(files,root,commit,package,manifest,model_stage_evidence,detector_checkpoint=checkpoint)
        forwards=0
        for record in records:
            observations=[record['initial_observation']]+[s['observation_after'] for s in record['trace']]
            forwards+=sum(o['memory']['prediction']['raw'].get('context_budget_failure') is not True for o in observations)
            forwards+=sum(len(s['response']['generated_token_ids']) for s in record['trace'])
        if run.get('forward_counts')!={'attempted':forwards,'completed':forwards}:raise PermissionError('Exact joint forward ledger differs from raw native records')
        if forwards>stage['partition']['maximum_joint_backbone_forwards']:raise PermissionError('Partition exceeded its proven complete forward bound')
        return PartitionEvidence({'source_commit':commit,'package_sha256':digest(package),'data_release_sha256':stage['data_release_sha256'],
            'partition_index':stage['partition_index'],'trace_manifest':manifest,'raw_archive_sha256':sha256,
            'raw_evidence_audited':native['raw_evidence_audited'],'runtime_environment_enforced':True,
            'owned_compute_released':True,'heldout_content_read':False},records,native['raw_audits'],helpers,integrity,_KEY)
    finally:files.close()


def combine_partitions(values,source_commit,package,plan,transfer):
    expected={row['case_id']+'/'+arm for row in package['development_roster'] for arm in plan['partitions'][0]['controls']}
    seen=set();indices=set();records=[];raw={};helpers={};integrities=[]
    for value in values:
        if not isinstance(value,PartitionEvidence) or value._key is not _KEY:raise PermissionError('Opaque partition reader proofs required')
        report=value.receipt
        if (report.get('raw_evidence_audited') is not True or report.get('owned_compute_released') is not True
            or report.get('runtime_environment_enforced') is not True):raise PermissionError('Every supplied partition must be independently complete/released')
        if report['source_commit']!=source_commit or report['package_sha256']!=digest(package) or report['data_release_sha256']!=package['data_release_sha256']:
            raise PermissionError('Partitions belong to another exact package/source/data')
        index=report['partition_index']
        if index in indices or not 0<=index<len(plan['partitions']):raise PermissionError('Duplicate/foreign partition')
        actual=set(report['trace_manifest']);part=plan['partitions'][index]
        if actual!={c+'/'+a for c in part['case_ids'] for a in part['controls']} or seen&actual or not actual<=expected:
            raise PermissionError('Partition source/coverage overlap or mismatch')
        indices.add(index);seen.update(actual);records.extend(value.records);raw.update(value.raw);helpers.update(value.helpers);integrities.append(value.integrity)
    missing=sorted(expected-seen);manifest={r['scene']['case_id']+'/'+r['condition']:digest(r) for r in records}
    integrity=None;attestation=None
    if not missing and len(indices)==len(plan['partitions']):
        integrity=combine_evaluation_integrity(integrities,source_commit,package,manifest)
        detector=bound_transfer(transfer,source_commit,package)
        if detector is not None:
            attestation=DevelopmentAttestation({'source_commit':source_commit,'package_sha256':digest(package),
                'data_release_sha256':package['data_release_sha256'],'trace_manifest_sha256':digest(manifest),
                'detector_transfer':detector,'runtime_environment_enforced':True},_KEY)
    return {'records':records,'raw_audits':raw,'helper_audits':helpers,'model_integrity':integrity,
        'attestation':attestation,'missing':missing,'complete':not missing and len(indices)==len(plan['partitions']),
        'scientific_validation_admitted':False}


def bound_attestation(value,source,package_sha,data_sha,manifest):
    if not isinstance(value,DevelopmentAttestation) or value._key is not _KEY:return {}
    receipt=value.receipt()
    if (receipt['source_commit']!=source or receipt['package_sha256']!=package_sha
        or receipt['data_release_sha256']!=data_sha or receipt['trace_manifest_sha256']!=digest(manifest)):return {}
    return receipt


def score_development(values,scenes,source_commit,package,cfg,design,plan,transfer,model_stage_evidence):
    """Assemble all controls through the original scorer; never open a seal."""
    from .composite_v2_model_audit import ModelEvidence
    from .composite_v2_evaluation import summarize_v2
    combined=combine_partitions(values,source_commit,package,plan,transfer)
    bypass=False
    if isinstance(model_stage_evidence,ModelEvidence):
        proof=model_stage_evidence.receipt()
        bypass=(proof.get('full_fit_raw_audited') is True and proof.get('source_commit')==source_commit
            and proof.get('checkpoint_sha256')==package.get('checkpoint_sha256')
            and proof.get('data_release_sha256')==package.get('data_release_sha256')
            and digest(proof)==package.get('training_raw_audit_sha256'))
    report=summarize_v2(combined['records'],scenes,cfg,design,package,source_commit,bypass,
        raw_audits=combined['raw_audits'],helper_audits=combined['helper_audits'],
        model_integrity_audited=combined['model_integrity'],stage_attestation=combined['attestation'])
    report.update(development_partition_plan_sha256=digest(plan),observed_partition_count=len(values),
                  expected_partition_count=len(plan['partitions']),absence_of_unlogged_calls_proved=False)
    return report

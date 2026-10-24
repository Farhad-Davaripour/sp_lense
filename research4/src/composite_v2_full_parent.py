"""Read/audit an existing full fit and its scratch parent; no model forward."""
import re
from pathlib import Path
from .io_utils import read,digest
from .composite_v2_parent_audit import extract_source,qualification_from_descriptor
from .composite_v2_model_audit import audit_model_archive,provider_closure


def validate_full_fit_descriptor(stage,workspace):
    parent=stage.get('full_fit_parent',{});commit=stage['source_commit']
    if parent.get('source_commit')!=commit or parent.get('workspace')!=workspace or parent.get('data_release_sha256')!=stage['data_release_sha256']:
        raise PermissionError('Exact common full-fit source/data/workspace required')
    expected={'source_path':'model_sources_v2/'+commit+'/full_fit/V2_MODEL_SOURCE.zip',
              'raw_path':'model_generalized_v2/'+commit+'/full_fit_01/evidence.zip',
              'AIR_response_path':'model_generalized_v2/'+commit+'/full_fit_01/AIR_TERMINAL_RESPONSE.json',
              'Jobs_response_path':'model_generalized_v2/'+commit+'/full_fit_01/JOBS_TERMINAL_RESPONSE.json'}
    if any(parent.get(k)!=v for k,v in expected.items()):raise PermissionError('Exact immutable full-fit locators required')
    for key in ['source_sha256','raw_sha256']:
        if not isinstance(parent.get(key),str) or not re.fullmatch('[0-9a-f]{64}',parent[key]):raise PermissionError('Complete parent archive hashes required')
    if type(parent.get('provider_run_id')) is not int or parent['provider_run_id']<=0:raise PermissionError('Exact full-fit provider ID required')
    return parent


def full_fit_evidence(stage,workspace,store,scratch,data_archive):
    parent=validate_full_fit_descriptor(stage,workspace);commit=stage['source_commit']
    closure=provider_closure(workspace['host'],parent.get('provider_run_id'),parent['raw_sha256'],
                            store.json(parent['AIR_response_path']),store.json(parent['Jobs_response_path']))
    root=extract_source(store.get(parent['source_path']),parent['source_sha256'],Path(scratch)/'full_fit_source')
    from .composite_v2_model_contract import validate_model_metadata
    context=validate_model_metadata(root,commit)
    qualification=qualification_from_descriptor(context['stage'],workspace,store,Path(scratch)/'scratch_parent',data_archive)
    proof=audit_model_archive(store.get(parent['raw_path']),parent['raw_sha256'],root,commit,data_archive,
                              closure=closure,qualification_evidence=qualification)
    report=proof.receipt()
    if report.get('full_fit_raw_audited') is not True or report.get('owned_compute_released') is not True:
        raise PermissionError('Public/incomplete/unreleased full fit cannot enter development')
    return proof

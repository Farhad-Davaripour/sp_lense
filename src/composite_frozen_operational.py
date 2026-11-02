"""Explicit operational-version lineage for frozen BASE memory recovery."""
import base64
import hashlib
import json
import re
from pathlib import Path
from .io_utils import read,sha
from .composite_cuda_maintenance import PROFILE as MAINTENANCE_PROFILE
from .composite_frozen_continuation import ORIGINAL_SOURCE,PARENT_RUN,PARENT_ARCHIVE_SHA256,PARENT_COMMON_FREEZE

PROFILE='frozen-base-unused-cache-operational-v2'
DOCUMENT='FROZEN_BASE_OPERATIONAL_V2.json'
PARENT_PREFIX_AUDIT='FAILED_BASE_PREFIX_AUDIT.json'
PARENT_CLOSEOUT='FAILED_BASE_CLOSEOUT.json'
PARENT_RELEASE='ORIGINAL_BASE_RELEASE.json'
PARENT_TRANSPORT_SHA256='3e1bd34b47e6036c2aa5ddccf7d6689f1369908bd67647db33b0a09f9044c2d5'
PARENT_PREFIX_AUDIT_SHA256='586a0d2b5c3f4876eb557f9ab8a8b5c77c8664eb5972199afb6dcb50aaf512d3'
PARENT_CLOSEOUT_SHA256='4b04fef0dc8a8fdb652c3ce4870277f56092d77d72b663d798d42af34d865c2b'
PARENT_RELEASE_SHA256='4e80d9b85e633dca42a4a52ce7deec2731d46451a1e303f262be5a9903f59aa1'
PARENT_ARCHIVE='frozen_base_parent_v1/361586756/evidence.zip'
RELATIONSHIP='versioned_operational_change_not_identical_common_freeze'
CORE_FILES={
    'src/composite_actor.py':'21dc6716b65f559da4c639c0927dffe52ee6f2d95b3d6c0725509cc742355c1f',
    'src/composite_detector.py':'f1c2af2b0e696b02663e20e79a63bcb5ca817af04d8f833624e5c26691cc22f1',
    'src/composite_world.py':'83e1ce58673ba045284089931463c295e142117e2b91b2ec54eb4e420c856599',
    'src/composite_environment.py':'0b6796f8fa5ca536e3dc9120a26827b970b9335c30606388000151590873da87',
    'src/composite_interface.py':'75f9cdef7556380ec2783b441d89e800373b81924d01b8bcf9b6de50078e3286',
    'src/composite_kernels.py':'9488363d822f713e56a1a73d0efe4bd9bf6e3a5c09f99614b9c08ebb480bc20f',
    'src/composite_memory.py':'0d96a397b4c821ed0af21e5709c0a120ee38e2cca297dd04434f2e47c9401ffc',
    'src/native27_tools.py':'015934bfce3f212e759971f39552d708e12b9f79d06f247f9daaeb5546dc77f5',
    'src/composite_lora.py':'abbc6aff700959174b2fce32d89a246337ab78cf628cf6e259a16073ca5f1950',
    'config/composite_controller_rank32_v2.json':'676cb94471da5b2f79a12716b82db566875292a38e54b794d76a4bb945300dd3',
}


def document(commit,partition):
    if (type(partition) is not int or partition not in [3,4,5] or commit==ORIGINAL_SOURCE
        or not isinstance(commit,str) or not re.fullmatch('[0-9a-f]{40}',commit)):
        raise PermissionError('Operational recovery applies only to the remaining original partitions with a new reviewed source')
    return {'profile':PROFILE,'source_commit':commit,'partition':partition,
        'original_scientific_source_commit':ORIGINAL_SOURCE,'original_common_freeze_sha256':PARENT_COMMON_FREEZE,
        'source_relationship':RELATIONSHIP,'maintenance_profile':MAINTENANCE_PROFILE,
        'scientific_core_files_unchanged':CORE_FILES,'parent_run_id':PARENT_RUN,
        'parent_archive_sha256':PARENT_ARCHIVE_SHA256,'parent_prefix_audit_sha256':PARENT_PREFIX_AUDIT_SHA256,
        'parent_closeout_sha256':PARENT_CLOSEOUT_SHA256,'parent_source_release_sha256':PARENT_RELEASE_SHA256,
        'parent_sealed_forward_counts':{'attempted':1260,'completed':1260},
        'preserved_parent_prefix_forwards':1259,'parent_interrupted_response_forwards_retained':1,
        'preserved_parent_native_responses':14 if partition==3 else 0,
        'tool_boundary_continuation':partition==3,'exact_interrupted_generation_resume':False,
        'interrupted_generation_reconstruction_required':partition==3,
        'configuration_models_cohorts_controls_criteria_unchanged':True,
        'mixed_operational_versions_must_be_disclosed':True,'original_uniform_source_aggregate_allowed':False,
        'optimizer_updates':0,'behavior_adapter_fitting':False,'heldout_content_read':False}


def validate_parent_inputs(inputs):
    root=Path(inputs)
    for name,expected in [(PARENT_PREFIX_AUDIT,PARENT_PREFIX_AUDIT_SHA256),
                          (PARENT_CLOSEOUT,PARENT_CLOSEOUT_SHA256),(PARENT_RELEASE,PARENT_RELEASE_SHA256)]:
        if sha(root/name)!=expected:raise PermissionError('Original failed-parent recovery evidence changed')
    prefix=read(root/PARENT_PREFIX_AUDIT);closeout=read(root/PARENT_CLOSEOUT);release=read(root/PARENT_RELEASE)
    if (prefix['partial_prefix_audited'] is not True or prefix['complete_base_partition_audited'] is not False
        or prefix['sealed_forward_counts']!={'attempted':1260,'completed':1260}
        or prefix['native_complete_responses']!=14 or prefix['additional_interrupted_response_forwards']!=1
        or prefix['exact_interrupted_generation_resume_supported'] is not False
        or closeout['run_id']!=PARENT_RUN or closeout['source_commit']!=ORIGINAL_SOURCE
        or closeout['owned_compute_released'] is not True or closeout['owned_cache_removed'] is not True
        or release['commit']!=ORIGINAL_SOURCE):
        raise PermissionError('The original consumed failed parent is not audited and released')
    return prefix


def validate_operational_transport(data):
    manifest=read(data.root/'TRANSPORT_MANIFEST.json')
    if manifest.get('version')!='frozen-base-operational-inputs-v2' or manifest.get('heldout_transferred') is not False:
        raise ValueError('Unregistered operational transport')
    raw=base64.b64decode(manifest['original_base_transport_manifest_bytes'],validate=True)
    if hashlib.sha256(raw).hexdigest()!=PARENT_TRANSPORT_SHA256:
        raise ValueError('The original complete BASE transport changed')
    original=json.loads(raw)
    # The complete original manifest is bound by its already admitted byte SHA.
    expected={**original['files'],PARENT_ARCHIVE:PARENT_ARCHIVE_SHA256}
    if manifest['files']!=expected or original.get('version')!='frozen-base-kaggle-inputs-v1':
        raise ValueError('Operational transport must preserve all original files and only the exact failed archive')
    return expected


def validate_runtime_profile(root,commit,pin,authority):
    root=Path(root);value=read(root/'inputs'/DOCUMENT)
    if pin.get('operational_profile')!=PROFILE or value!=document(commit,pin['partition']):
        raise PermissionError('The explicit operational source/parent/partition relationship changed')
    if authority.get('operational_profile')!=PROFILE or authority.get('operational_document_sha256')!=sha(root/'inputs'/DOCUMENT):
        raise PermissionError('Human continuation does not bind this versioned operational descriptor')
    validate_parent_inputs(root/'inputs')
    original=read(root/'inputs'/PARENT_RELEASE)
    for name,expected in CORE_FILES.items():
        if original['files'].get(name)!=expected or sha(root/name)!=expected:
            raise ValueError('A scientific model/interface/world/criterion source changed in an operational recovery')
    return value

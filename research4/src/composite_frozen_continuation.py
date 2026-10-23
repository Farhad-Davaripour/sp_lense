"""Tool-boundary continuation libraries; no worker or submission admission.

An interrupted generation's KV/output is not restored. This preserves finished
native/tool steps and explicitly records any required generation reconstruction.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import io
import json
import zipfile

from .io_utils import digest
from .composite_audit import _replay_trajectory
from .composite_controller import _rollout_from_world
from .composite_environment import ENVIRONMENT
from .composite_exports import verify_evidence_archive

PROFILE='frozen-tool-boundary-continuation-v1'
ORIGINAL_SOURCE='8b86888018a3d70c8ae30e2c5bdc395e874ab075'
PARENT_RUN=361586756
PARENT_ARCHIVE_SHA256='1479d766b0cf36d11666cfebe232664229743bd759d2706b8c170e3300e31238'
PARENT_COMMON_FREEZE='e11e5bb720c7bf3dd4e84356c40823b6a345a71501aa88de611230ee3a5385ab'


def validate_checkpoint(record,certificate):
    if (certificate.get('partial_prefix_audited') is not True
        or certificate.get('source_commit')!=ORIGINAL_SOURCE
        or certificate.get('archive_sha256')!=PARENT_ARCHIVE_SHA256
        or certificate.get('partition')!=3
        or certificate.get('complete_base_partition_audited') is not False
        or certificate.get('all_original_partition_cases_pending') is not True
        or certificate.get('exact_interrupted_generation_resume_supported') is not False
        or certificate.get('interrupted_KV_cache_durable') is not False
        or certificate.get('interrupted_response_output_durable') is not False):
        raise PermissionError('Only the audited original failed partition is a continuation parent')
    if (certificate['sealed_forward_counts']!={'attempted':1260,'completed':1260}
        or any(type(value) is not int for value in certificate['sealed_forward_counts'].values())
        or any(type(certificate[key]) is not int for key in ['fully_accounted_prefix_forwards','additional_interrupted_response_forwards','native_complete_responses'])
        or certificate['fully_accounted_prefix_forwards']!=1259
        or certificate['additional_interrupted_response_forwards']!=1
        or certificate['native_complete_responses']!=14):
        raise PermissionError('Unknown or differently accounted parent work cannot become a fresh attempt')
    boundary=certificate['last_complete_tool_boundary']
    if (digest(record)!=boundary['durable_prefix_sha256'] or record['condition']!='base'
        or record['environment']!=ENVIRONMENT or record['scene']['case_id']!=certificate['case_ids'][0]
        or record['scene']['case_id']!=boundary['case_id'] or len(record['trace'])!=14
        or record['outcome']['running'] is not True
        or record['terminal_memory']!=boundary['predicted_memory']
        or record['terminal_memory']['generation_active'] is not False
        or record['trace'][-1]['observation_after']['sequence']!=boundary['sequence']):
        raise ValueError('The exact durable tool boundary was altered or made terminal')
    return {'profile':PROFILE,'parent_run_id':PARENT_RUN,'parent_source_commit':ORIGINAL_SOURCE,
        'parent_archive_sha256':PARENT_ARCHIVE_SHA256,'original_common_freeze_sha256':PARENT_COMMON_FREEZE,
        'durable_prefix_sha256':boundary['durable_prefix_sha256'],'preserved_native_responses':14,
        'parent_sealed_forward_counts':certificate['sealed_forward_counts'],
        'parent_interrupted_response_forwards_retained':1,
        'exact_interrupted_generation_resume':False,'interrupted_generation_reconstruction_required':True,
        'model_or_detector_forwards_during_restore':0,'completed_parent_trajectories':0}


def prefix_from_parent(raw,certificate,scene):
    if hashlib.sha256(raw).hexdigest()!=PARENT_ARCHIVE_SHA256:
        raise ValueError('The exact failed parent archive changed')
    verify_evidence_archive(raw)
    with zipfile.ZipFile(io.BytesIO(raw)) as stream:
        case_id=scene['case_id'];n=certificate['native_complete_responses']
        steps=[json.loads(stream.read('steps/'+case_id+'_'+str(i+1).zfill(3)+'.json')) for i in range(n)]
        initial=json.loads(stream.read('observations/'+case_id+'_000.json'))
    record={'scene':scene,'condition':'base','environment':ENVIRONMENT,'initial_observation':initial,
        'trace':steps,'outcome':steps[-1]['outcome'],'terminal_memory':steps[-1]['observation_after']['memory']}
    validate_checkpoint(record,certificate)
    return record


def restore_tool_boundary(composite,scene,record,certificate,tokenizers=None,eos=None):
    """Reconstruct inert state with saved actions; never call either backbone."""
    proof=validate_checkpoint(record,certificate)
    if asdict(scene)!=record['scene']:raise ValueError('Continuation must retain the original complete scene')
    if composite.memory.generation_active or composite.memory.episode_id:
        raise PermissionError('A tool-boundary restore requires a fresh idle composite instance')
    bt,qt=(None,None) if tokenizers is None else tokenizers
    world,memory=_replay_trajectory(record,.9,bt,qt,eos,'reference',
        None if tokenizers is None else {'detector':2048,'actor':6144},False)
    if not world.running or memory.generation_active:raise PermissionError('The recovered boundary is not an idle running episode')
    composite.actor.clear_cache()
    composite.memory=memory
    composite.initial_gate=record['initial_observation']['memory']['applied_gate']
    composite.observations=deepcopy([record['initial_observation']]+[step['observation_after'] for step in record['trace']])
    proof.update(restored_world_outcome=world.outcome(),restored_memory=memory.snapshot(),
                 prefix_actions_replayed_only_inside_inert_simulator=True)
    return world,deepcopy(record['trace']),proof


def continue_base_rollout(composite,scene,record,certificate,guard,progress=None,tokenizers=None,eos=None):
    world,trace,proof=restore_tool_boundary(composite,scene,record,certificate,tokenizers,eos)
    value=_rollout_from_world(composite,world,'base',guard,trace,progress)
    return value,proof

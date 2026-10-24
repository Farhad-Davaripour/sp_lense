"""Versioned Kaggle continuation binding; no provider calls or allocation.

The first registered mode is complete CPU data export. Full model training gets
its own reviewed device/runtime profile; old inference never admits fitting.
"""
import os,re
from pathlib import Path
from .io_utils import read,sha,verify_release

PROFILE='research4-kaggle-remainder-v1'
FILE='KAGGLE_REMAINDER_PROFILE.json'
HUMAN='HUMAN_KAGGLE_REMAINDER_SCOPE.json'
HUMAN_MESSAGE='01a16e46-ff20-7eb3-a35b-dabc0efe6e5f'
HUMAN_SHA='b2d838773bdfe3d3a62bfe65eb76f59ae3f4a7e1543c71ef7ee90d681917332d'
OUTPUT_BASE='/kaggle/working/research4_campaign/runs'
INPUT_BASE='/kaggle/working/research4_inputs'
DATA_MODE='generalized-data-export-v2'


def identity(notebook):
    if not isinstance(notebook,str) or not re.fullmatch('farhad89/research4-[a-z0-9-]+',notebook):
        raise PermissionError('Exact authorized Research4 Kaggle notebook required')
    return {'platform':'kaggle','account':'farhad89','label':'Kaggle Research4','notebook':notebook,'artifact_root':OUTPUT_BASE}


def validate_identity(value):
    if value!=identity(value.get('notebook')):raise PermissionError('Kaggle account/notebook/artifact identity changed')
    return value


def data_destination(value,commit):
    validate_identity(value)
    if not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('Exact source commit required')
    return OUTPUT_BASE+'/'+commit+'/data_generalized_v2/export_01'


def validate_source(root,commit):
    root=Path(root);verify_release(root,commit);manifest=read(root/'inputs/MANIFEST.json')
    for name in [FILE,HUMAN,'WORKSPACE.json']:
        if manifest.get(name)!=sha(root/'inputs'/name):raise PermissionError('Kaggle source/provider input binding changed')
    if sha(root/'inputs'/HUMAN)!=HUMAN_SHA or read(root/'inputs'/HUMAN).get('human_message_id')!=HUMAN_MESSAGE:
        raise PermissionError('Bind the verified direct human Kaggle correction')
    value=read(root/'inputs'/FILE);who=validate_identity(read(root/'inputs/WORKSPACE.json'))
    expected={'version':PROFILE,'platform':'kaggle','account':'farhad89','notebook':who['notebook'],
        'source_commit':commit,'mode':DATA_MODE,'visible_devices':'','scientific_GPU_count':0,
        'hard_timeout_seconds':900,'worker_seconds':600,'export_reserve_seconds':60,
        'maximum_raw_export_bytes':256*1024**2,'maximum_concurrent_owned_workers':1,
        'automatic_scientific_retries':0,'heldout_content_read':False,
        'human_scope_sha256':HUMAN_SHA,'destination':data_destination(who,commit)}
    if value!=expected:raise PermissionError('Unreviewed Kaggle mode/device/runtime/export profile')
    return value,who


def require_v2_runtime():
    from .composite_runtime import require_scientific_runtime,platform
    require_scientific_runtime()
    if platform()!='kaggle':raise PermissionError('Current Research4 remainder uses Kaggle; Databricks requires a later explicit human selection')
    root=Path(os.environ['RESEARCH4_KAGGLE_V2_SOURCE_ROOT']).resolve()
    if not root.is_relative_to(Path(INPUT_BASE)) or root.name!='source':raise PermissionError('Owned frozen Kaggle source root required')
    commit=read(root/'RELEASE.json')['commit'];profile,_=validate_source(root,commit)
    if os.environ.get('RESEARCH4_KAGGLE_V2_PROFILE_SHA256')!=sha(root/'inputs'/FILE):raise PermissionError('Actual Kaggle profile differs')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=profile['visible_devices']:raise PermissionError('CPU data stage must expose no GPUs')


def actual_identity(root,commit):
    require_v2_runtime();profile,who=validate_source(root,commit)
    if Path(root).resolve()!=Path(os.environ['RESEARCH4_KAGGLE_V2_SOURCE_ROOT']).resolve():raise PermissionError('Another Kaggle source root entered the stage')
    return who


def assert_output_namespace(path):
    require_v2_runtime();root=Path(os.environ['RESEARCH4_KAGGLE_V2_SOURCE_ROOT'])
    profile,_=validate_source(root,read(root/'RELEASE.json')['commit'])
    target=Path(path).resolve();base=Path(profile['destination']).resolve()
    if ('heldout' in str(target).lower() or 'sealed' in str(target).lower()
        or (target!=base and not target.is_relative_to(base))):raise PermissionError('Output escaped the exact unsealed Kaggle stage')
    return target

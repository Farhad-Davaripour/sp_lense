"""Reuse authenticated, already prepared diagnostic inputs; no model work."""
import json, os, types
from support import HERE, ROOT, require, sha, json_bytes, check_freeze

PARENT_PATH = HERE.parent / 'diagnostic_reader.py'
PARENT_SHA256 = '7b4d3509ac1a51bb3401a54f855c7d513a4e3f8e184ee25ab187ad85e5629ef9'

def keys():
    return tuple(f'{f}_{c}__{o}' for f in ('G10','G11')
        for c in ('self_shutdown','other_shutdown','non_termination_control')
        for o in ('KEEP_then_STOP','STOP_then_KEEP')) + tuple(
        f'{k}__{o}' for k in ('O09','O10') for o in ('A_then_B','B_then_A'))

def accepted_reader():
    frozen = check_freeze()
    relative = PARENT_PATH.relative_to(ROOT).as_posix()
    matches = [p for p in frozen['external_sources'] if p['path'] == relative]
    require(len(matches) == 1 and matches[0]['sha256'] == PARENT_SHA256,
        'ACCEPTED_READER_SOURCE_PIN')
    require(PARENT_PATH.is_file() and not PARENT_PATH.is_symlink()
        and PARENT_PATH.resolve().is_relative_to(ROOT.resolve())
        and PARENT_PATH.stat().st_size <= 5 * 1024**2, 'ACCEPTED_READER_FILE_BOUND')
    raw = PARENT_PATH.read_bytes()
    require(sha(raw) == PARENT_SHA256, 'ACCEPTED_READER_BYTES')
    module = types.ModuleType('prechoice_capture_accepted_reader')
    module.__file__ = str(PARENT_PATH)
    exec(compile(raw, str(PARENT_PATH), 'exec'), module.__dict__)
    return module

def build_inputs():
    result = accepted_reader().reuse_accepted_prepared_inputs()['prepared_inputs']
    require(tuple(c['case_key'] for c in result['cases']) == keys(), 'EXACT_DIAGNOSTIC_KEYS')
    return result

def read_bundle(base, release):
    require(release['input_data_lock_sha256'] == sha((HERE / 'DATA_LOCK.json').read_bytes()),
        'RELEASE_DATA_LOCK')
    result = build_inputs()
    require(result['data_lock_sha256'] == release['input_data_lock_sha256'],
        'PREPARED_DATA_LOCK_JOIN')
    require(result['certificate_sha256'] == release['certificate_sha256'],
        'RELEASE_CERTIFICATE_JOIN')
    require(sha(json_bytes(result)) == release['inputs_sha256'], 'RELEASE_EXACT_INPUTS')
    return result

def read():
    from authority import read_release, RELEASE
    release = read_release(os.environ.get('SP_NATIVE_RELEASE_SHA', ''))
    return read_bundle(RELEASE.parent, release)

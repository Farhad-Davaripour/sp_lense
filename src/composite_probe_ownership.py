"""Owned ephemeral paths; a parent may clean cache only after confirmed exit."""
import json
from pathlib import Path
import re
import shutil

CACHE_BASE = '/kaggle/temp/research4_frozen_probe_cache'
MARKER = 'CACHE_OWNER.json'


def cache_target(cache, commit, namespace=CACHE_BASE):
    if not re.fullmatch('[0-9a-f]{40}', commit): raise ValueError('Invalid source-bound cache identity')
    base = (Path(namespace) / commit).resolve(); target = Path(cache).resolve()
    if target == base or not target.is_relative_to(base) or Path(cache).is_symlink():
        raise ValueError('Cache escaped the explicitly owned ephemeral namespace')
    return target


def create_owned_cache(cache, commit, token, namespace=CACHE_BASE):
    if not re.fullmatch('[0-9a-f]{32}', token): raise ValueError('Invalid cache ownership token')
    target = cache_target(cache, commit, namespace)
    target.mkdir(parents=True, exist_ok=False)
    with (target / MARKER).open('x') as stream:
        json.dump({'source_commit': commit, 'token': token, 'path': str(target)}, stream)
    for name in ['scratch','hf_home','xet']:(target/name).mkdir()
    return str(target)


def verify_owned_cache(cache, commit, token, namespace=CACHE_BASE):
    if not re.fullmatch('[0-9a-f]{32}', token): raise ValueError('Invalid cache ownership token')
    target = cache_target(cache, commit, namespace)
    value = json.loads((target / MARKER).read_text())
    if value != {'source_commit': commit, 'token': token, 'path': str(target)}:
        raise PermissionError('The ephemeral cache belongs to another owner')
    return target


def cleanup_owned_cache(cache, commit, token, child_exited, namespace=CACHE_BASE):
    if child_exited is not True:
        raise PermissionError('Never remove cache while its child is live or exit is unknown')
    target = verify_owned_cache(cache, commit, token, namespace)
    # target is resolved, inside the explicit source namespace, and marker-bound.
    # rmtree does not follow directory symlinks to remove their external targets.
    shutil.rmtree(target)
    return {'owned_cache_removed': not target.exists(), 'child_exit_confirmed': True}

"""Prepare one bounded rank-16 trajectory after all rank-8 DEV audits pass.

This helper creates a specification and operator-gate record only. It never
starts a model or optimizer. Prior specs, checkpoints and evidence are kept.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = 'dc1d812ba3e2b2aa147b1fbe148440ff6a5327a5a67feb8d32f330883ac98f70'
RANK8_OWNER = '07c3439302aa424cbf435be58dce0583'
RANK8_SPEC_SHA = 'eebcca4b2435751335dac9780af7519ca3983eb6c20d3c930652e2cf20dd4722'
RANK16_QUAL_SHA = 'bfc81611ddb64588d88d4bbfa4a379e17b1229e961cc68d1f14321bcb1f71bd3'
AUTH = '.local/experiments/all64_broader_fallback_actualwidth_v7_20261119/authorities'
AUDIT_DIR = '.local/reviews/all64_broader_v2_20261119'
MASK_SHA = 'b2ed41957a4e56e8e09236d7cbbce1807f9d7fca2e66529733cc06be767752ba'
V8_SUP = 'mac_studio/all64_broader_fallback_actualwidth_supervision_v8_20261120/supervisor.py'
V8_SUP_SHA = '92bdc804f714093d4656c4b85ca86bc535f729a97bd449902281677c6a62b3ce'
V8_DEV = 'mac_studio/all64_broader_fallback_actualwidth_v8_20261120/eval_worker.py'
V8_DEV_SHA = '5abecbb48960a734169317323e4f24aac87c042a73ddd2e4a6171ddd8f719c8d'
AUDITS = {
    1: ('37204c49b7c7496aaafd2f8e79ef0b6a', 'e7ef7ea27cc8854d0f7d2fb7b6c484bac6fcb495f6f9cc2def64af64d44c8fc5',
        'f026bdb42cb76888767373ca9b54410986c8300e0c220e82aaf493d5a231c8d4'),
    2: ('9a611958cc4c4e5d89f9bf0e99cfff34', 'a72d22080f97b6b4282e80c875cb9a209a7c341922808c9ab8fddde0167e3f2b',
        '27446c258dc27bb87c8eee9bb966bf1abe75ba0e32adf54d9df22c7c02783c90'),
    3: ('22c114e6b40541b5bedc635d9cb85f62', None,
        '6f9a8b1f60f477f3fc2583621c3316a28ddcfb4f19d16839ad86beb6352bb9c7'),
}


def exact(relative, digest):
    if (not isinstance(relative, str) or not relative or '\\' in relative or ':' in relative
            or Path(relative).is_absolute() or '..' in Path(relative).parts
            or not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest)):
        raise ValueError('Repository-relative evidence and exact SHA-256 required')
    path = ROOT / relative
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Expected a non-symlink repository file')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('Evidence hash differs: ' + relative)
    return json.loads(raw)


def source_hash(relative, digest):
    path = ROOT / relative
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Expected a non-symlink reviewed source')
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError('Reviewed source differs: ' + relative)


def write_once(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    with os.fdopen(fd, 'w', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def create(epoch3_audit_sha):
    old = exact(AUTH + '/' + RANK8_OWNER + '/SPEC.json', RANK8_SPEC_SHA)
    if (old.get('phase'), old.get('rank'), old.get('epoch'), old.get('manifest_sha256')) != ('trajectory', 8, 0, MANIFEST):
        raise ValueError('Original frozen rank-8 trajectory spec differs')
    if old['prerequisites']['16']['sha256'] != RANK16_QUAL_SHA:
        raise ValueError('Original rank-16 qualification differs')
    source_hash(V8_SUP, V8_SUP_SHA); source_hash(V8_DEV, V8_DEV_SHA)
    refs = []
    for epoch, (owner, fixed_sha, checkpoint_sha) in AUDITS.items():
        digest = epoch3_audit_sha if epoch == 3 else fixed_sha
        path = AUDIT_DIR + '/RANK8_DEV_EPOCH%d_INDEPENDENT_AUDIT_%s.json' % (epoch, owner)
        audit = exact(path, digest)
        if (audit.get('schema'), audit.get('status'), audit.get('owner'), audit.get('rank'),
                audit.get('epoch'), audit.get('optimizer_absent'), audit.get('development_no_optimizer_updates'),
                audit.get('worker_group_empty'), audit.get('shared_lease_absent'),
                audit.get('frozen_mask_sha256')) != (
                'research4.rank8-epoch%d-independent-dev-audit.v1' % epoch, 'PASS', owner, 8,
                epoch, True, True, True, True, MASK_SHA):
            raise ValueError('Rank-8 epoch %d DEV audit or release differs' % epoch)
        if (audit.get('checkpoint', {}).get('sha256') != checkpoint_sha or
                (audit.get('summary', {}).get('rows'), audit.get('summary', {}).get('families'),
                 audit.get('summary', {}).get('action_tokens')) != (36, 11, 674)):
            raise ValueError('Rank-8 epoch %d checkpoint or DEV totals differ' % epoch)
        closeout_ref = audit.get('closeout')
        if type(closeout_ref) is not dict or set(closeout_ref) != {'path', 'sha256'}:
            raise ValueError('Exact DEV closeout reference required')
        closeout = exact(closeout_ref['path'], closeout_ref['sha256'])
        if (closeout.get('owner'), closeout.get('phase'), closeout.get('rank'), closeout.get('epoch'),
                closeout.get('status'), closeout.get('worker_exit_code'), closeout.get('owned_cleanup_confirmed')) != (
                owner, 'dev', 8, epoch, 'completed', 0, True):
            raise ValueError('Rank-8 epoch %d DEV closeout differs' % epoch)
        refs.append({'path': path, 'sha256': digest})
    owner = uuid.uuid4().hex; spec = dict(old); spec['source_files'] = dict(old['source_files'])
    spec['source_files'].update({V8_SUP: V8_SUP_SHA, V8_DEV: V8_DEV_SHA})
    spec.update({'owner': owner, 'phase': 'trajectory', 'rank': 16, 'epoch': 0, 'checkpoint': None,
                 'deadline_epoch': time.time() + 43100, 'max_wall_seconds': 43200})
    folder = ROOT / AUTH / owner; folder.mkdir(mode=0o700, parents=False, exist_ok=False)
    spec_path = folder / 'SPEC.json'; write_once(spec_path, spec)
    gate = {'schema': 'research4.rank16-after-rank8-dev-gate.v1', 'status': 'PREPARED_NOT_LAUNCHED',
            'owner': owner, 'rank8_dev_audits': refs,
            'rank16_spec': {'path': spec_path.relative_to(ROOT).as_posix(),
                            'sha256': hashlib.sha256(spec_path.read_bytes()).hexdigest()},
            'optimizer_updates_authorized_by_this_file': 0}
    write_once(folder / 'OPERATOR_GATE.json', gate)
    return {'owner': owner, **gate['rank16_spec'], 'deadline_epoch': spec['deadline_epoch'],
            'gate_sha256': hashlib.sha256((folder / 'OPERATOR_GATE.json').read_bytes()).hexdigest(),
            'model_executed': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch3-audit-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(create(args.epoch3_audit_sha256)))

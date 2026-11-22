import hashlib
import json
import time

import pytest

import create_rank16_trajectory_spec as target


def write(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf8')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup_evidence(root, monkeypatch):
    monkeypatch.setattr(target, 'ROOT', root)
    original = {'phase': 'trajectory', 'rank': 8, 'epoch': 0, 'manifest_sha256': target.MANIFEST,
                'source_files': {'prior.py': '1' * 64},
                'prerequisites': {'16': {'sha256': target.RANK16_QUAL_SHA}}}
    old_path = target.AUTH + '/' + target.RANK8_OWNER + '/SPEC.json'
    monkeypatch.setattr(target, 'RANK8_SPEC_SHA', write(root, old_path, original))
    for rel, constant in ((target.V8_SUP, 'V8_SUP_SHA'), (target.V8_DEV, 'V8_DEV_SHA')):
        path = root / rel; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(rel.encode())
        monkeypatch.setattr(target, constant, hashlib.sha256(path.read_bytes()).hexdigest())
    refs = {}
    for epoch in (1, 2, 3):
        owner, _, checkpoint_sha = target.AUDITS[epoch]
        closeout_path = '.local/monitor/' + owner + '/CLOSEOUT.json'
        closeout_sha = write(root, closeout_path, {'owner': owner, 'phase': 'dev', 'rank': 8,
            'epoch': epoch, 'status': 'completed', 'worker_exit_code': 0, 'owned_cleanup_confirmed': True})
        audit = {'schema': 'research4.rank8-epoch%d-independent-dev-audit.v1' % epoch,
            'status': 'PASS', 'owner': owner, 'rank': 8, 'epoch': epoch,
            'optimizer_absent': True, 'development_no_optimizer_updates': True,
            'worker_group_empty': True, 'shared_lease_absent': True,
            'frozen_mask_sha256': target.MASK_SHA, 'checkpoint': {'sha256': checkpoint_sha},
            'summary': {'rows': 36, 'families': 11, 'action_tokens': 674},
            'closeout': {'path': closeout_path, 'sha256': closeout_sha}}
        audit_path = target.AUDIT_DIR + '/RANK8_DEV_EPOCH%d_INDEPENDENT_AUDIT_%s.json' % (epoch, owner)
        refs[epoch] = (owner, write(root, audit_path, audit), checkpoint_sha)
    monkeypatch.setattr(target, 'AUDITS', {epoch: (owner, sha if epoch != 3 else None, checkpoint)
                                           for epoch, (owner, sha, checkpoint) in refs.items()})
    return original, refs


def test_rank16_spec_requires_all_three_audited_dev_epochs(tmp_path, monkeypatch):
    original, refs = setup_evidence(tmp_path, monkeypatch)
    before = time.time(); result = target.create(refs[3][1])
    spec = json.loads((tmp_path / result['path']).read_text())
    gate = json.loads((tmp_path / result['path']).with_name('OPERATOR_GATE.json').read_text())
    assert (spec['phase'], spec['rank'], spec['epoch'], spec['checkpoint']) == ('trajectory', 16, 0, None)
    assert spec['prerequisites'] == original['prerequisites']
    assert spec['source_files']['prior.py'] == original['source_files']['prior.py']
    assert spec['source_files'][target.V8_SUP] == target.V8_SUP_SHA
    assert spec['source_files'][target.V8_DEV] == target.V8_DEV_SHA
    assert before + 43099 <= spec['deadline_epoch'] <= time.time() + 43100
    assert len(gate['rank8_dev_audits']) == 3 and result['model_executed'] is False
    assert gate['optimizer_updates_authorized_by_this_file'] == 0


def test_wrong_epoch3_audit_hash_cannot_create_spec(tmp_path, monkeypatch):
    _, refs = setup_evidence(tmp_path, monkeypatch)
    before = set((tmp_path / target.AUTH).iterdir())
    with pytest.raises(ValueError, match='hash differs'):
        target.create('0' * 64)
    assert set((tmp_path / target.AUTH).iterdir()) == before

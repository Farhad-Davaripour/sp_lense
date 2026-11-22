"""Regression for the real posthoc terminal guard-exit marker path."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import supervisor as sup


class ReachedAfterGuard(Exception):
    pass


class TerminalMarkerReplayTests(unittest.TestCase):
    def make_fixture(self, root, include_marker=True, tamper_marker=False, marker_name='EXIT_DURING_GUARD.json'):
        owner = 'a' * 32
        monitor = root / sup.MONITOR / owner
        run = root / sup.RUNS / owner
        monitor.mkdir(parents=True); run.mkdir(parents=True)
        def write(path, value):
            path.write_text(json.dumps(value), encoding='utf8')
            return {'path': path.relative_to(root).as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        target = {'pid': 1234, 'create_time_epoch': 123.5}
        def healthy(phase):
            return {'schema': 'research4.inference-native-guard-sample.v1',
                    'policy_id': 'research4.mini.inference_NATIVE_R_NORMAL_only.6GiB_launch_4GiB_run.v1',
                    'phase': phase, 'target': target, 'valid': True, 'stop_required': False,
                    'assessment': {'pressure_flag_distinct_from_bytes': 1}}
        invalid = {'schema': 'research4.inference-native-guard-sample.v1',
                   'policy_id': 'research4.mini.inference_NATIVE_R_NORMAL_only.6GiB_launch_4GiB_run.v1',
                   'phase': 'run', 'target': target, 'valid': False, 'stop_required': True,
                   'assessment': {'reason': 'invalid_cache_estimate_sample',
                                  'errors': ['rusage reports exited process'],
                                  'pressure_flag_distinct_from_bytes': 1}}
        guard = monitor / 'GUARD.jsonl'
        samples = [healthy('launch'), healthy('run')] + ([invalid] if include_marker else [])
        guard.write_text(''.join(json.dumps(x, separators=(',', ':')) + '\n' for x in samples), encoding='utf8')
        guard_ref = {'path': guard.relative_to(root).as_posix(), 'sha256': hashlib.sha256(guard.read_bytes()).hexdigest()}
        closeout = {'schema': 'research4.all64-broader-supervisor-closeout.v1', 'status': 'completed',
                    'phase': 'trajectory', 'rank': 8, 'manifest_sha256': sup.MANIFEST, 'owner': owner,
                    'worker_exit_code': 0, 'owned_cleanup_confirmed': True,
                    'worker_receipt': write(run / 'worker_receipt.json', {'owner': owner}),
                    'release': write(monitor / 'RELEASE.json', {'owner': owner, 'lease_released': True,
                        'owned_cleanup_confirmed': True, 'error': None}),
                    'worker_identity': write(monitor / 'WORKER_IDENTITY.json', {'owner': owner, **target}),
                    'guard_evidence': guard_ref}
        if include_marker:
            marker = {'schema': 'research4.all64-terminal-guard-exit-race.v1',
                      'worker_pid': 4321 if tamper_marker else target['pid'],
                      'worker_create_time_epoch': target['create_time_epoch'],
                      'sample_sha256': sup.sample_hash(invalid), 'waitid_exited_after_sample': True,
                      'invalid_sample_treated_as_healthy': False, 'worker_allowed_to_continue': False}
            closeout['guard_exit_race'] = write(monitor / marker_name, marker)
        ref = write(monitor / 'CLOSEOUT.json', closeout)
        return ref, guard, target, closeout

    def test_live_and_posthoc_use_same_bound_marker_and_reject_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); ref, guard, target, closeout = self.make_fixture(root)
            marker = json.loads((root / closeout['guard_exit_race']['path']).read_text())
            self.assertEqual(sup.verify_guard_evidence(guard, target['pid'], target['create_time_epoch'], marker), 3)
            with self.assertRaisesRegex(ValueError, 'Unhealthy'):
                sup.verify_guard_evidence(guard, target['pid'], target['create_time_epoch'])
            original = sup.verify_guard_evidence
            def stop_after_real_guard(*args):
                self.assertEqual(original(*args), 3)
                self.assertEqual(args[3], marker)
                raise ReachedAfterGuard()
            with patch.object(sup, 'PROJECT', root), patch.object(sup, 'verify_guard_evidence', side_effect=stop_after_real_guard):
                with self.assertRaises(ReachedAfterGuard):
                    sup.validate_completed_closeout(ref, 'trajectory', 8)

    def test_marker_wrong_owner_or_path_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); ref, _, _, _ = self.make_fixture(root, tamper_marker=True)
            with patch.object(sup, 'PROJECT', root):
                with self.assertRaisesRegex(ValueError, 'Unhealthy'):
                    sup.validate_completed_closeout(ref, 'trajectory', 8)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); ref, _, _, _ = self.make_fixture(root, marker_name='other.json')
            with patch.object(sup, 'PROJECT', root):
                with self.assertRaisesRegex(ValueError, 'marker path'):
                    sup.validate_completed_closeout(ref, 'trajectory', 8)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); ref, _, _, closeout = self.make_fixture(root)
            closeout['guard_exit_race']['sha256'] = '0' * 64
            closeout_path = root / ref['path']
            closeout_path.write_text(json.dumps(closeout), encoding='utf8')
            ref['sha256'] = hashlib.sha256(closeout_path.read_bytes()).hexdigest()
            with patch.object(sup, 'PROJECT', root):
                with self.assertRaisesRegex(ValueError, 'missing or changed'):
                    sup.validate_completed_closeout(ref, 'trajectory', 8)

    def test_no_marker_healthy_prior_closeout_still_replays(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); ref, _, _, _ = self.make_fixture(root, include_marker=False)
            original = sup.verify_guard_evidence
            def stop_after_real_guard(*args):
                self.assertEqual(original(*args), 2)
                self.assertIsNone(args[3])
                raise ReachedAfterGuard()
            with patch.object(sup, 'PROJECT', root), patch.object(sup, 'verify_guard_evidence', side_effect=stop_after_real_guard):
                with self.assertRaises(ReachedAfterGuard):
                    sup.validate_completed_closeout(ref, 'trajectory', 8)

    def test_only_exact_v7_or_v8_producer_allowed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            current = root / sup.SUPERVISOR; current.parent.mkdir(parents=True); current.write_bytes(b'v8')
            legacy = hashlib.sha256(b'v7').hexdigest(); modern = hashlib.sha256(b'v8').hexdigest()
            with patch.object(sup, 'PROJECT', root), patch.object(sup, 'PREVIOUS_SUPERVISOR_SHA', legacy):
                sup.validate_producer_supervisor({sup.PREVIOUS_SUPERVISOR: legacy}, {'supervisor_sha256': legacy})
                sup.validate_producer_supervisor({sup.SUPERVISOR: modern}, {'supervisor_sha256': modern})
                with self.assertRaisesRegex(ValueError, 'supervisor source'):
                    sup.validate_producer_supervisor({sup.SUPERVISOR: modern, sup.PREVIOUS_SUPERVISOR: legacy},
                                                     {'supervisor_sha256': legacy})
                with self.assertRaisesRegex(ValueError, 'supervisor source'):
                    sup.validate_producer_supervisor({sup.PREVIOUS_SUPERVISOR: '0' * 64}, {'supervisor_sha256': '0' * 64})


if __name__ == '__main__': unittest.main()

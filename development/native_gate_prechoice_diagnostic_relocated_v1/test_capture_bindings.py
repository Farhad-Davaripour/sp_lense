"""Model-free tests for the relocated capture's binding-only candidate."""
import builtins
from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
CAPTURE = HERE / 'capture'
ROOT = HERE.parents[1]
ORIGINAL = ROOT / 'development/native_gate_prechoice_readout_v1/diagnostic_capture'
INPUT_SHA = '8a899b396d504cc0ccbac2e373cc0e933329803478f6320fc97cbc305c0a7b0f'
FREEZE_SHA = '433f7c1aea7b4016bf2352c894d3061f6ba4e13ea6bdf48a691c1a41dd477a7f'

@contextmanager
def modules():
    names = ('support', 'input_reader', 'authority')
    prior = {n: sys.modules.get(n) for n in names}
    paths = list(sys.path)
    original_import = builtins.__import__
    def checked_import(name, *args, **kwargs):
        if name.split('.')[0] in ('torch', 'transformers', 'datasets',
                                  'huggingface_hub', 'transformer_lens'):
            raise AssertionError('PROVIDER_IMPORT_FORBIDDEN')
        return original_import(name, *args, **kwargs)
    try:
        for n in names:
            sys.modules.pop(n, None)
        sys.path.insert(0, str(CAPTURE))
        with mock.patch('builtins.__import__', side_effect=checked_import):
            yield tuple(importlib.import_module(n) for n in names)
    finally:
        for n in names:
            sys.modules.pop(n, None)
            if prior[n] is not None:
                sys.modules[n] = prior[n]
        sys.path[:] = paths

def release_document(support, authority):
    frozen = support.check_freeze()
    result = dict(schema='prechoice_diagnostic_capture_release.v1', approved=True,
        attempt=support.ATTEMPT, limits=dict(authority.LIMITS),
        role='DIAGNOSTIC_PRECHOICE', output=str(support.output()),
        reserved_bytes=19628032, inputs_sha256=INPUT_SHA,
        certificate_sha256='57e700c03c46243b2cee9f1cd9dd6dd4fd0a556fec472a325af5e390a848571f',
        fit_freeze_sha256=FREEZE_SHA,
        trace_sources={n:h for n,h in frozen['source_sha256'].items() if n.endswith('.py')})
    for name, field in [('SOURCE_FREEZE.json','source_freeze_sha256'),
        ('CHECKPOINT.json','checkpoint_lock_sha256'),
        ('OWNED_IDENTITY.json','owned_identity_sha256'), ('DATA_LOCK.json','input_data_lock_sha256')]:
        result[field] = support.sha((CAPTURE / name).read_bytes())
    return result

class CaptureBindingTests(unittest.TestCase):
    def test_01_existing_inputs_exact_and_no_provider(self):
        with modules() as (s, r, a):
            bundle = r.build_inputs()
            self.assertEqual(s.sha(s.json_bytes(bundle)), INPUT_SHA)
            self.assertEqual(tuple(c['case_key'] for c in bundle['cases']), r.keys())
            self.assertEqual(len(bundle['cases']), 16)

    def test_02_input_release_joins_fail_closed(self):
        with modules() as (s, r, a):
            release = release_document(s, a)
            self.assertEqual(s.sha(s.json_bytes(r.read_bundle(None, release))), INPUT_SHA)
            for key in ('input_data_lock_sha256','certificate_sha256','inputs_sha256'):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    r.read_bundle(None, {**release, key:'0'*64})

    def test_03_substituted_reader_rejected_before_exec(self):
        with modules() as (s, r, a):
            frozen = s.check_freeze()
            original = Path.read_bytes
            def corrupted(path):
                return b'raise AssertionError("EXECUTED")' if path == r.PARENT_PATH else original(path)
            with mock.patch.object(r, 'check_freeze', return_value=frozen):
                with mock.patch.object(Path, 'read_bytes', corrupted):
                    with self.assertRaisesRegex(ValueError, 'ACCEPTED_READER_BYTES'):
                        r.accepted_reader()

    def test_04_parent_external_pin_required(self):
        with modules() as (s, r, a):
            frozen = {**s.check_freeze(), 'external_sources':[]}
            with mock.patch.object(r, 'check_freeze', return_value=frozen):
                with self.assertRaisesRegex(ValueError, 'ACCEPTED_READER_SOURCE_PIN'):
                    r.accepted_reader()

    def test_05_checkpoint_and_release_approval_join(self):
        with modules() as (s, r, a):
            before = (a.RELEASE.exists(), s.output().exists())
            original = Path.read_bytes
            for approved, fit in ((True, FREEZE_SHA), (True, '0'*64), (False, FREEZE_SHA)):
                document = {**release_document(s, a), 'approved':approved, 'fit_freeze_sha256':fit}
                raw = s.json_bytes(document)
                def mocked(path):
                    return raw if path == a.RELEASE else original(path)
                with mock.patch.object(Path, 'read_bytes', mocked):
                    if approved and fit == FREEZE_SHA:
                        self.assertEqual(a.read_release(s.sha(raw)), document)
                    else:
                        with self.assertRaises(ValueError):
                            a.read_release(s.sha(raw))
            self.assertEqual((a.RELEASE.exists(), s.output().exists()), before)

    def test_06_paths_limits_and_unchanged_engine(self):
        with modules() as (s, r, a):
            self.assertEqual(s.ROOT.resolve(), ROOT.resolve())
            self.assertEqual(s.ATTEMPT, 'prechoice_diagnostic_relocated_capture_attempt_001')
            self.assertEqual(a.LIMITS, dict(loads=1,forwards=16,derivatives=0,
                worker_seconds=300,audit_seconds=120,shared_cleanup_seconds=15,
                total_bytes=64*1024**2,file_bytes=5*1024**2))
            self.assertEqual(sum(s.GROUP_CAPS.values())+65536,19628032)
            for name in ('OWNED_IDENTITY.json','PREPARATION_OWNED_IDENTITY.json'):
                owner=json.loads((CAPTURE/name).read_bytes())
                self.assertEqual(Path(owner['launch_image']).resolve(), (ROOT/'.venv/Scripts/python.exe').resolve())
                self.assertEqual(hashlib.sha256(Path(owner['launch_image']).read_bytes()).hexdigest(),owner['launch_sha256'])
                self.assertEqual(hashlib.sha256(Path(owner['base_image']).read_bytes()).hexdigest(),owner['base_sha256'])
            changed={'input_reader.py','authority.py','support.py','fit_source_auth.py','audit_saved.py',
                'OWNED_IDENTITY.json','PREPARATION_OWNED_IDENTITY.json'}
            old=json.loads((ORIGINAL/'SOURCE_FREEZE.json').read_bytes())
            for name, digest in old['source_sha256'].items():
                if name not in changed:
                    self.assertEqual(hashlib.sha256((CAPTURE/name).read_bytes()).hexdigest(),digest,name)
            namespace='development/native_gate_prechoice_diagnostic_relocated_v1/capture'
            for name in ('fit_source_auth.py','audit_saved.py'):
                self.assertIn(namespace,(CAPTURE/name).read_text())

    def test_07_real_entry_preflight_stays_disabled(self):
        release = CAPTURE/'root_release/RELEASE.json'
        evidence = CAPTURE/'real_evidence'
        before = (release.exists(), evidence.exists())
        result=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'-E','-S','-B',
            str(CAPTURE/'launch.py'),'--approved-release-sha256','0'*64,'--preflight'],
            cwd=str(ROOT),capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,2,result.stderr)
        self.assertEqual(json.loads(result.stdout),dict(status='DISABLED_NO_ROOT_RELEASE',model_work=False))
        self.assertEqual((release.exists(), evidence.exists()), before)

if __name__ == '__main__':
    unittest.main()

"""Mini-specific native reserve guard; model admission remains with the caller.

The five frozen source files retain their historical bytes. This wrapper uses
their full API/ABI, ownership, native-clock, source, and reserve checks and
tightens pressure to NORMAL for both phases. It never owns or extends a budget.
"""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import threading
import time

STAGE = Path(__file__).resolve().parent
POLICY_ID = 'research4.mini.inference_NATIVE_R_NORMAL_only.6GiB_launch_4GiB_run.v1'
METRIC = 'free_plus_pageable_filebacked_estimate_bytes'
TOTAL = 68719476736
PAGE = 16384
SOURCE_NAMES = ('mini_guard.py', 'free_headroom_collector.py',
                'darwin_memory_query.c', 'free_reserve_policy.py',
                'cache_headroom_policy.py', 'legacy_available_guard.py')
HEX = re.compile(r'^[0-9a-f]{64}$')
POLICY = {
    'launch_floor_bytes': 6 * 1024**3,
    'run_floor_bytes': 4 * 1024**3,
    'launch_allowed_pressure_flags': [1],
    'run_allowed_pressure_flags': [1],
    'consumption_max_age_ns': 2_000_000_000,
    'native_pressure_VM_max_span_ns': 250_000_000,
    'native_transport_timeout_seconds': 1.0,
    'legacy_available_is_diagnostic_only': True,
    'no_legacy_available_clearance': True,
    'deadline_owned_by_caller': True,
    'context_capacity_qualified': False,
}
_IMPORT_LOCK = threading.Lock()


def digest(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Regular non-symlink file required')
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            result.update(block)
    return result.hexdigest()


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    def bad(value):
        raise ValueError('Nonfinite JSON value: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)


def safe_local(name):
    if type(name) is not str or '\\' in name or ':' in name:
        raise ValueError('Stage-relative POSIX path required')
    relative = Path(name)
    if relative.is_absolute() or not relative.parts or '..' in relative.parts:
        raise ValueError('Unsafe stage-relative path')
    path = STAGE / relative
    if (not path.resolve().is_relative_to(STAGE) or
            any(part.is_symlink() for part in (path, *path.parents)) or
            'onedrive' in str(path.resolve()).lower()):
        raise ValueError('Outside, symlinked, or OneDrive artifact')
    return path


def bound(entry):
    if (type(entry) is not dict or set(entry) != {'path', 'sha256'} or
            type(entry['sha256']) is not str or not HEX.fullmatch(entry['sha256'])):
        raise ValueError('Exact file binding required')
    path = safe_local(entry['path'])
    if digest(path) != entry['sha256']:
        raise ValueError('Pinned artifact bytes changed: ' + entry['path'])
    return path


def source_pins():
    return {name: digest(safe_local(name)) for name in SOURCE_NAMES}


def validate_compiler_bundle(entry):
    """Authenticate Studio compiler evidence; never call it a Mini SDK install."""
    receipt_path = bound(entry)
    record = strict_json(receipt_path.read_bytes())
    if (record.get('schema') != 'research4.studio-to-mini-native-build.v1' or
            record.get('model_execution') is not False or
            record.get('all_children_reaped') is not True or
            record.get('compiler_environment', {}).get('machine') != 'arm64' or
            record.get('C_source_sha256') != digest(safe_local('darwin_memory_query.c')) or
            record.get('compile', {}).get('returncode') != 0 or
            record.get('compile', {}).get('reaped') is not True or
            'arm64' not in record.get('file', {}).get('stdout', '') or
            record.get('file', {}).get('returncode') != 0 or
            record.get('otool_load_commands', {}).get('returncode') != 0 or
            record.get('otool_libraries', {}).get('returncode') != 0):
        raise ValueError('Actual Studio source/compile/arm64 evidence differs')
    prefix = receipt_path.parent.relative_to(STAGE).as_posix()
    helper = bound({'path': prefix + '/darwin_memory_query',
                    'sha256': record['binary_sha256']})
    expected_headers = record.get('sdk_headers_sha256')
    collector, _ = load_sources()
    if type(expected_headers) is not dict or set(expected_headers) != set(collector.SDK_HEADERS):
        raise ValueError('Exact nine compiler SDK header digests required')
    for name, expected in expected_headers.items():
        bound({'path': prefix + '/sdk_headers/usr/include/' + name, 'sha256': expected})
    return record, helper, prefix + '/sdk_headers'


def load_sources():
    """Load the unchanged policy with isolated dependency names, no native call."""
    def load(name, file):
        spec = importlib.util.spec_from_file_location('_research4_mini_' + name, STAGE / file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    with _IMPORT_LOCK:
        names = ('legacy_available_guard', 'free_reserve_policy')
        previous = {name: sys.modules.get(name) for name in names}
        try:
            legacy = load('legacy', 'legacy_available_guard.py')
            sys.modules['legacy_available_guard'] = legacy
            base = load('base', 'free_reserve_policy.py')
            sys.modules['free_reserve_policy'] = base
            policy = load('policy', 'cache_headroom_policy.py')
        finally:
            for name, value in previous.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value
    return load('collector', 'free_headroom_collector.py'), policy


def assess(policy, raw, consumed_ns, phase, identity, target):
    original = policy.assess(raw, consumed_ns, phase, identity, TOTAL, PAGE, target)
    result = dict(original)
    result.update(policy_id=POLICY_ID, source_policy_assessment=original,
                  allowed_pressure_flags_for_phase=[1], warning_accepted=False,
                  deadline_owned_by_caller=True, context_capacity_qualified=False)
    result.pop('scoped_short_diagnostic_seconds', None)
    flag = original.get('pressure_flag_distinct_from_bytes')
    if original.get('valid') is True and (type(flag) is not int or flag != 1):
        result.update(stop_required=True, reason='native_pressure_not_NORMAL_for_inference',
                      healthy_pressure=False)
    return result


def validate_binding(binding_path, expected_sha256):
    """Authenticate exact candidate binding and original saved consume times.

    No native process, model, lease, or source identity query occurs here.
    Caller must independently review and pin expected_sha256, not discover it
    from an untrusted runtime manifest.
    """
    if type(expected_sha256) is not str or not HEX.fullmatch(expected_sha256):
        raise ValueError('Independently pinned Mini binding SHA256 required')
    path = Path(binding_path)
    if not path.is_absolute():
        path = safe_local(str(path))
    if path.resolve().parent != STAGE or path.is_symlink() or digest(path) != expected_sha256:
        raise ValueError('Mini binding location/hash differs')
    binding = strict_json(path.read_bytes())
    fixed = {'schema': 'research4.mini-native-guard-crossbuilt-binding.v2',
             'status': 'MODEL_FREE_NATIVE_VALIDATED_NOT_MODEL_ADMITTED',
             'policy_id': POLICY_ID, 'policy': POLICY,
             'expected_physical_bytes': TOTAL, 'expected_page_bytes': PAGE,
             'model_execution': False, 'operational_admission': False,
             'sdk_provenance_kind': 'AUTHENTICATED_STUDIO_COMPILER_HEADERS_MIRROR',
             'sdk_installed_on_runtime_host_asserted': False}
    if any(binding.get(key) != value or
           (type(value) in (bool, int) and type(binding.get(key)) is not type(value))
           for key, value in fixed.items()):
        raise ValueError('Mini guard binding scope/policy differs')
    if any(type(binding['policy'].get(key)) is not type(value)
           for key, value in POLICY.items()):
        raise ValueError('Exact policy value types required')
    if binding.get('source_sha256') != source_pins():
        raise ValueError('Mini guard source bytes differ')
    helper = bound(binding['native_helper'])
    compiler, compiled_helper, header_root = validate_compiler_bundle(binding['compiler_provenance'])
    if helper != compiled_helper or binding.get('sdk_header_root') != header_root:
        raise ValueError('Compiler artifact/header-mirror links differ')
    native_identity = binding['source_identity']
    if (native_identity.get('collector_sha256') != binding['source_sha256']['free_headroom_collector.py'] or
            native_identity.get('native_helper_source_sha256') != binding['source_sha256']['darwin_memory_query.c'] or
            native_identity.get('native_helper_sha256') != digest(helper) or
            native_identity.get('os', {}).get('machine') != 'arm64' or
            native_identity.get('psutil_version') != '7.2.2' or
            native_identity.get('sdk_headers_sha256') != compiler['sdk_headers_sha256']):
        raise ValueError('Mini native source/API/platform binding differs')
    evidence = strict_json(bound(binding['evidence']).read_bytes())
    if (evidence.get('model_execution') is not False or evidence.get('operational_admission') is not False
            or evidence.get('source_sha256_before') != binding['source_sha256']
            or evidence.get('source_sha256_after') != binding['source_sha256']
            or evidence.get('all_children_reaped') is not True
            or len(evidence.get('samples', [])) != 3
            or evidence.get('compiler_provenance') != binding['compiler_provenance']
            or evidence.get('sdk_installed_on_runtime_host_asserted') is not False):
        raise ValueError('Actual model-free capture evidence differs')
    _, policy = load_sources()
    for item in evidence['samples']:
        for phase in ('launch', 'run'):
            replay = assess(policy, item['raw'], item['consumed_monotonic_ns'], phase,
                            native_identity, evidence['target'])
            if (replay != item['assessments'][phase] or replay.get('valid') is not True
                    or replay.get('stop_required') is not False):
                raise ValueError('Saved actual native readback failed exact replay')
    return binding


class Guard:
    """A synchronous native guard. Owner supplies lifecycle and absolute deadline."""
    def __init__(self, binding_path, binding_sha256):
        self.binding_path = binding_path
        self.binding_sha256 = binding_sha256
        self.binding = validate_binding(binding_path, binding_sha256)
        if platform.system() != 'Darwin' or platform.machine() != 'arm64':
            raise ValueError('Apple silicon macOS required')
        import psutil
        self.psutil = psutil
        self.collector, self.policy = load_sources()
        self.helper = bound(self.binding['native_helper'])
        self.identity = self.binding['source_identity']
        self.sdk_header_root = safe_local(self.binding['sdk_header_root'])
        actual = self.collector.source_identity(self.helper, self.sdk_header_root, psutil)
        if actual != self.identity:
            raise ValueError('Actual Mini OS/SDK/dependency/helper identity changed')
        self.minimum = {'bytes': None, 'samples': 0}
        self._lock = threading.Lock()

    def sample(self, target, phase):
        with self._lock:
            started = time.monotonic_ns()
            raw = None
            error = None
            consumed = None
            try:
                if (type(target) is not dict or set(target) != {'pid', 'create_time_epoch'} or
                        type(target['pid']) is not int or not 0 < target['pid'] <= 2**31-1 or
                        type(target['create_time_epoch']) not in (int, float) or
                        not math.isfinite(target['create_time_epoch']) or target['create_time_epoch'] <= 0 or
                        phase not in ('launch', 'run')):
                    raise ValueError('Exact live target PID/creation and launch/run required')
                if source_pins() != self.binding['source_sha256']:
                    raise ValueError('Guard sources changed during serving')
                if digest(self.binding_path) != self.binding_sha256:
                    raise ValueError('Guard binding changed during serving')
                raw = self.collector.collect_once(self.helper, target['pid'], self.psutil, self.sdk_header_root)
                consumed = time.monotonic_ns()
                result = assess(self.policy, raw, consumed, phase, self.identity, target)
                if result.get('valid') is True:
                    reserve = result[METRIC]
                    self.minimum['bytes'] = reserve if self.minimum['bytes'] is None else min(reserve, self.minimum['bytes'])
                    self.minimum['samples'] += 1
            except Exception as exc:
                error = {'type': type(exc).__name__, 'message': str(exc)}
                result = {'valid': False, 'stop_required': True, 'policy_id': POLICY_ID,
                          'reason': 'mini_native_guard_exception', 'raw_evidence': {'error': error}}
            finished = time.monotonic_ns()
            if not 0 <= finished - started <= POLICY['consumption_max_age_ns']:
                result = dict(result, valid=False, stop_required=True,
                              reason='mini_whole_sample_exceeded_2_seconds')
            return {'schema': 'research4.inference-native-guard-sample.v1', 'policy_id': POLICY_ID,
                    'phase': phase, 'target': target, 'valid': result.get('valid') is True,
                    'stop_required': result.get('valid') is not True or result.get('stop_required') is not False,
                    'assessment': result, 'raw': raw, 'source': self.identity, 'error': error,
                    'minimum_valid_R': dict(self.minimum), 'deadline_owned_by_caller': True,
                    'context_capacity_qualified': False,
                    'timing': {'sample_started_monotonic_ns': started,
                               'sample_finished_monotonic_ns': finished,
                               'consumed_monotonic_ns': consumed}}


def load_runtime(binding_path, expected_binding_sha256):
    """Caller provides separately reviewed binding SHA; no model authority here."""
    path = Path(binding_path).absolute()
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError('Symlinked binding path prohibited')
    return Guard(path, expected_binding_sha256)

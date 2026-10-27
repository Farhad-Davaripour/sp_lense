"""Read-only, source-bound partial actor evidence; never complete admission."""
from copy import deepcopy
from pathlib import Path
import base64
import hashlib
import math
import re
import tempfile
from urllib.parse import parse_qs, urlsplit
from contextlib import contextmanager

from .io_utils import read, sha, digest
from .composite_bounded_behavior_evaluation_profile import MODE, MANIFEST, STAGE, OWNER
from .composite_bounded_behavior_evaluation_source import validate_source, notebook_code
from .composite_v2_model_audit import EvidenceFiles, frozen_integrity, audit_runtime

_KEY = object()


class PartialEvaluationEvidence:
    def __init__(self, value, key=None):
        if key is not _KEY:
            raise PermissionError('Read the original partial raw archive and frozen source')
        self._key = key
        self._value = deepcopy(value)

    def receipt(self):
        return deepcopy(self._value)


class PartialKaggleClosure:
    def __init__(self, value, key=None):
        if key is not _KEY:
            raise PermissionError('Read actual terminal, release, controller and saved source')
        self._key = key
        self._value = deepcopy(value)


def require_partial(value, *, require_release=True):
    if not isinstance(value, PartialEvaluationEvidence) or value._key is not _KEY:
        raise PermissionError('A summary or complete proof is not partial raw authority')
    report = value.receipt()
    if (report.get('partial_evidence_raw_audited') is not True
            or report.get('complete_partition_raw_audited') is not False
            or report.get('scientific_validation_admitted') is not False
            or report.get('heldout_content_read') is not False
            or (require_release and report.get('owned_compute_released') is not True)):
        raise PermissionError('Source-bound partial diagnostics and actual release are required')
    return report


def _page(block, notebook, run=None, *, release=False):
    lines = block.splitlines()
    browser = next((re.fullmatch(r'Browser tab: \d+, Title: "[^"]*", URL: "([^"]+)"\.', line)
                    for line in lines if line.startswith('Browser tab: ')), None)
    if browser is None:
        raise PermissionError('Primary browser page identity required')
    def route(raw, ax=False):
        if ax and raw.startswith('kaggle.com/'):
            raw = 'https://www.' + raw
        url = urlsplit(raw)
        paths = ['/code/' + notebook + suffix for suffix in ['', '/log', '/output'] + (['/edit'] if release else [])]
        query = parse_qs(url.query, keep_blank_values=True)
        if (url.scheme != 'https' or url.netloc != 'www.kaggle.com' or url.path not in paths
                or url.fragment or (query != {'scriptVersionId': [str(run)]}
                                     and not (release and not query and url.path.endswith('/edit')))):
            raise PermissionError('Exact primary notebook and unique run URL required')
        return url.path, url.query
    expected = route(browser[1])
    pages = [m for line in lines if (m := re.fullmatch(r'\s*\d+ AXWebArea .*, URL: (\S+)\s*', line))]
    if len(pages) != 1 or route(pages[0][1], True) != expected:
        raise PermissionError('Browser and AX primary identities differ')
    return lines


def _terminal_status(text, notebook, run):
    if len(re.findall(r'^Browser tab: ', text, re.M)) != 1:
        raise PermissionError('One exact captured terminal page is required')
    lines = _page(text, notebook, run)
    table = next((i for i, line in enumerate(lines) if re.match(r'^\s*\d+ table(?:\s|$)', line)), len(lines))
    lines = lines[:table]
    logs = [(i, m) for i, line in enumerate(lines) if (m := re.fullmatch(r'(\s*)\d+ heading Logs, Value: 2\s*', line))]
    tabs = [i for i, line in enumerate(lines) if re.fullmatch(
        r'\s*\d+ tab \((?=[^)]*\bselected\b)[^)]*\) Description: Logs, \d+ of \d+, Value: 1\s*', line)]
    if len(logs) != 1 or len(tabs) != 1 or tabs[0] >= logs[0][0]:
        raise PermissionError('Actual selected provider Logs status required')
    headings = [(i, m) for i, line in enumerate(lines)
                if (m := re.fullmatch(r'(\s*)\d+ heading (.+), Value: 5\s*', line))
                and len(m[1].expandtabs(4)) == len(logs[0][1][1].expandtabs(4)) + 4]
    if len(headings) != 1:
        raise PermissionError('One unambiguous provider status heading required')
    index, heading = headings[0]
    if index <= logs[0][0]:
        raise PermissionError('Status must be in the provider Logs heading region')
    label = heading[2]
    duration = r'((?:0|[1-9]\d*)(?:\.\d+)?)s'
    patterns = [('success', r'Successfully ran in ' + duration),
                ('cancelled', r'Version \d+ was cancel(?:ed|led) after ' + duration),
                ('failed', r'(?:Version \d+ (?:was )?failed|Failed(?: to run)?)(?: after| in) ' + duration),
                ('timed_out', r'(?:Version \d+ (?:was )?timed out|Timed out)(?: after| in) ' + duration)]
    matched = [(status, m) for status, pattern in patterns if (m := re.fullmatch(pattern, label))]
    if len(matched) != 1 or not math.isfinite(float(matched[0][1][1])):
        raise PermissionError('Unrecognized, running, negated or nonfinite provider terminal status')
    child = next((line for line in lines[index + 1:] if line.strip()), '')
    value = re.fullmatch(r'(\s*)\d+ text (.+)\s*', child)
    if (value is None or len(value[1].expandtabs(4)) <= len(heading[1].expandtabs(4))
            or ' '.join(value[2].split()) != ' '.join(label.split())):
        raise PermissionError('Provider status heading and its child disagree')
    return matched[0][0]


def kaggle_partial_closure(binding, raw_sha, terminal, release, controller, actual, prepared):
    source = binding.get('source_commit'); endpoint = binding.get('endpoint'); run = binding.get('scriptVersionId')
    notebook = 'farhad89/research4-bounded-actor-' + str(endpoint) + '-v4'
    expected = {'source_commit': source, 'mode': MODE, 'endpoint': endpoint, 'notebook': notebook,
                'phase': 'bounded_actor_' + str(endpoint), 'owner_thread_id': OWNER}
    if (endpoint not in ['behavior', 'competence'] or type(run) is not int or run <= 0
            or any(binding.get(k) != v for k, v in expected.items())
            or not re.fullmatch('[0-9a-f]{40}', source or '')
            or not re.fullmatch('[0-9a-f]{64}', raw_sha or '') or not actual or digest(actual) != digest(prepared)):
        raise PermissionError('Exact consumed partial actor identity and actual saved code required')
    status = _terminal_status(terminal, notebook, run)
    starts = [m.start() for m in re.finditer(r'^Browser tab: ', release, re.M)]
    blocks = [release[a:b] for a, b in zip(starts, starts[1:] + [len(release)])]
    if not blocks:
        raise PermissionError('Actual target provider release capture required')
    pages = [_page(block, notebook, run, release=True) for block in blocks]
    zero = any(re.fullmatch(r'\s*\d+ text 0\s+Active Events\s*', line) for lines in pages for line in lines)
    draft = any(re.fullmatch(r'\s*\d+ (?:button|text) .*Draft Session off(?: .*)?\s*', line)
                for lines in pages for line in lines)
    if not zero or not draft:
        raise PermissionError('Actual zero active events and target draft off observations required')
    if controller is not None:
        if (not isinstance(controller, dict) or any(controller.get(k) != v for k, v in expected.items())
                or digest(controller.get('automatic_scientific_retries')) != digest(0)
                or controller.get('heldout_content_read') is not False):
            raise PermissionError('Controller belongs to another source, owner, phase or policy')
    observed = controller or {}; child = observed.get('child', {})
    if not isinstance(child, dict):
        raise PermissionError('Explicit child observation or unknown is required')
    confirmed = child.get('child_exit_confirmed') is True
    not_started = (confirmed and child.get('child_started') is False
                   and child.get('exit_code') is None and child.get('pid') is None)
    if confirmed and not not_started and (child.get('child_started') is not True or type(child.get('exit_code')) is not int):
        raise PermissionError('Confirmed child exit requires actual start and exit code')
    if confirmed and type(observed.get('exit_code')) is int and observed['exit_code'] != child['exit_code']:
        raise PermissionError('Controller and confirmed child exit codes disagree')
    cleanup = observed.get('raw_prefix_rescue', {})
    if not isinstance(cleanup, dict): raise PermissionError('Cleanup observation must be explicit or unknown')
    lifecycle = ('child_not_started' if not_started else
                 'external_hard_timeout' if observed.get('timed_out') is True else
                 'controlled_clean_exit' if confirmed and child['exit_code'] == 0 else
                 'confirmed_external_child_termination' if confirmed and child['exit_code'] < 0 else
                 'controlled_failed_exit' if confirmed else 'child_exit_unknown')
    return PartialKaggleClosure({**expected, 'provider_run_id': run, 'raw_archive_sha256': raw_sha,
        'provider_terminal_status': status, 'owned_compute_released': True,
        'controller_observation': deepcopy(controller), 'controller_lifecycle': lifecycle,
        'child_exit_confirmed': confirmed, 'child_started': child.get('child_started'), 'owned_scratch_cleanup_verified':
        True if cleanup.get('status') == 'owned_scratch_absent' else None,
        'actual_code_cells_sha256': digest(actual), 'controller_sha256': digest(controller),
        'terminal_sha256': hashlib.sha256(terminal.encode()).hexdigest(),
        'release_sha256': hashlib.sha256(release.encode()).hexdigest()}, _KEY)


def _inventory(files, context):
    stage = context.stage(); roster = {r['case_id']: r for r in context.roster()}; cells = []
    allowed = set(); stopped = False
    for case in stage['partition']['case_ids']:
        for arm in ['base', 'always_on']:
            prefix = 'trajectories/' + case + '/' + arm + '/'
            members = {n for n in files.manifest if n.startswith(prefix)}
            native = {}; responses = {}; terminal = None
            for name in members:
                suffix = name[len(prefix):]
                if suffix == 'TERMINAL.json': terminal = files.json(name)
                elif (m := re.fullmatch(r'(native|response)_(\d{2})\.json', suffix)):
                    (native if m[1] == 'native' else responses)[int(m[2])] = files.json(name)
                else: raise PermissionError('Unregistered partial trajectory member')
            n, r = len(native), len(responses)
            if (sorted(native) != list(range(n)) or sorted(responses) != list(range(r))
                    or not 0 <= r <= n <= 24 or n - r > 1 or (terminal is not None and (not r or n != r))):
                raise ValueError('Partial native/response sequence has gaps, duplicates or impossible coverage')
            if stopped and members:
                raise ValueError('Partial collection is not the registered ordered trajectory prefix')
            for index, value in native.items():
                if (value.get('case_id') != case or value.get('group_id') != roster[case]['group_id']
                        or value.get('arm') != arm or value.get('index') != index
                        or value.get('world_action_not_yet_applied') is not True):
                    raise ValueError('Durable native prefix identity changed')
            if terminal is not None and (terminal.get('scene') != roster[case] or terminal.get('arm') != arm):
                raise ValueError('Partial terminal belongs to another fixed scene/arm')
            cells.append({'case_id': case, 'arm': arm, 'scene': roster[case], 'terminal': terminal,
                          'responses': [responses[i] for i in range(r)], 'prefixes': [native[i] for i in range(n)]})
            allowed.update(members)
            if terminal is None: stopped = True
    if {n for n in files.manifest if n.startswith('trajectories/')} != allowed or any(n.endswith('.pt') for n in files.manifest):
        raise PermissionError('Unaccounted case/arm or training tensor in partial evaluation')
    for name in files.manifest:
        if name.startswith('pairs/'):
            case = name.removeprefix('pairs/').removesuffix('.json')
            terminals = [c['terminal'] for c in cells if c['case_id'] == case]
            pair = files.json(name)
            expected = {'case_id': case, 'outcomes': terminals, 'completed_trajectories': 2,
                        'optimizer_updates': 0, 'teacher_forcing': False, 'teacher_compiler_called': False,
                        'decoder_repair': False, 'scientific_validation_admitted': False, 'heldout_content_read': False}
            if (name != 'pairs/' + case + '.json' or len(terminals) != 2 or None in terminals
                    or any(digest(pair.get(k)) != digest(v) for k, v in expected.items())):
                raise ValueError('Pair completion record lacks both exact durable terminals')
    return cells


def _capacity_records(files, cells):
    result = []
    for name in sorted(n for n in files.manifest if n.startswith('capacity_failures/')):
        match = re.fullmatch(r'capacity_failures/([^/]+)/(base|always_on)/state_(\d{2})\.json', name)
        if match is None: raise PermissionError('Unregistered capacity failure record')
        case, arm, index = match[1], match[2], int(match[3]); value = files.json(name)
        cell = next((c for c in cells if c['case_id'] == case and c['arm'] == arm), None)
        expected = {'case_id': case, 'arm': arm, 'index': index, 'context_ceiling': 6144,
                    'model_forward_attempted': False, 'world_action_applied': False,
                    'context_truncated': False, 'simulator_terminal_invented': False, 'remaining_coverage': 'unknown'}
        first = next((c for c in cells if c['terminal'] is None), None)
        if (cell is None or cell is not first or result or cell['terminal'] is not None or index != len(cell['responses'])
                or len(cell['prefixes']) != len(cell['responses'])
                or not isinstance(value.get('text'), str) or not isinstance(value.get('input_token_ids'), list)
                or len(value['input_token_ids']) <= 6144
                or any(type(v) is not int or v < 0 for v in value['input_token_ids'])
                or any(digest(value.get(k)) != digest(v) for k, v in expected.items())):
            raise ValueError('Capacity record is not an unexecuted unchanged-context failure')
        result.append({'path': name, 'case_id': case, 'arm': arm, 'index': index,
                       'recorded_context_tokens': len(value['input_token_ids']),
                       'model_forward_attempted': False, 'world_action_applied': False,
                       'prompt_tokenization_independently_verified': False, 'terminal_evidence': 'missing_unknown'})
    return result


_BEFORE = {'PRECALL_FREEZE.json', 'HARDWARE_PREFLIGHT.json', 'BACKBONE_LIVE_BEFORE.json',
           'CHECKPOINT_LIVE_BEFORE.json', 'CHECKPOINT_RESTORE.json', 'RUNTIME_RAW.json',
           'ALLOCATOR_BEFORE_COLLECTION.json', 'MODEL_LOAD_ADMISSION.json',
           'ADAPTER_CAPACITY_RUNTIME.json', 'MODEL_FILE_VERIFICATION.json', 'MODEL_METADATA_BYTES.json'}


def _freeze(files, context, root, commit):
    value = files.json('PRECALL_FREEZE.json')
    if (digest(value.get('stage')) != digest(context.stage()) or value.get('source_commit') != commit
            or value.get('data_archive_sha256') != context.stage()['source_data_archive_sha256']
            or value.get('input_manifest_sha256') != sha(Path(root) / 'inputs/MANIFEST.json')
            or digest(value.get('source_release')) != digest(read(Path(root) / 'RELEASE.json'))
            or value.get('raw_training_rows_canonical_sha256') is not None):
        raise ValueError('Original partial raw freeze/source/data/no-training bindings differ')


def _metadata(files, context, root, commit):
    """The complete reader's model/runtime guards, without waiving absent AFTER."""
    from .composite_bounded_behavior_evaluation_audit import _checkpoint
    from .composite_frozen_probe import (verified_model_context, validate_download_proof,
        verify_metadata_bytes, validate_loading_info, audit_loaded_layout)
    from .composite_bounded_behavior_training import adapter_layout
    from .composite_placement import parameter_device
    from .composite_task_memory_profile import verify_allocator_record
    stage = context.stage()
    model = verified_model_context(root, commit, context.model.cfg['qwen'], stage, sha(Path(root) / 'inputs' / STAGE))
    layout = adapter_layout(model['metadata'], model['plan'])
    shapes = {n: v['shape'] for n, v in model['metadata']['tensors'].items() if not n.startswith('mtp.')}
    devices = {n: 'cuda:' + str(parameter_device(n, model['plan'])) for n in shapes}
    hardware = files.json('HARDWARE_PREFLIGHT.json')['devices']
    if (len(hardware) != 2 or any(type(r.get('index')) is not int for r in hardware)
            or {r.get('index') for r in hardware} != {0, 1}):
        raise ValueError('Both actual T4 device records are required')
    for row in hardware:
        lower = model['metadata_bounds']['device_weight_lower_bounds_bytes'][row['index']]
        if (row.get('name') != 'Tesla T4' or row.get('optimistic_weight_lower_bound_bytes') != lower
                or row['free_bytes'] < lower + 2 * 1024**3):
            raise ValueError('Actual separate per-device model admission differs')
    before = files.json('BACKBONE_LIVE_BEFORE.json')
    # Self comparison validates the complete BEFORE snapshot only. It does not
    # establish unchanged post-run bytes, which is reported separately below.
    frozen_integrity(before, before, shapes, parameter_devices=devices)
    audit_runtime(files.json('RUNTIME_RAW.json'), before, shapes,
                  expected_map=model['plan']['device_map'], parameter_devices=devices, require_native_bf16=False)
    _checkpoint(files.json('CHECKPOINT_LIVE_BEFORE.json'), context, layout)
    _checkpoint(files.json('CHECKPOINT_RESTORE.json'), context, layout)
    verify_allocator_record(files.json('ALLOCATOR_BEFORE_COLLECTION.json'))
    load = files.json('MODEL_LOAD_ADMISSION.json'); loading = load['loading']
    validate_loading_info({'missing_keys': loading['missing_keys'], 'mismatched_keys': loading['mismatched_keys'],
        'error_msgs': loading['error_msgs'], 'unexpected_keys': loading['ignored_checkpoint_only_MTP_keys']})
    audit_loaded_layout(load['layout'], model, {r['name']: r for r in before['parameters']})
    capacity = files.json('ADAPTER_CAPACITY_RUNTIME.json')
    if (capacity['adapter_parameters'] != 233455616 or capacity['projection_count'] != 496
            or digest(capacity['covered_blocks']) != digest(list(range(64)))):
        raise ValueError('Complete frozen all64-block FP32 adapter required')
    storage = capacity['per_device_storage_admission']['devices']
    if (len(storage) != 2 or any(type(r.get('device')) is not int for r in storage)
            or {r['device'] for r in storage} != {0, 1}):
        raise ValueError('Both actual frozen adapter device budgets are required')
    for row in storage:
        if (row['adapter_weight_bytes'] != layout['device_parameters'][row['device']] * 4
                or row['reserve_bytes'] != 2 * 1024**3
                or row['free_bytes'] < row['adapter_weight_bytes'] + row['reserve_bytes']):
            raise ValueError('Frozen inference adapter storage/reserve differs')
    proof = files.json('MODEL_FILE_VERIFICATION.json'); encoded = files.json('MODEL_METADATA_BYTES.json')
    validate_download_proof(model, proof); verify_metadata_bytes(model['model_pin'], proof, encoded)
    after = {'backbone': None, 'checkpoint': None, 'allocator': None}
    if 'BACKBONE_LIVE_AFTER.json' in files.manifest:
        frozen_integrity(before, files.json('BACKBONE_LIVE_AFTER.json'), shapes, parameter_devices=devices)
        after['backbone'] = True
    if 'CHECKPOINT_LIVE_AFTER.json' in files.manifest:
        _checkpoint(files.json('CHECKPOINT_LIVE_AFTER.json'), context, layout); after['checkpoint'] = True
    if 'ALLOCATOR_AFTER_COLLECTION.json' in files.manifest:
        verify_allocator_record(files.json('ALLOCATOR_AFTER_COLLECTION.json')); after['allocator'] = True
    for name in files.manifest:
        if name.startswith('pairs/'):
            case = name.removeprefix('pairs/').removesuffix('.json')
            if name != 'pairs/' + case + '.json' or case not in stage['partition']['case_ids']:
                raise PermissionError('Unregistered pair checkpoint evidence')
            pair = files.json(name)
            _checkpoint(pair['live_checkpoint_before'], context, layout)
            _checkpoint(pair['live_checkpoint_after'], context, layout)
    return model, encoded, after


@contextmanager
def _tokenizer(encoded):
    with tempfile.TemporaryDirectory(prefix='r4-partial-actor-tokenizer-') as folder:
        target = Path(folder).resolve()
        for name, value in encoded.items():
            path = (target / name).resolve()
            if not path.is_relative_to(target):
                raise PermissionError('Tokenizer metadata escaped owned temporary storage')
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(base64.b64decode(value, validate=True))
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(target, local_files_only=True, trust_remote_code=False)
        generation = read(target / 'generation_config.json'); eos = set()
        for value in [generation.get('eos_token_id'), tokenizer.eos_token_id]:
            if value is not None: eos.update(value if isinstance(value, list) else [value])
        yield tokenizer, eos


def _replay_prefix(archive, scene, cell, tokenizer, eos, vocabulary, previous_cache):
    """Apply only durable response actions; never apply a trailing native entry."""
    from .composite_bounded_behavior_evaluation_runtime import replay_scope
    from .composite_bounded_behavior_evaluation_collection import world_for
    from .composite_v2_trace_audit import _native
    from . import composite_bounded_behavior_interface as interface
    arm = cell['arm']; responses = cell['responses']; prefixes = cell['prefixes']; forwards = 0
    with replay_scope(archive, scene, arm, [r['submitted_action'] for r in responses]):
        world = world_for(scene, archive.context.stage()['endpoint'])
        for index, prefix in enumerate(prefixes):
            if not world.running: raise ValueError('Native prefix follows a stopped trajectory')
            visible = world.visible(); text = visible[0] if isinstance(visible, tuple) else visible
            expected = {'case_id': scene.case_id, 'group_id': scene.group_id, 'arm': arm, 'index': index,
                        'text': text, 'native': prefix['native'], 'world_action_not_yet_applied': True}
            if digest(prefix) != digest(expected): raise ValueError('Incomplete native-before-action prefix differs')
            previous_cache = _native(prefix['native'], text, arm == 'always_on', tokenizer, eos,
                                     vocabulary, previous_cache, interface=interface)
            forwards += len(prefix['native']['generated_token_ids'])
            if index == len(responses):
                # The action may or may not have happened before interruption.
                # It is never applied by this reader without a durable response.
                break
            response = responses[index]; parsed = prefix['native']['actions']
            action = None if parsed is None else parsed[0]
            if digest(action) != digest(response['submitted_action']):
                raise ValueError('Incomplete native output was repaired or changed')
            clock = world.clock; reply = world.step(action)
            expected = {'case_id': scene.case_id, 'group_id': scene.group_id, 'arm': arm, 'index': index,
                        'scene': cell['scene'], 'text': text, 'native': prefix['native'], 'submitted_action': action,
                        'reply': reply, 'clock_before': clock, 'clock_after': world.clock, 'outcome': world.outcome()}
            if digest(response) != digest(expected): raise ValueError('Incomplete durable response replay differs')
    return forwards, previous_cache


def _counts(files, stage, receipt, verified_forwards, closure):
    maximum = stage['maximum_backbone_forwards']; observed = []
    if not 0 <= verified_forwards <= maximum: raise ValueError('Verified native forwards exceed fixed ceiling')
    def check(value):
        if (not isinstance(value, dict) or set(value) != {'attempted', 'completed'}
                or any(type(v) is not int for v in value.values())
                or not 0 <= value['completed'] <= value['attempted'] <= maximum):
            raise ValueError('Invalid partial forward count evidence')
        return value
    progress = sorted(n for n in files.manifest if n.startswith('forward_progress/'))
    for name in progress:
        value = files.json(name); m = re.fullmatch(r'forward_progress/(\d{6})\.json', name)
        count = check(value.get('counts_lower_bound'))
        if not m or int(m[1]) != count['completed'] or value.get('exact_at_exit') is not False:
            raise ValueError('Partial forward progress identity/exactness differs')
        if observed and any(count[k] < observed[-1][k] for k in count):
            raise ValueError('Recorded partial forward progress went backwards')
        observed.append(count)
    recorded = None if receipt is None else receipt.get('forward_counts')
    if recorded is not None:
        recorded = check(recorded)
        if receipt.get('counts_exact_at_controlled_exit') is True and observed and any(recorded[k] < observed[-1][k] for k in recorded):
            raise ValueError('Terminal forward counts precede recorded progress')
        observed.append(recorded)
    if (recorded is not None and receipt.get('counts_exact_at_controlled_exit') is True
            and recorded['completed'] < verified_forwards):
        raise ValueError('Recorded counts omit independently verified native forwards')
    child_confirmed = bool(closure and closure._value['child_exit_confirmed'] and closure._value['child_started'] is True)
    exact = bool(recorded is not None and receipt and receipt.get('worker_exiting') is True
                 and receipt.get('counts_exact_at_controlled_exit') is True and child_confirmed)
    return {'attempted_lower_bound': max([verified_forwards] + [v['attempted'] for v in observed]),
            'completed_lower_bound': max([verified_forwards] + [v['completed'] for v in observed]),
            'native_forwards_independently_verified': verified_forwards, 'recorded_terminal_counts': recorded,
            'counts_exact_at_controlled_exit': exact, 'unlogged_calls_absence_proved': False}


def audit_partial_partition(raw, raw_sha, root, commit, fit_archive, qualification_archive,
                            data_archive, alias_fit_archive, alias_qualification_archive, closure=None):
    context = validate_source(root, commit, fit_archive, qualification_archive, data_archive,
                              alias_fit_archive, alias_qualification_archive)
    stage = context.stage(); files = EvidenceFiles(raw, raw_sha, stage['maximum_raw_evidence_bytes'])
    try:
        if closure is not None:
            if not isinstance(closure, PartialKaggleClosure) or closure._key is not _KEY:
                raise PermissionError('A complete/serialized closure cannot establish partial release')
            value = closure._value
            if (value['source_commit'] != commit or value['raw_archive_sha256'] != raw_sha
                    or value['notebook'] != stage['notebook'] or value['endpoint'] != stage['endpoint']
                    or value['actual_code_cells_sha256'] != digest([notebook_code(root, commit, sha(Path(root).parent / MANIFEST))])):
                raise PermissionError('Partial provider/raw/saved-source identity differs')
        from .composite_bounded_behavior_evaluation_resources import validate_controller_resource
        controller_resource = validate_controller_resource(stage,
            closure._value['controller_observation'] if closure else None, allow_unknown=True)
        receipt = files.json('RUN_RECEIPT.json') if 'RUN_RECEIPT.json' in files.manifest else None
        if receipt is not None:
            expected = {'mode': MODE, 'source_commit': commit, 'phase': stage['phase'],
                        'heldout_content_read': False, 'automatic_scientific_retries': 0, 'public_fixture_test_mode': False}
            if any(digest(receipt.get(k)) != digest(v) for k, v in expected.items()):
                raise PermissionError('Partial worker source/phase/policy differs')
            for name, target in [('optimizer_updates', 0), ('model_instances', 1)]:
                if receipt.get(name) is not None and (type(receipt[name]) is not int or receipt[name] != target):
                    raise PermissionError('Partial actor cannot update weights or instantiate another model')
        if 'PRECALL_FREEZE.json' in files.manifest: _freeze(files, context, root, commit)
        cells = _inventory(files, context); capacity = _capacity_records(files, cells)
        missing_before = sorted(_BEFORE - set(files.manifest))
        audited = []; prefixes = []; forwards = 0; after = {'backbone': None, 'checkpoint': None, 'allocator': None}
        if not missing_before:
            model, encoded, after = _metadata(files, context, root, commit)
            from .composite_bounded_behavior_evaluation_audit import (EvaluationArchive, _ARCHIVE_KEY, replay_trajectory)
            from .composite_bounded_behavior_evaluation_collection import scene_for
            records = {c['case_id'] + '/' + c['arm']: {'scene': c['scene'], 'responses': c['responses']} for c in cells}
            archive = EvaluationArchive(context, records, _ARCHIVE_KEY); previous = None
            with _tokenizer(encoded) as (tokenizer, eos):
                for cell in cells:
                    if cell['terminal'] is not None:
                        scene = scene_for(cell['scene'], stage['endpoint'])
                        value, count, previous = replay_trajectory(archive, scene, cell['arm'], cell['terminal'],
                            cell['responses'], cell['prefixes'], tokenizer, eos, model['metadata']['tensors']['lm_head.weight']['shape'][0], previous)
                        audited.append(value)
                    elif cell['prefixes']:
                        scene = scene_for(cell['scene'], stage['endpoint'])
                        count, previous = _replay_prefix(archive, scene, cell, tokenizer, eos,
                            model['metadata']['tensors']['lm_head.weight']['shape'][0], previous)
                    else: count = 0
                    forwards += count
        for cell in cells:
            if cell['terminal'] is None and cell['prefixes']:
                prefixes.append({'case_id': cell['case_id'], 'arm': cell['arm'],
                    'recorded_responses': len(cell['responses']), 'recorded_native_prefixes': len(cell['prefixes']),
                    'native_only_trailing_entry': len(cell['prefixes']) > len(cell['responses']),
                    'trailing_native_world_application': 'unknown' if len(cell['prefixes']) > len(cell['responses']) else None,
                    'prefix_native_and_response_replay_verified': not missing_before,
                    'terminal_evidence': 'missing_unknown', 'scored': False})
        matched = {case for case in stage['partition']['case_ids']
                   if {v['arm'] for v in audited if v['case_id'] == case} == {'base', 'always_on'}}
        from .composite_bounded_behavior_evaluation_scoring import summarize_behavior, summarize_competence
        scorer = summarize_behavior if stage['endpoint'] == 'behavior' else summarize_competence
        cfg = read(Path(root) / 'config/composite_bounded_behavior_v4.json')
        counts = _counts(files, stage, receipt, forwards, closure)
        unknown = (['before_model_integrity_unverified:' + n for n in missing_before]
                   + ['after_' + k + '_integrity_unknown' for k, v in after.items() if v is not True]
                   + (['capacity_prompt_tokenization_unverified'] if capacity else [])
                   + ([] if closure else ['actual_provider_release_not_verified']))
        return PartialEvaluationEvidence({'reader': 'bounded-actor-partial-native-raw-v1',
            'source_commit': commit, 'endpoint': stage['endpoint'], 'partition_index': stage['partition']['index'],
            'raw_archive_sha256': raw_sha, 'checkpoint_sha256': stage['checkpoint_sha256'],
            'source_data_archive_sha256': stage['source_data_archive_sha256'],
            'frozen_model_sha256': stage['frozen_model_sha256'], 'training_lineage': stage['training_lineage'],
            'roster_sha256': stage['roster_sha256'], 'configuration_sha256': stage['configuration_sha256'],
            'partial_evidence_raw_audited': True, 'complete_partition_raw_audited': False,
            'owned_compute_released': True if closure else None,
            'provider_lifecycle': deepcopy(closure._value) if closure else None,
            'controller_resource_validation': controller_resource,
            'resource_amendment_version': stage.get('resource_amendment_version'),
            'resource_policy_sha256': stage.get('resource_policy_sha256'),
            'worker_status': receipt.get('status') if receipt else None,
            'worker_failure': {k: receipt.get(k) for k in ['error_type', 'error', 'model_or_cache_release_error']} if receipt else None,
            'failure_record_sha256': files.manifest.get('FAILURE.json'),
            'recorded_optimizer_updates': receipt.get('optimizer_updates') if receipt else None,
            'recorded_terminal_trajectories': sum(c['terminal'] is not None for c in cells),
            'completed_terminal_trajectories': len(audited), 'complete_matched_pairs': len(matched),
            'audited_terminals': audited, 'incomplete_prefixes': prefixes, 'unexecuted_capacity_records': capacity,
            'completed_arm_descriptive_summary': scorer(context.roster(), audited, cfg, complete_required=False),
            'matched_pair_descriptive_summary': scorer(context.roster(), [v for v in audited if v['case_id'] in matched], cfg, complete_required=False),
            'forward_counts': counts, 'before_integrity_verified': not missing_before,
            'after_integrity': after, 'full_scientific_integrity_proved': False,
            'missing_or_unverified': unknown, 'new_model_forwards': 0, 'heldout_content_read': False,
            'scientific_validation_admitted': False, 'population_reliability_admitted': False,
            'learned_dynamic_or_sealed_admitted': False, 'successor_admission_permitted': False,
            'interpretation': 'Completed recorded trajectories only; missing coverage and missing after-integrity remain unknown.'}, _KEY)
    finally:
        files.close()

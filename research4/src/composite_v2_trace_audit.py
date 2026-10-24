"""Read-only replay of V2 evaluation evidence; no model, teacher or optimizer.

A replay pass establishes consistency of recorded evidence, not frozen tensor
integrity, causal helper benefit, scientific admission or learned compliance.
Only verified tokenizer metadata can enable the raw-native evidence flag.
Public fixtures always remain engineering evidence.
"""
from contextlib import contextmanager
from dataclasses import asdict
import re

from .composite_audit import audit_observation
from .composite_generalized_interface import PROFILE, input_ids, parse
from .composite_generalized_world import GeneralizedScene, World
from .composite_memory import EpisodeMemory
from .composite_v2_controller import CONDITIONS, ENVIRONMENT
from .io_utils import digest
from .native27_tools import decode


_TOKENIZER_KEY = object()
GENERATION = {'max_new_tokens':160, 'do_sample':False, 'num_beams':1,
              'num_return_sequences':1, 'use_cache':True}


def _same(actual, expected, message):
    # JSON hashes distinguish bool from int; Python equality does not.
    if digest(actual) != digest(expected):
        raise ValueError(message)


class TraceTokenizers:
    """Opaque binding created only by the original metadata verifier below."""
    def __init__(self, detector, actor, eos, vocabulary, proof, key=None):
        if key is not _TOKENIZER_KEY:
            raise PermissionError('Use the verified tokenizer metadata reader')
        self.detector = detector
        self.actor = actor
        self.eos = frozenset(eos)
        self.vocabulary = vocabulary
        self.proof = proof
        self._key = key


@contextmanager
def verified_trace_tokenizers(files, context, package, source):
    """Reconstruct pinned small tokenizer files; never load model weights.

    The shared reader verifies the complete original download manifest and the
    exact metadata bytes, including both tokenizer pins, with remote code off.
    The caller still owns source/package and scientific execution admission.
    """
    from .composite_frozen_metadata import verified_tokenizers
    vocabulary = context['metadata']['tensors']['lm_head.weight']['shape'][0]
    if type(vocabulary) is not int or vocabulary <= 0:
        raise ValueError('Pinned vocabulary geometry required')
    with verified_tokenizers(files, context, package, source) as (bert, qwen, eos):
        proof = {'model_file_manifest_sha256':digest(files.json('MODEL_FILE_VERIFICATION.json')),
                 'detector_file_manifest_sha256':digest(files.json('DETECTOR_FILE_VERIFICATION.json')),
                 'metadata_bytes_verified':True, 'model_weights_loaded':False}
        yield TraceTokenizers(bert, qwen, eos, vocabulary, proof, _TOKENIZER_KEY)


def _native(response, prompt, selected_gate, actor, eos, vocabulary, previous_cache):
    """Audit the complete recorded native call, including invalid model output."""
    _same(response.get('generation_parameters'), GENERATION, 'Native generation recipe differs')
    if response.get('native_interface_profile') != PROFILE or response.get('kernel_profile') != 'reference':
        raise ValueError('Native interface or kernel profile differs')
    if type(response.get('adapter_enabled')) is not bool or response['adapter_enabled'] is not selected_gate:
        raise ValueError('Native adapter gate differs')
    ids = response['generated_token_ids']
    if (not isinstance(ids, list) or not 0 < len(ids) <= 160
        or any(type(token) is not int or not 0 <= token < vocabulary for token in ids)
        or any(token in eos for token in ids[:-1])
        or (ids[-1] not in eos and len(ids) != 160)):
        raise ValueError('Native token range, EOS or full generation budget differs')
    expected = input_ids(actor, prompt)
    if len(expected) > 6144:
        raise ValueError('Executed actor context exceeded the unchanged ceiling')
    _same(response['input_token_ids'], expected, 'Native prompt tokens differ')
    raw = decode(actor, ids, eos)
    if response['raw_text'] != raw:
        raise ValueError('Native token decode differs')
    ended = ids[-1] in eos
    if type(response.get('finished_with_eos')) is not bool or response['finished_with_eos'] is not ended:
        raise ValueError('Native EOS receipt differs')
    try:
        actions = parse(raw) if ended else None
    except (ValueError, TypeError, RecursionError):
        actions = None
    _same(response['actions'], actions, 'Native action parsing differs; no repair is permitted')
    cache = response['cache']
    identity = cache.get('cache_identity')
    previous = cache.get('previous_cache_identity')
    alive = cache.get('previous_cache_alive')
    if (cache.get('cache_object_present') is not True or cache.get('prefill_checked') is not True
        or type(cache.get('initial_cache_length')) is not int or cache['initial_cache_length'] != 0
        or cache.get('previous_cache_reused') is not False
        or type(identity) is not int or identity <= 0 or type(alive) is not bool
        or alive != (previous is not None)
        or (alive and (type(previous) is not int or previous <= 0 or previous == identity))
        or (alive and previous_cache is not None and previous != previous_cache)):
        raise ValueError('Fresh response cache provenance differs')
    return identity


def _observation(world, value, memory, detector):
    audit_observation(world, value, memory, 0.9, detector, None if detector is None else 2048)
    text, span = world.visible()
    _same(value['evaluator'], world.truth(), 'Evaluator labels or boolean types differ')
    _same(value['evidence_chars'], list(span), 'Evidence span types differ')
    _same(value['sequence'], world.sequence, 'Observation sequence types differ')
    _same(value['memory'], memory.snapshot(), 'Memory field types or state differ')
    if value['text'] != text:
        raise ValueError('Observation text differs')


def audit_trace(record, expected_scene, source_commit, tokenizers=None, *,
                public_fixture=False, fixture_tokenizers=None, require_terminal=True):
    """Replay recorded nonsealed actions, without choosing or generating any.

    Passing no tokenizer evidence is useful for structural failure diagnosis;
    it never yields raw_evidence_audited=True. A prefix remains an incomplete
    trajectory even if its saved boundary and all prior steps replay exactly.
    """
    if not isinstance(source_commit, str) or not re.fullmatch('[0-9a-f]{40}', source_commit):
        raise ValueError('Exact source commit required')
    if type(require_terminal) is not bool:
        raise ValueError('Explicit complete or prefix audit mode required')
    if not isinstance(expected_scene, GeneralizedScene) or expected_scene.role == 'sealed':
        raise PermissionError('This reader never opens or replays held-out cases')
    _same(record['scene'], asdict(expected_scene), 'Scene differs from the admitted nonsealed roster')
    _same(record.get('environment'), ENVIRONMENT, 'V2 environment differs')
    if record.get('condition') not in CONDITIONS:
        raise ValueError('Unregistered comparison arm')
    if (record.get('model_training_during_evaluation') is not False
        or record.get('teacher_compiler_called_during_evaluation') is not False):
        raise ValueError('Evaluation must explicitly exclude model fitting and teacher calls')
    if type(public_fixture) is not bool or (expected_scene.role == 'public_design_fixture') is not public_fixture:
        raise PermissionError('Public fixtures cannot be relabeled as scientific evidence')
    if fixture_tokenizers is not None and (not public_fixture or tokenizers is not None):
        raise PermissionError('Injected tokenizers are only for explicit public unit fixtures')
    verified = isinstance(tokenizers, TraceTokenizers) and tokenizers._key is _TOKENIZER_KEY
    if tokenizers is not None and not verified:
        raise PermissionError('Opaque verified tokenizer evidence required')
    if fixture_tokenizers is not None:
        detector, actor, eos, vocabulary = fixture_tokenizers
    elif verified:
        detector, actor, eos, vocabulary = (tokenizers.detector, tokenizers.actor,
                                           tokenizers.eos, tokenizers.vocabulary)
    else:
        detector = actor = None
        eos = vocabulary = None
    if actor is not None and (not eos or type(vocabulary) is not int or vocabulary <= 0
                             or any(type(v) is not int or not 0 <= v < vocabulary for v in eos)):
        raise ValueError('Exact native vocabulary and EOS set required')
    # World enforces Databricks-only execution for any actual scientific scene.
    world = World(expected_scene)
    memory = EpisodeMemory()
    epoch = record['initial_observation']['memory']['cache_epoch']
    if type(epoch) is not int or epoch < 2:
        raise ValueError('Invalid initial cache epoch')
    memory.cache_epoch = epoch - 2
    memory.reset(expected_scene.case_id, world.worker, world.peer)
    _observation(world, record['initial_observation'], memory, detector)
    initial = memory.applied_gate
    previous_cache = None
    trace = record['trace']
    if not isinstance(trace, list) or len(trace) > expected_scene.hard_stop + 1:
        raise ValueError('Recorded episode exceeds the unchanged external bound')
    for step in trace:
        if not world.running:
            raise ValueError('Recorded model call after external terminal')
        response = step['response']
        gate = {'base':False, 'always_on':True, 'initial_gate':initial,
                'learned_dynamic':memory.applied_gate, 'oracle_dynamic':world.truth()['gate']}[record['condition']]
        memory.begin_response()
        _same(response['memory_before_response'], memory.snapshot(), 'Response memory differs')
        for field, expected in [('selected_gate',gate), ('adapter_enabled',gate),
                                ('predicted_gate',memory.raw_gate), ('evaluator_gate',world.truth()['gate'])]:
            if type(response.get(field)) is not bool or response[field] is not expected:
                raise ValueError('Response routing differs')
        if actor is not None:
            previous_cache = _native(response, world.visible()[0], gate, actor, eos, vocabulary, previous_cache)
        elif (response['cache'].get('prefill_checked') is not True
              or type(response['cache'].get('initial_cache_length')) is not int
              or response['cache']['initial_cache_length'] != 0):
            raise ValueError('Response cache boundary differs')
        actions = response['actions']
        if actions is not None and (not isinstance(actions, list) or len(actions) != 1):
            raise ValueError('One native action or an invalid-response marker required')
        action = actions[0] if actions else None
        _same(step['executed_action'], action, 'Evaluator replaced the recorded native action')
        memory.end_response()
        _same(step['reply'], world.step(action), 'Tool reply replay differs')
        _observation(world, step['observation_after'], memory, detector)
        _same(step['outcome'], world.outcome(), 'Per-step outcome replay differs')
    if world.running and actor is not None and len(input_ids(actor, world.visible()[0])) > 6144:
        world.running = False
        world.reason = 'external_context_budget'
        memory.terminal = True
        memory.applied_gate = False
    if require_terminal and world.running:
        raise ValueError('Incomplete trajectory cannot be labeled complete')
    _same(record['outcome'], world.outcome(), 'Terminal outcome replay differs')
    _same(record['terminal_memory'], memory.snapshot(), 'Terminal memory replay differs')
    for name, value in [('primitive_trace',world.engine.trace), ('helper_definitions',world.engine.definitions),
                        ('helper_calls',world.helper_calls)]:
        _same(record.get(name), value, 'Raw '+name+' replay differs')
    complete = not world.running
    return {'case_id':expected_scene.case_id, 'condition':record['condition'],
            'trace_sha256':digest(record), 'source_commit':source_commit,
            'native_interface_profile':PROFILE, 'responses':len(trace),
            'trajectory_replayed':True, 'trajectory_complete':complete,
            'raw_evidence_audited':bool(require_terminal and complete and verified and not public_fixture),
            'tokenizer_metadata_verified':bool(verified),
            'tokenizer_proof':tokenizers.proof if verified else None,
            'native_tokens_replayed':actor is not None,
            'public_fixture_test_mode':public_fixture, 'prefix_only':not require_terminal,
            'model_tensor_integrity_audited':False, 'helper_causal_benefit_audited':False,
            'model_behavior_success':None, 'scientific_validation_admitted':False,
            'external_enforcement_is_not_learned_compliance':True,
            'model_forwards':0, 'optimizer_updates':0, 'heldout_content_read':False}


def audit_partition(records, scenes, source_commit, tokenizers=None, *,
                    conditions=CONDITIONS, public_fixture=False, fixture_tokenizers=None):
    """Exact coverage of supplied nonsealed roster; absent traces stay unknown."""
    if (not scenes or any(not isinstance(s, GeneralizedScene) or s.role == 'sealed' for s in scenes)
        or len({s.case_id for s in scenes}) != len(scenes)):
        raise ValueError('Exact unique nonsealed roster required')
    if not conditions or len(set(conditions)) != len(conditions) or not set(conditions) <= set(CONDITIONS):
        raise ValueError('Only the original unique comparison arms are allowed')
    lookup = {s.case_id:s for s in scenes}
    expected = {(s.case_id,a) for s in scenes for a in conditions}
    proofs = {}
    for record in records:
        case = record['scene']['case_id']
        key = (case, record['condition'])
        if key not in expected or '/'.join(key) in proofs:
            raise ValueError('Duplicate or foreign evaluation trace')
        proofs['/'.join(key)] = audit_trace(record, lookup[case], source_commit, tokenizers,
            public_fixture=public_fixture, fixture_tokenizers=fixture_tokenizers)
    missing = sorted('/'.join(key) for key in expected if '/'.join(key) not in proofs)
    return {'raw_audits':proofs, 'expected_trajectories':len(expected),
            'observed_trajectories':len(proofs), 'missing':missing,
            'complete':not missing,
            'raw_evidence_audited':not missing and all(p['raw_evidence_audited'] for p in proofs.values()),
            'missing_behavior_results':'unknown', 'scientific_validation_admitted':False,
            'model_forwards':0, 'optimizer_updates':0, 'heldout_content_read':False}

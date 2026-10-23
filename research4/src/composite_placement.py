"""Inert two-device frozen-model planning and raw activation audit helpers.

No model loader, forward pass, optimizer, GPU allocation, or execution authority
is supplied here. The existing scientific worker does not admit this profile.
"""
import math
import re
import hashlib
import json

PROFILE = 'qwen-two-t4-frozen-candidate-v1'
INDEX_SHA256 = 'b3737e9d00bda0e37f0b873629d98bf9b407bef35735b9193c23d9844bcc96a6'
METADATA_SHA256 = 'a2477ac2c15f9c198d5137204f00da050d8aa87716a3cc5f0156f00b84034f7f'


def candidate_plan(config, profile):
    expected = {'repository': 'Qwen/Qwen3.5-27B',
                'revision': 'fc05daec18b0a78c049392ed2e771dde82bdf654',
                'layers': 64, 'hidden_size': 5120, 'rank': 32, 'alpha': 64,
                'maximum_tokens': 6144, 'maximum_new_tokens': 160,
                'quantization': 'nf4-double-bfloat16'}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError('Two-device proposal may not alter the frozen model or capacity')
    if profile['version'] != PROFILE or profile['status'] != 'engineering_proposal_not_runtime_admitted':
        raise ValueError('Unregistered candidate placement')
    for name in ['repository', 'revision', 'layers', 'hidden_size', 'maximum_tokens', 'maximum_new_tokens', 'quantization']:
        if profile[name] != expected[name]:
            raise ValueError('Placement profile differs from the original frozen model')
    if profile['first_device_layer_count'] != 18 or profile['devices'] != [0, 1]:
        raise ValueError('Unreviewed device partition')
    if profile['model_index_sha256'] != INDEX_SHA256 or profile['tensor_metadata_sha256'] != METADATA_SHA256:
        raise ValueError('Pinned complete tensor-header metadata changed')
    if profile['retained_adapter_capacity'] != {'scope': 'all_blocks', 'rank': 32, 'alpha': 64}:
        raise ValueError('Adapter capacity may not be reduced')
    if profile['scope'] != 'frozen_base_activation_analysis_only' or profile['kernel_profile'] != 'reference':
        raise ValueError('Frozen analysis scope or kernels changed')
    if profile['generation'] != {'do_sample': False, 'num_beams': 1, 'num_return_sequences': 1, 'use_cache': True}:
        raise ValueError('Generation semantics changed')
    if profile['optimizer_updates'] != 0 or profile['behavior_adapter_fitting'] or profile['heldout_content_read'] or profile['CPU_or_disk_offload']:
        raise PermissionError('Candidate profile does not authorize fitting, held-out access or offload')
    devices = {'model.visual': 0, 'model.language_model.embed_tokens': 0,
               'model.language_model.rotary_emb': 0, 'model.language_model.norm': 1,
               'lm_head': 0}
    devices.update({'model.language_model.layers.' + str(index): 0 if index < 18 else 1
                    for index in range(64)})
    return {'profile': PROFILE, 'repository': expected['repository'], 'revision': expected['revision'],
            'hidden_size': 5120, 'layers': 64, 'device_map': devices,
            'model_config_sha256': profile['model_config_sha256'],
            'model_index_sha256': profile['model_index_sha256'],
            'tensor_metadata_sha256': profile['tensor_metadata_sha256'],
            'runtime_admitted': False, 'numerical_parity_admitted': False,
            'scientific_validation_admitted': False, 'compute_submissions': 0}


def parameter_device(name, plan):
    choices = [(prefix, device) for prefix, device in plan['device_map'].items()
               if name == prefix or name.startswith(prefix + '.')]
    if not choices:
        raise ValueError('Unregistered runtime parameter: ' + name)
    device = max(choices, key=lambda value: len(value[0]))[1]
    if type(device) is not int or device not in [0, 1]:
        raise ValueError('CPU/disk/automatic placement is not admitted')
    return device


def metadata_bounds(metadata, plan):
    if metadata['repository'] != plan['repository'] or metadata['revision'] != plan['revision']:
        raise ValueError('Tensor metadata belongs to another frozen model')
    if metadata['config_sha256'] != plan['model_config_sha256'] or metadata['tensor_values_read']:
        raise ValueError('Expected pinned header-only metadata')
    fingerprint = hashlib.sha256(json.dumps(metadata['tensors'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if metadata['index_sha256'] != plan['model_index_sha256'] or fingerprint != plan['tensor_metadata_sha256']:
        raise ValueError('Complete tensor metadata fingerprint differs')
    lower = [0.0, 0.0]
    names = []
    ignored = []
    layers = set()
    unquantized = set()
    for name, value in metadata['tensors'].items():
        if name.startswith('mtp.'):
            ignored.append(name)  # The pinned reference class already ignores ^mtp.*.
            continue
        if value['dtype'] not in ['BF16', 'F32'] or not value['shape'] or any(type(n) is not int or n <= 0 for n in value['shape']):
            raise ValueError('Unexpected pinned tensor geometry or dtype')
        device = parameter_device(name, plan)
        match = re.match(r'model\.language_model\.layers\.(\d+)\.', name)
        if match:
            layers.add(int(match.group(1)))
        full = name.endswith('embed_tokens.weight') or name.endswith('lm_head.weight')
        if full:
            unquantized.add(name)
        # Optimistic lower bound, never a runtime admission: every tensor except
        # embeddings/output is assumed exactly four bits, even norms/SSM state.
        lower[device] += math.prod(value['shape']) * (2 if full else 0.5)
        names.append(name)
    if layers != set(range(64)) or len(unquantized) != 2:
        raise ValueError('Incomplete full-model parameter coverage')
    return {'profile': PROFILE, 'runtime_parameter_names': sorted(names),
            'ignored_checkpoint_only_MTP_keys': sorted(ignored),
            'device_weight_lower_bounds_bytes': lower,
            'lower_bound_excludes': ['NF4 metadata', 'normally unquantized norms/SSM tensors',
                                     'adapters', 'activations/caches', 'CUDA workspaces', 'detector'],
            'runtime_memory_admitted': False, 'model_forwards': 0}


def audit_frozen_activations(record, plan, metadata, expected_input_ids):
    """Check exported raw data only; this establishes no generation parity."""
    bounds = metadata_bounds(metadata, plan)
    if record['profile'] != plan['profile'] or record['revision'] != plan['revision']:
        raise ValueError('Activation record belongs to another profile/model')
    if (record['model_instances'], record['optimizer_updates'], record['trainable_parameters']) != (1, 0, 0):
        raise ValueError('Frozen analysis requires one untrained backbone')
    if record['heldout_content_read'] or record['kernel_profile'] != 'reference':
        raise PermissionError('Held-out or unreviewed kernel execution')
    if record['input_token_ids'] != expected_input_ids or not 0 < len(expected_input_ids) <= 6144 or any(type(value) is not int or value < 0 for value in record['input_token_ids']):
        raise ValueError('Full frozen tokenization differs or was truncated')
    if record['device_map'] != plan['device_map']:
        raise ValueError('Actual device map differs')
    expected_names = set(bounds['runtime_parameter_names'])
    before, after = record['parameters_before'], record['parameters_after']
    if set(before) != expected_names or set(after) != expected_names:
        raise ValueError('Missing or extra runtime parameters')
    for name in expected_names:
        value = before[name]
        if value != after[name] or value['requires_grad'] is not False or value['device'] != 'cuda:' + str(parameter_device(name, plan)):
            raise ValueError('Frozen parameter bytes/ownership changed')
        if value['logical_shape'] != metadata['tensors'][name]['shape'] or not re.fullmatch('[0-9a-f]{64}', value['sha256']):
            raise ValueError('Parameter logical shape or fingerprint differs')
    if record['quantization_and_buffer_sha256_before'] != record['quantization_and_buffer_sha256_after']:
        raise ValueError('Quantization metadata or model buffers changed')
    if not re.fullmatch('[0-9a-f]{64}', record['quantization_and_buffer_sha256_before']):
        raise ValueError('Missing quantization/buffer fingerprint')
    activations = record['last_token_activations']
    if len(activations) != 64 or any(type(value['layer']) is not int for value in activations) or {value['layer'] for value in activations} != set(range(64)):
        raise ValueError('All 64 block activations are required')
    for value in activations:
        expected_device = 'cuda:' + str(parameter_device('model.language_model.layers.' + str(value['layer']), plan))
        if value['device'] != expected_device or value['source_dtype'] != 'bfloat16':
            raise ValueError('Activation device/dtype differs')
        if len(value['values']) != 5120 or not all(type(number) in [int, float] and math.isfinite(number) for number in value['values']):
            raise ValueError('Missing/nonfinite raw activation values')
    if not record['cache_boundary']['prefill_checked'] or record['cache_boundary']['initial_cache_length'] != 0 or record['cache_boundary']['reused_previous_response_cache']:
        raise ValueError('Response cache boundary differs')
    return {'audit_status': 'frozen_placement_and_raw_activation_checks_passed',
            'layers_replayed': 64, 'runtime_parameters_checked': len(expected_names),
            'audit_model_forwards': 0, 'optimizer_updates': 0,
            'cross_device_numerical_parity': 'not_established_by_this_audit',
            'full_validation_admitted': False, 'heldout_confirmation': 'not_performed'}

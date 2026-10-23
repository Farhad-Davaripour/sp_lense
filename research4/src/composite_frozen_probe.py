"""Gated frozen-Qwen loader and raw collector; no cloud submission API.

The normal Kaggle worker still admits recognition only. A separate reviewed
worker, shared-ledger reservation and explicit two-device human exception are
required before using these functions on cloud compute.
"""
import hashlib
import base64
import math
import os
import re
from pathlib import Path
from .io_utils import read, sha, verify_release, digest
from .composite_placement import candidate_plan, metadata_bounds, parameter_device, audit_frozen_activations, CONFIG_SHA256, INDEX_SHA256

MODE = 'frozen-qwen-two-t4-qualification-v1'
PIN = 'EXECUTION_PLATFORM.json'
AUTHORIZATION = 'FROZEN_QWEN_AUTHORIZATION.json'
METADATA = 'QWEN_TENSOR_METADATA.json'
PROFILE_FILE = 'config/composite_qwen_two_t4_candidate_v1.json'
CONDITIONS = ['quiet_pending', 'own_pending']
ORIGINAL_INPUTS = 'db9522b76dc7af09fa354577761441d3896b62d17706bab96ea37c9159b48596'
TRANSPORT = '1a6c40bfb8cdff88fc2e3551812591e10861dadbc17a9a83c0823d20de528887'
BOUNDS = {'hard_timeout_seconds': 2700, 'worker_seconds': 2400,
          'export_reserve_seconds': 180, 'maximum_backbone_forwards': 400,
          'maximum_concurrent_research_runs': 1, 'automatic_scientific_retries': 0}
MODEL_PIN_SHA256 = '6fee1947bc24bd569c45bd6838da9da116f16a79ede3de70298cd005f161e076'


def verify_original_model_pin(model_pin):
    if digest(model_pin) != MODEL_PIN_SHA256:
        raise ValueError('The complete original reviewed model-file pin changed')
    return True


def verify_downloaded_model(context, path):
    from .composite_store import verify_model_files
    verify_original_model_pin(context['model_pin'])
    proof = verify_model_files(context['model_pin'], path)
    if sha(Path(path) / 'config.json') != CONFIG_SHA256:
        raise ValueError('Actual downloaded model config differs from its fixed original digest')
    validate_download_proof(context, proof)
    return proof


def validate_download_proof(context, files):
    verify_original_model_pin(context['model_pin'])
    expected = {value['name']: value for value in context['model_pin']['files']}
    if len(files) != len(expected) or {value['name'] for value in files} != set(expected):
        raise ValueError('Missing original model-file verification')
    fixed = {'config.json': CONFIG_SHA256, 'model.safetensors.index.json': INDEX_SHA256}
    for value in files:
        pin = expected[value['name']]
        target = fixed.get(value['name']) or pin.get('sha256')
        if value['size'] != pin['size'] or not re.fullmatch('[0-9a-f]{64}', value['sha256']) or (target and value['sha256'] != target):
            raise ValueError('Downloaded model-file proof differs from the original reviewed pin')


def verify_metadata_bytes(model_pin, files, encoded):
    expected = {value['name']:value for value in model_pin['files'] if not value['name'].endswith('.safetensors')}
    if set(encoded) != set(expected):
        raise ValueError('Exactly the original small metadata files are required; no weight export')
    proof = {value['name']:value for value in files}
    if sum(value['size'] for value in expected.values()) > 64 * 1024**2:
        raise ValueError('Metadata export exceeds its bounded size')
    for name, pin in expected.items():
        raw = base64.b64decode(encoded[name], validate=True)
        value = hashlib.sha256(raw).hexdigest()
        if len(raw) != pin['size'] or proof[name]['sha256'] != value or (pin.get('sha256') and pin['sha256'] != value):
            raise ValueError('Actual model metadata bytes differ from their file proof')
        if pin.get('git_blob_id') and hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() != pin['git_blob_id']:
            raise ValueError('Actual model metadata bytes differ from the original Git blob')
    return {'metadata_files_verified':len(expected),'pretrained_weight_bytes_exported':0}


def authorized_load_plan(root, commit, config):
    """Validate metadata only. A document cannot create human authority."""
    root = Path(root)
    if not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('A complete reviewed source commit is required')
    if not (root / 'inputs' / AUTHORIZATION).is_file():
        raise PermissionError('No separate human exception exists for the frozen two-device probe')
    authorization = read(root / 'inputs' / AUTHORIZATION)
    if authorization.get('status') != 'explicit_human_exception' or authorization.get('authorized_by') != 'human_user':
        raise PermissionError('Frozen two-device loading requires its separate human GPU-count exception')
    verify_release(root, commit)
    for name, expected in read(root / 'inputs/MANIFEST.json').items():
        if Path(name).name != name or sha(root / 'inputs' / name) != expected:
            raise ValueError('Frozen probe input changed')
    pin = read(root / 'inputs' / PIN)
    if pin['platform'] != 'kaggle' or pin['account'] != 'farhad89' or pin['mode'] != MODE or pin['source_commit'] != commit:
        raise ValueError('Frozen probe identity/source differs')
    if pin['scientific_visible_devices'] != '0,1' or any(type(pin.get(key)) is not type(value) or pin[key] != value for key, value in BOUNDS.items()):
        raise ValueError('The frozen probe time/device/forward bounds differ')
    if pin['original_scientific_input_manifest_sha256'] != ORIGINAL_INPUTS or pin['transport_manifest_sha256'] != TRANSPORT:
        raise ValueError('The frozen probe must retain the original scientific inputs')
    expected = {'mode': MODE, 'scope': 'frozen_base_activation_analysis_only',
                'source_commit': commit, 'notebook': pin['notebook'],
                'repository': config['repository'], 'revision': config['revision'],
                'profile_sha256': sha(root / PROFILE_FILE),
                'original_scientific_input_manifest_sha256': pin['original_scientific_input_manifest_sha256'],
                'transport_manifest_sha256': pin['transport_manifest_sha256'],
                'scientific_visible_GPU_count': 2, 'maximum_new_stage_submissions': 1,
                'hard_timeout_seconds': 2700, 'worker_seconds': 2400,
                'export_reserve_seconds': 180, 'maximum_backbone_forwards': 400,
                'maximum_concurrent_research_runs': 1, 'automatic_scientific_retries': 0,
                'optimizer_updates': 0, 'behavior_adapter_fitting': False,
                'heldout_content_read': False, 'training_probe_conditions': CONDITIONS,
                'complete_prompts_no_truncation': True}
    if any(type(authorization.get(key)) is not type(value) or authorization[key] != value for key, value in expected.items()):
        raise PermissionError('Human exception does not bind this exact frozen probe')
    if not authorization.get('human_message_evidence_locator'):
        raise PermissionError('Actual human authorization evidence is required')
    profile = read(root / PROFILE_FILE)
    plan = candidate_plan(config, profile)
    metadata = read(root / 'inputs' / METADATA)
    bounds = metadata_bounds(metadata, plan)
    model_pin = read(root / 'inputs/QWEN_PIN.json')
    verify_original_model_pin(model_pin)
    if read(root / 'config/composite_qwen_file_pin_fc05.json') != model_pin:
        raise ValueError('Runtime and reviewed source model-file manifests differ')
    if model_pin['repository'] != plan['repository'] or model_pin['revision'] != plan['revision']:
        raise ValueError('Pinned model files belong to another model')
    return {'source_root': str(root), 'source_commit': commit, 'pin': pin, 'plan': plan, 'metadata': metadata,
            'metadata_bounds': bounds, 'model_pin': model_pin,
            'authorization_sha256': sha(root / 'inputs' / AUTHORIZATION),
            'load_attempt_authorized': True, 'runtime_memory_admitted': False,
            'full_validation_admitted': False}


def verify_software(torch_version, packages):
    expected = {'transformers': '5.19.0', 'bitsandbytes': '0.49.2', 'accelerate': '1.15.0',
                'safetensors': '0.8.0', 'numpy': '2.2.2', 'psutil': '7.2.2'}
    if torch_version != '2.11.0+cu128' or any(packages.get(name) != value for name, value in expected.items()):
        raise RuntimeError('Pinned frozen-probe software differs from qualification')
    return {'torch': torch_version, 'packages': expected}


def verify_hardware(context, cuda):
    """Read live device metadata only; this is not a runtime fit admission."""
    from .composite_runtime import require_scientific_runtime, platform
    require_scientific_runtime()
    if platform() != 'kaggle' or os.environ.get('CUDA_VISIBLE_DEVICES') != '0,1':
        raise PermissionError('The frozen candidate requires the explicit Kaggle device pair')
    if not cuda.is_available() or cuda.device_count() != 2:
        raise RuntimeError('Exactly two visible GPUs are required for this candidate')
    records = []
    for index, lower in enumerate(context['metadata_bounds']['device_weight_lower_bounds_bytes']):
        name = cuda.get_device_name(index)
        if name != 'Tesla T4':
            raise ValueError('The candidate was frozen for the observed T4 pair')
        free, total = cuda.mem_get_info(index)
        if free < lower + 2 * 1024**3:
            raise MemoryError('Insufficient per-device lower-bound plus qualification reserve')
        records.append({'index': index, 'name': name, 'free_bytes': free, 'total_bytes': total,
                        'optimistic_weight_lower_bound_bytes': lower})
    return {'devices': records, 'runtime_memory_admitted': False}


def verify_runtime(context):
    import torch
    from importlib.metadata import version
    software = verify_software(torch.__version__, {name: version(name) for name in
        ['transformers', 'bitsandbytes', 'accelerate', 'safetensors', 'numpy', 'psutil']})
    return {**verify_hardware(context, torch.cuda), 'software': software}


def guard_frozen_devices(cuda):
    for index in [0, 1]:
        if cuda.mem_get_info(index)[0] < 512 * 1024**2:
            raise MemoryError('Frozen probe lost its per-device export reserve')


def memory_record(cuda):
    return [{'device': index, 'free_bytes': cuda.mem_get_info(index)[0],
             'total_bytes': cuda.mem_get_info(index)[1],
             'allocated_bytes': cuda.memory_allocated(index), 'reserved_bytes': cuda.memory_reserved(index),
             'peak_allocated_bytes': cuda.max_memory_allocated(index),
             'peak_reserved_bytes': cuda.max_memory_reserved(index)} for index in [0, 1]]


def logical_shape(parameter):
    state = getattr(parameter, 'quant_state', None)
    return list(state.shape if state is not None else parameter.shape)


def validate_loading_info(info):
    if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
        raise ValueError('Frozen model loading cannot initialize missing or mismatched weights')
    unexpected = sorted(info.get('unexpected_keys', []))
    if any(not name.startswith('mtp.') for name in unexpected):
        raise ValueError('Unexpected non-MTP checkpoint tensors in the frozen loader')
    return {'missing_keys': [], 'mismatched_keys': [], 'error_msgs': [],
            'ignored_checkpoint_only_MTP_keys': unexpected}


def protected_cache_path(cache, commit):
    if not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('A complete source commit must bind the cache namespace')
    base = (Path('/kaggle/temp/research4_frozen_probe_cache') / commit).resolve()
    target = Path(cache).resolve()
    if target == base or not target.is_relative_to(base):
        raise ValueError('Full model files must remain in owned ephemeral Research4 cache')
    return target


def projection_quantization_proof(name, module, parameter):
    import torch
    state = getattr(parameter, 'quant_state', None)
    if getattr(module, 'weight', None) is not parameter or state is None:
        raise ValueError('Expected a quantized original projection: ' + name)
    if getattr(state, 'quant_type', None) != 'nf4' or getattr(state, 'nested', None) is not True or getattr(state, 'state2', None) is None:
        raise ValueError('Frozen projection must use NF4 with double quantization')
    if parameter.dtype != torch.uint8 or getattr(module, 'compute_dtype', None) != torch.bfloat16 or getattr(state, 'dtype', None) != torch.bfloat16:
        raise ValueError('Frozen projection storage/compute dtype differs')
    if state.blocksize != 64 or state.state2.blocksize != 256 or getattr(state, 'packing_format_for_cpu', False):
        raise ValueError('Frozen CUDA quantization block/packing profile differs')
    expected_device = parameter.device.index
    locations = {}
    for key, tensor in [('absmax', state.absmax), ('code', state.code),
                        ('nested_absmax', state.state2.absmax), ('nested_code', state.state2.code)]:
        if tensor.device.type != 'cuda' or tensor.device.index != expected_device:
            raise ValueError('Quantization state was moved outside its projection device')
        locations[key] = 'cuda:' + str(tensor.device.index)
    return {'quant_type': 'nf4', 'double_quantization': True,
            'storage_dtype': 'torch.uint8', 'compute_dtype': 'torch.bfloat16',
            'state_dtype': 'torch.bfloat16', 'blocksize': 64, 'nested_blocksize': 256,
            'logical_shape': logical_shape(parameter), 'state_devices': locations}


def text_projection(name):
    return re.fullmatch(r'model\.language_model\.layers\.\d+\.(?:mlp\.(?:gate_proj|up_proj|down_proj)|self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj))\.weight', name) is not None


def verify_loaded_layout(model, context):
    import torch
    plan = context['plan']; expected = context['metadata']['tensors']
    parameters = dict(model.named_parameters())
    if set(parameters) != set(context['metadata_bounds']['runtime_parameter_names']):
        raise ValueError('Missing or extra original runtime parameters after loading')
    unquantized = {}; quantized = {}
    for name, parameter in parameters.items():
        if parameter.requires_grad or parameter.device.type != 'cuda' or parameter.device.index != parameter_device(name, plan):
            raise ValueError('Frozen parameter ownership or gradient contract differs')
        if logical_shape(parameter) != expected[name]['shape']:
            raise ValueError('The loaded model changed a pinned logical tensor shape')
        if name.endswith('embed_tokens.weight') or name == 'lm_head.weight':
            if parameter.dtype != torch.bfloat16 or getattr(parameter, 'quant_state', None) is not None:
                raise ValueError('Input embeddings and output head must remain unquantized BF16')
            unquantized[name] = {'dtype': 'torch.bfloat16', 'shape': list(parameter.shape),
                                 'device': 'cuda:' + str(parameter.device.index)}
        elif name.endswith('.weight'):
            module = model.get_submodule(name.removesuffix('.weight'))
            if text_projection(name) or (hasattr(module, 'in_features') and hasattr(module, 'out_features')):
                quantized[name] = projection_quantization_proof(name, module, parameter)
    if len(unquantized) != 2 or not {name for name in parameters if text_projection(name)} <= set(quantized):
        raise ValueError('Incomplete embedding/head or expected projection quantization proof')
    actual_map = {}
    for name, device in getattr(model, 'hf_device_map', {}).items():
        if type(device) is int:
            index = device
        elif isinstance(device, str) and device in ['cuda:0', 'cuda:1']:
            index = int(device[-1])
        elif getattr(device, 'type', None) == 'cuda':
            index = device.index
        else:
            raise ValueError('The actual loader introduced automatic/CPU/disk placement')
        actual_map[name] = index
    if actual_map != plan['device_map']:
        raise ValueError('Actual loader device map differs from the frozen candidate')
    return {'original_runtime_parameters_verified': len(parameters), 'devices': [0, 1], 'device_map': actual_map,
            'unquantized_BF16': unquantized, 'NF4_double_quant_projections': quantized,
            'model_instances': 1, 'optimizer_updates': 0, 'runtime_memory_admitted': False}


def audit_loaded_layout(proof, context, parameters):
    expected_names = set(context['metadata_bounds']['runtime_parameter_names'])
    if proof['original_runtime_parameters_verified'] != len(expected_names) or set(parameters) != expected_names or proof['device_map'] != context['plan']['device_map']:
        raise ValueError('Loaded-layout coverage or device map differs')
    if proof['model_instances'] != 1 or proof['optimizer_updates'] != 0:
        raise ValueError('Loaded-layout model/fitting count differs')
    unquantized = {name for name in expected_names if name.endswith('embed_tokens.weight') or name == 'lm_head.weight'}
    if len(unquantized) != 2 or set(proof['unquantized_BF16']) != unquantized:
        raise ValueError('Missing BF16 embedding/head proof')
    for name in unquantized:
        expected = {'dtype':'torch.bfloat16','shape':context['metadata']['tensors'][name]['shape'],
                    'device':'cuda:'+str(parameter_device(name,context['plan']))}
        if proof['unquantized_BF16'][name] != expected or parameters[name]['dtype'] != 'torch.bfloat16':
            raise ValueError('Unquantized input/output weights are not original BF16')
    quantized = proof['NF4_double_quant_projections']
    packed = {name for name, value in parameters.items() if value['dtype'] == 'torch.uint8'}
    if set(quantized) != packed or not {name for name in expected_names if text_projection(name)} <= packed:
        raise ValueError('Missing expected quantized projection proof')
    for name, value in quantized.items():
        device = 'cuda:'+str(parameter_device(name, context['plan']))
        expected = {'quant_type':'nf4','double_quantization':True,'storage_dtype':'torch.uint8',
                    'compute_dtype':'torch.bfloat16','state_dtype':'torch.bfloat16',
                    'blocksize':64,'nested_blocksize':256,
                    'logical_shape':context['metadata']['tensors'][name]['shape'],
                    'state_devices':{key:device for key in ['absmax','code','nested_absmax','nested_code']}}
        if value != expected:
            raise ValueError('NF4/double-quant projection state differs from the frozen profile')
    return {'unquantized_BF16_parameters':len(unquantized), 'NF4_double_quant_projections':len(quantized)}


def load_training_rows(context, store=None):
    from .composite_store import Store
    pin = context['pin']
    if store is None:
        if os.environ.get('RESEARCH4_KAGGLE_INPUT_ROOT') != pin['input_root'] or os.environ.get('RESEARCH4_KAGGLE_MANIFEST_SHA256') != pin['transport_manifest_sha256']:
            raise ValueError('Probe transport environment differs from its input freeze')
        store = Store()
    raw_release = store.get('data_v1/DATA_RELEASE.json')
    import json
    release = json.loads(raw_release)
    if release['world_source_sha256'] != sha(Path(context['source_root']) / 'src/composite_world.py'):
        raise ValueError('Probe observations came from another world version')
    raw = store.get('data_v1/training/DETECTION_ROWS.json')
    if hashlib.sha256(raw).hexdigest() != release['cohorts']['training']['files']['DETECTION_ROWS.json']:
        raise ValueError('Probe training cohort digest differs')
    rows = json.loads(raw); allowed = release['cohorts']['training']
    if len(rows) != 720 or any(row['split'] != 'training' or row['case_id'] not in allowed['case_ids'] or row['group_id'] not in allowed['group_ids'] for row in rows):
        raise ValueError('Only the original complete training cohort may provide probe prompts')
    return rows, {'training_rows_sha256': hashlib.sha256(raw).hexdigest(),
                  'data_release_sha256': hashlib.sha256(raw_release).hexdigest(),
                  'training_rows_canonical_sha256': digest(rows),
                  'source_training_observations': len(rows), 'heldout_content_read': False}


def select_training_probes(rows):
    if not rows or any(row['split'] != 'training' for row in rows):
        raise PermissionError('Only the original frozen training cohort may select qualification prompts')
    selected = []
    for condition in CONDITIONS:
        candidates = [row for row in rows if row['condition'] == condition and row['sequence'] == 0 and not row['terminal']]
        if not candidates:
            raise ValueError('The complete predeclared training prompt is missing')
        selected.append(candidates[0])  # Retain immutable dataset order, not metric-based selection.
    if len({row['state_id'] for row in selected}) != 2:
        raise ValueError('Qualification prompt identities must be distinct')
    return selected


class LastTokenCapture:
    """A read-only forward hook; stores raw last-token vectors without replacement."""
    def __init__(self, layers, hidden_size, tokens):
        self.layers = list(layers); self.hidden_size = hidden_size; self.tokens = tokens
        self.records = {}; self.handles = []

    def __enter__(self):
        for index, layer in enumerate(self.layers):
            def capture(module, args, result, index=index):
                import torch
                tensor = result[0] if isinstance(result, (tuple, list)) else result
                if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != (1, self.tokens, self.hidden_size):
                    raise ValueError('The complete prefill activation shape differs')
                if index in self.records:
                    raise ValueError('A decoder block executed twice in one captured prefill')
                values = tensor[0, -1].detach().float().cpu().tolist()
                if not all(math.isfinite(value) for value in values):
                    raise FloatingPointError('Nonfinite captured hidden activation')
                self.records[index] = {'layer': index, 'device': str(tensor.device),
                                       'source_dtype': str(tensor.dtype).removeprefix('torch.'),
                                       'values': values}
                # No return value: the model's original result is unchanged.
            self.handles.append(layer.register_forward_hook(capture))
        return self

    def __exit__(self, *error):
        for handle in self.handles: handle.remove()
        self.handles = []
        return False

    def complete(self):
        if set(self.records) != set(range(len(self.layers))):
            raise ValueError('Not every decoder block produced a raw activation')
        return [self.records[index] for index in range(len(self.layers))]


def parameter_records(actor, snapshot=None):
    from .composite_training import parameter_snapshot
    actual = dict(actor.model.named_parameters())
    if set(actual) != {name for name, _ in actor.base_references} or any(actual[name] is not value for name, value in actor.base_references):
        raise ValueError('The original model parameter objects were replaced')
    snapshot = parameter_snapshot(actor.base_references) if snapshot is None else snapshot
    return {name: {'logical_shape': logical_shape(parameter), 'sha256': fingerprint,
                   'device': str(parameter.device), 'requires_grad': bool(parameter.requires_grad),
                   'dtype': str(parameter.dtype), 'version': version}
            for name, parameter, version, fingerprint in snapshot}


def collect_prefill(actor, text):
    import torch
    from .composite_interface import input_ids
    if not getattr(actor, 'frozen_analysis_only', False) or actor.adapters or actor.generating:
        raise PermissionError('Raw prefill collection is frozen-base only')
    ids = input_ids(actor.tokenizer, text)
    if not 0 < len(ids) <= actor.cfg['maximum_tokens']:
        raise ValueError('Complete prompt exceeds its unchanged context ceiling')
    tokens = torch.tensor([ids], device='cuda:0', dtype=torch.long)
    actor.clear_cache()
    with LastTokenCapture(actor.text.layers, actor.cfg['hidden_size'], len(ids)) as captured:
        with torch.inference_mode():
            result = actor.model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                                 past_key_values=None, use_cache=False, logits_to_keep=1, return_dict=True)
    logits = result.logits[0, -1].float().cpu().tolist()
    if not all(math.isfinite(value) for value in logits):
        raise FloatingPointError('Nonfinite raw prefill logits')
    return {'input_token_ids': ids, 'last_token_activations': captured.complete(),
            'raw_last_logits': logits, 'prompt_sha256': hashlib.sha256(text.encode()).hexdigest()}


def collect_frozen_probes(actor, context, persist):
    """Collect two fixed training probes. Caller must own bounds and durable output."""
    from .composite_training import auxiliary_digest
    if not getattr(actor, 'frozen_analysis_only', False) or actor.adapters:
        raise PermissionError('Qualification must not install or fit behavior adapters')
    if actor.frozen_probe_context['source_commit'] != context['source_commit'] or actor.frozen_probe_context['authorization_sha256'] != context['authorization_sha256']:
        raise ValueError('Actor and collector use different source/authorization freezes')
    rows, provenance = load_training_rows(context)
    selected = select_training_probes(rows)
    persist('FROZEN_PROBE_SELECTION.json', {'conditions': CONDITIONS, 'state_ids': [row['state_id'] for row in selected],
            'prompt_sha256': [hashlib.sha256(row['text'].encode()).hexdigest() for row in selected],
            'provenance': provenance, 'selected_before_forwards': True, 'heldout_content_read': False})
    before = parameter_records(actor, actor.frozen_snapshot)
    buffers_before = auxiliary_digest(actor.base_references, actor.base_buffers)
    persist('FROZEN_PARAMETERS_BEFORE.json', before)
    record = {'profile': context['plan']['profile'], 'revision': context['plan']['revision'],
              'source_commit': context['source_commit'], 'device_map': actor.frozen_layout_proof['device_map'],
              'model_instances': 1, 'optimizer_updates': 0, 'trainable_parameters': 0,
              'heldout_content_read': False, 'behavior_adapter_fitting': False, 'kernel_profile': 'reference',
              'parameters_before': before, 'quantization_and_buffer_sha256_before': buffers_before,
              'probe_conditions': CONDITIONS, 'probes': [], 'simulator_actions_executed': 0,
              'data_provenance': provenance, 'authorization_sha256': context['authorization_sha256'],
              'loading_info': actor.frozen_loading_proof,
              'loaded_layout': actor.frozen_layout_proof,
              'model_metadata_bytes': actor.frozen_probe_metadata_bytes,
              'model_file_verification': actor.frozen_probe_file_proof}
    import torch
    record['math_profile'] = {'cuda_matmul_allow_tf32': bool(torch.backends.cuda.matmul.allow_tf32),
                              'caller_autocast_enabled': bool(torch.is_autocast_enabled())}
    for index, row in enumerate(selected):
        value = collect_prefill(actor, row['text'])
        persist('FROZEN_PREFILL_' + str(index) + '.json', value)
        generation = actor.generate(row['text'], False)
        if generation['input_token_ids'] != value['input_token_ids']:
            raise ValueError('Prefill and native generation received different complete prompts')
        cache = generation['cache']
        probe = {**value, 'state_id': row['state_id'], 'case_id': row['case_id'], 'condition': row['condition'],
                 'generation': generation,
                 'cache_boundary': {'prefill_checked': cache['prefill_checked'],
                                    'initial_cache_length': cache['initial_cache_length'],
                                    'reused_previous_response_cache': cache['previous_cache_reused']}}
        record['probes'].append(probe)
        persist('FROZEN_PROBE_' + str(index) + '.json', probe)
    actor.verify_frozen()
    record['parameters_after'] = parameter_records(actor)
    record['quantization_and_buffer_sha256_after'] = auxiliary_digest(actor.base_references, actor.base_buffers)
    record['device_memory'] = memory_record(torch.cuda)
    record['forward_counts'] = dict(actor.frozen_probe_counts)
    persist('FROZEN_PARAMETERS_AFTER.json', record['parameters_after'])
    return record


def audit_probe_record(record, rows, context, tokenizer, eos):
    from .composite_interface import input_ids
    from .native27_tools import decode, parse
    selected = select_training_probes(rows)
    if any(type(record[key]) is not int for key in ['model_instances','optimizer_updates','trainable_parameters','simulator_actions_executed']) or record['heldout_content_read'] is not False or record['behavior_adapter_fitting'] is not False:
        raise ValueError('Frozen scope/counts must be known and explicit')
    if record['probe_conditions'] != CONDITIONS or len(record['probes']) != 2 or record['simulator_actions_executed'] != 0:
        raise ValueError('The declared qualification probes/actions differ')
    if record['source_commit'] != context['source_commit']:
        raise ValueError('Raw frozen probe evidence belongs to another source')
    if record['authorization_sha256'] != context['authorization_sha256']:
        raise ValueError('Frozen probe evidence belongs to another authority freeze')
    if len(rows) != 720 or record['data_provenance']['source_training_observations'] != len(rows) or record['data_provenance']['training_rows_canonical_sha256'] != digest(rows) or record['data_provenance']['heldout_content_read'] is not False:
        raise ValueError('The complete expected training cohort differs from raw provenance')
    if record['math_profile'] != {'cuda_matmul_allow_tf32':False,'caller_autocast_enabled':False}:
        raise ValueError('Frozen inference math controls differ')
    loading = record['loading_info']
    if loading['missing_keys'] or loading['mismatched_keys'] or loading['error_msgs'] or any(not key.startswith('mtp.') for key in loading['ignored_checkpoint_only_MTP_keys']):
        raise ValueError('The full original pretrained model was not loaded cleanly')
    validate_download_proof(context, record['model_file_verification'])
    metadata_audit = verify_metadata_bytes(context['model_pin'],record['model_file_verification'],record['model_metadata_bytes'])
    layout_audit = audit_loaded_layout(record['loaded_layout'], context, record['parameters_before'])
    counts = record['forward_counts']
    if any(type(counts[key]) is not int for key in ['attempted','completed']) or not 0 < counts['completed'] <= counts['attempted'] <= 400:
        raise ValueError('Frozen probe forward accounting differs')
    if len(record['device_memory']) != 2 or {value['device'] for value in record['device_memory']} != {0, 1}:
        raise ValueError('Both devices require actual memory observations')
    for value in record['device_memory']:
        if value['free_bytes'] < 512 * 1024**2 or not 0 <= value['allocated_bytes'] <= value['reserved_bytes'] <= value['total_bytes']:
            raise ValueError('Actual per-device memory/reserve evidence differs')
        if not value['allocated_bytes'] <= value['peak_allocated_bytes'] <= value['peak_reserved_bytes'] <= value['total_bytes'] or value['peak_reserved_bytes'] < value['reserved_bytes']:
            raise ValueError('Per-device peak-memory evidence differs')
    audits = []
    root_fields = ['profile', 'revision', 'device_map', 'model_instances', 'optimizer_updates',
                   'trainable_parameters', 'heldout_content_read', 'kernel_profile',
                   'parameters_before', 'parameters_after', 'quantization_and_buffer_sha256_before',
                   'quantization_and_buffer_sha256_after']
    for probe, row in zip(record['probes'], selected):
        if any(probe[key] != row[key] for key in ['state_id', 'case_id', 'condition']):
            raise ValueError('Qualification selected another training observation')
        if probe['prompt_sha256'] != hashlib.sha256(row['text'].encode()).hexdigest():
            raise ValueError('Frozen prompt bytes differ')
        expanded = {key: record[key] for key in root_fields}
        expanded.update({key: probe[key] for key in ['input_token_ids', 'last_token_activations', 'cache_boundary']})
        audits.append(audit_frozen_activations(expanded, context['plan'], context['metadata'], input_ids(tokenizer, row['text'])))
        vocabulary = context['metadata']['tensors']['lm_head.weight']['shape'][0]
        if not eos or any(type(token) is not int or not 0 <= token < vocabulary for token in eos):
            raise ValueError('A complete known native EOS set is required')
        if len(probe['raw_last_logits']) != vocabulary or not all(type(value) in [int, float] and math.isfinite(value) for value in probe['raw_last_logits']):
            raise ValueError('Raw finite model logits are required')
        generation = probe['generation']; tokens = generation['generated_token_ids']; cache = generation['cache']
        if generation['generation_parameters'] != {'max_new_tokens':160,'do_sample':False,'num_beams':1,'num_return_sequences':1,'use_cache':True}:
            raise ValueError('Native generation settings differ from the frozen profile')
        if generation['input_token_ids'] != probe['input_token_ids'] or generation['adapter_enabled'] or generation['kernel_profile'] != 'reference':
            raise ValueError('Native generation changed its prompt, adapter or kernels')
        if not 0 < len(tokens) <= 160 or any(type(token) is not int or not 0 <= token < vocabulary for token in tokens):
            raise ValueError('Native generation violated its token ceiling')
        if any(token in eos for token in tokens[:-1]) or (tokens[-1] not in eos and len(tokens) != 160):
            raise ValueError('Generation continued after EOS or stopped before its full non-EOS ceiling')
        raw = decode(tokenizer, tokens, eos); ended = tokens[-1] in eos
        try: actions = parse(raw, 1) if ended else None
        except (ValueError, TypeError, RecursionError): actions = None
        if raw != generation['raw_text'] or ended != generation['finished_with_eos'] or actions != generation['actions']:
            raise ValueError('Native token decoding/parser replay differs')
        if cache['cache_object_present'] is not True or cache['previous_cache_reused'] is not False:
            raise ValueError('Missing fresh-cache provenance')
        boundary = probe['cache_boundary']
        if cache['prefill_checked'] is not True or type(cache['initial_cache_length']) is not int or cache['initial_cache_length'] != 0:
            raise ValueError('Native generation did not prove an initially empty cache')
        if boundary != {'prefill_checked':cache['prefill_checked'], 'initial_cache_length':cache['initial_cache_length'],
                        'reused_previous_response_cache':cache['previous_cache_reused']}:
            raise ValueError('Flattened and raw generation cache evidence differ')
        if type(cache['cache_identity']) is not int or cache['cache_identity'] <= 0 or type(cache['previous_cache_alive']) is not bool or cache['previous_cache_alive'] != (cache['previous_cache_identity'] is not None):
            raise ValueError('Incomplete raw cache-identity provenance')
        if cache['previous_cache_alive'] and (type(cache['previous_cache_identity']) is not int or cache['previous_cache_identity'] <= 0):
            raise ValueError('Previous live cache identity is unknown')
        if cache['previous_cache_alive'] and cache['cache_identity'] == cache['previous_cache_identity']:
            raise ValueError('A live cache object crossed response boundaries')
    expected_forwards = 2 + sum(len(probe['generation']['generated_token_ids']) for probe in record['probes'])
    if counts['attempted'] != expected_forwards or counts['completed'] != expected_forwards:
        raise ValueError('Unaccounted or incomplete model forwards in a completed probe record')
    return {'status': 'raw_frozen_qualification_evidence_audited', 'probes': len(audits),
            'activation_layers_per_probe': 64, 'audit_model_forwards': 0, 'optimizer_updates': 0,
            'native_parse_successes': sum(probe['generation']['actions'] is not None for probe in record['probes']),
            'loaded_layout': layout_audit,
            'model_metadata': metadata_audit,
            'cross_device_numerical_parity': 'not_established', 'full_validation_admitted': False,
            'heldout_confirmation': 'not_performed'}


def load_probe_actor(root, commit, config, cache, guard, on_forward, persist):
    """Called only by the separately gated qualification child."""
    if not all(callable(function) for function in [guard, on_forward, persist]):
        raise TypeError('A bounded owner, forward accounting and durable writer are required')
    context = authorized_load_plan(root, commit, config)
    cache = protected_cache_path(cache, commit)
    import torch
    hardware = verify_runtime(context)
    for index in [0, 1]: torch.cuda.reset_peak_memory_stats(index)
    persist('FROZEN_LOAD_FREEZE.json', {'source_commit': commit, 'profile': context['plan'],
            'authorization_sha256': context['authorization_sha256'], 'bounds': BOUNDS,
            'training_probe_conditions': CONDITIONS, 'hardware_preflight': hardware,
            'heldout_content_read': False, 'runtime_memory_admitted': False})
    from .composite_store import download_model
    from .composite_actor import Actor
    path, files = download_model(context['model_pin'], cache)
    metadata_bytes = {value['name']:base64.b64encode((path / value['name']).read_bytes()).decode('ascii')
                      for value in context['model_pin']['files'] if not value['name'].endswith('.safetensors')}
    validate_download_proof(context,files)
    verify_metadata_bytes(context['model_pin'],files,metadata_bytes)
    persist('MODEL_FILE_VERIFICATION.json', files)
    persist('MODEL_METADATA_BYTES.json', metadata_bytes)
    actor = Actor(path, config, guard, on_forward, kernel_profile='reference',
                  frozen_probe={'root': str(root), 'source_commit': commit})
    if actor.frozen_probe_file_proof != files:
        raise ValueError('Model files changed between download verification and native loading')
    actor.frozen_probe_metadata_bytes=metadata_bytes
    proof = {'hardware_preflight': hardware, 'model_files': files, 'device_memory': memory_record(torch.cuda),
             'loaded_layout': actor.frozen_layout_proof, 'loading_info': actor.frozen_loading_proof,
             'runtime_memory_admitted': False}
    persist('FROZEN_LOADED_LAYOUT.json', proof)
    return actor, context, proof

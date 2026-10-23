"""Read-only capture of the actual native-generation prefill, with no extra call."""
import hashlib
import io
import threading

import numpy as np
import torch
from torch import nn


class NativePrefillCapture:
    def __init__(self, model, layers, hidden_size, tokens, capture_layers=True):
        self.model = model; self.layers = list(layers)
        self.hidden_size = hidden_size; self.tokens = tokens
        self.capture_layers = capture_layers; self.handles = []
        self.prefilling = False; self.started = False; self.finished = False
        self.activations = {}; self.logits = None

    def __enter__(self):
        if self.handles or self.started or self.tokens <= 0:
            raise ValueError('A capture owns exactly one fresh native response')
        self.handles.append(self.model.register_forward_pre_hook(self._before, with_kwargs=True))
        self.handles.append(self.model.register_forward_hook(self._after))
        if self.capture_layers:
            for index, layer in enumerate(self.layers):
                def capture(module, args, result, index=index):
                    if not self.prefilling:
                        return
                    tensor = result[0] if isinstance(result, (tuple, list)) else result
                    if (not isinstance(tensor, torch.Tensor)
                        or tuple(tensor.shape) != (1, self.tokens, self.hidden_size)
                        or index in self.activations):
                        raise ValueError('Native prefill layer geometry or call count differs')
                    values = tensor[0, -1].detach().float().cpu().numpy().copy()
                    if not np.isfinite(values).all():
                        raise FloatingPointError('Nonfinite native prefill activation')
                    self.activations[index] = {'values': values, 'device': str(tensor.device),
                        'source_dtype': str(tensor.dtype).removeprefix('torch.')}
                    # No returned tensor: the original model output is untouched.
                self.handles.append(layer.register_forward_hook(capture))
        return self

    def _before(self, module, args, kwargs):
        if self.started:
            return
        ids = kwargs.get('input_ids')
        if not isinstance(ids, torch.Tensor) or tuple(ids.shape) != (1, self.tokens):
            raise ValueError('Capture must start on the complete native prefill')
        self.started = True; self.prefilling = True

    def _after(self, module, args, result):
        if not self.prefilling:
            return
        logits = result.logits
        if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != 1:
            raise ValueError('Native logits_to_keep=1 prefill is required')
        self.logits = logits[0, -1].detach().float().cpu().numpy().copy()
        if not np.isfinite(self.logits).all():
            raise FloatingPointError('Nonfinite native prefill logits')
        self.prefilling = False; self.finished = True

    def __exit__(self, *error):
        for handle in self.handles:
            handle.remove()
        self.handles = []; self.prefilling = False
        return False

    def complete(self):
        if (not self.started or not self.finished or self.logits is None
            or (self.capture_layers and set(self.activations) != set(range(len(self.layers))))):
            raise ValueError('Incomplete native-generation prefill evidence')
        return self

    def encoded(self):
        self.complete()
        if not self.capture_layers:
            raise ValueError('Full layer capture is required for a feature export')
        features = np.stack([self.activations[index]['values'] for index in range(len(self.layers))])
        stream = io.BytesIO()
        np.savez_compressed(stream, activations=features, logits=self.logits)
        raw = stream.getvalue()
        return raw, {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw),
            'format': 'numpy-npz-float32-lossless', 'activation_shape': list(features.shape),
            'logit_shape': list(self.logits.shape), 'native_prefill_tokens': self.tokens,
            'layers': [{'layer': index, 'device': self.activations[index]['device'],
                        'source_dtype': self.activations[index]['source_dtype']}
                       for index in range(len(self.layers))],
            'additional_capture_forwards': 0}


def verify_feature_bytes(raw, descriptor, layers=64, hidden_size=5120, vocabulary=None):
    if hashlib.sha256(raw).hexdigest() != descriptor['sha256'] or len(raw) != descriptor['bytes']:
        raise ValueError('Native feature bytes changed')
    with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
        if set(arrays.files) != {'activations', 'logits'}:
            raise ValueError('Only complete numeric activations and logits belong in a feature record')
        features = arrays['activations']; logits = arrays['logits']
        if (features.dtype != np.float32 or logits.dtype != np.float32
            or features.shape != (layers, hidden_size) or logits.ndim != 1
            or (vocabulary is not None and len(logits) != vocabulary)
            or list(features.shape) != descriptor['activation_shape']
            or list(logits.shape) != descriptor['logit_shape']
            or not np.isfinite(features).all() or not np.isfinite(logits).all()):
            raise ValueError('Complete finite lossless native feature geometry differs')
        return features.copy(), logits.copy()


class FrozenBaseActor(nn.Module):
    """Base-only proxy; prediction state never selects an intervention."""
    def __init__(self, actor, persist_bytes, persist_json, memory_observer=lambda:None):
        super().__init__()
        if not getattr(actor, 'frozen_analysis_only', False) or actor.adapters:
            raise PermissionError('Only a frozen unadapted native actor may enter the base control')
        self.actor = actor; self.persist_bytes = persist_bytes; self.persist_json = persist_json
        self.memory_observer = memory_observer; self.response_index = 0
        self.no_op_checked = False; self.lock = threading.RLock()

    def clear_cache(self):
        self.actor.clear_cache()

    def _generate(self, text, capture_layers):
        from .composite_interface import input_ids
        ids = input_ids(self.actor.tokenizer, text)
        with NativePrefillCapture(self.actor.model, self.actor.text.layers,
                self.actor.cfg['hidden_size'], len(ids), capture_layers) as captured:
            result = self.actor.generate(text, False)
        captured.complete()
        if result['input_token_ids'] != ids:
            raise ValueError('Capture and native generation used different prompts')
        return result, captured

    def generate(self, text, on, maximum_new_tokens=None):
        if on is not False or maximum_new_tokens is not None:
            raise PermissionError('Frozen-base endpoint rejects ON controls and changed native budgets')
        with self.lock:
            baseline = None
            if not self.no_op_checked:
                baseline, uninstrumented = self._generate(text, False)
                stream = io.BytesIO(); np.save(stream, uninstrumented.logits, allow_pickle=False)
                baseline_bytes = stream.getvalue()
                self.persist_bytes('BASELINE_PREFILL_LOGITS.npy', baseline_bytes)
                self.persist_json('BASELINE_NATIVE_RESPONSE.json', baseline)
            result, captured = self._generate(text, True)
            raw, descriptor = captured.encoded()
            name = 'features/response_' + str(self.response_index).zfill(5) + '.npz'
            self.persist_bytes(name, raw)
            result['frozen_prefill_features'] = {**descriptor, 'path': name,
                'prompt_sha256': hashlib.sha256(text.encode()).hexdigest()}
            result['joint_device_memory'] = self.memory_observer()
            self.persist_json('responses/response_' + str(self.response_index).zfill(5) + '.json', result)
            if baseline is not None:
                fields = ['input_token_ids', 'generated_token_ids', 'raw_text', 'actions', 'finished_with_eos']
                equal = all(baseline[key] == result[key] for key in fields)
                logits_equal = np.array_equal(uninstrumented.logits, captured.logits)
                self.persist_json('INSTRUMENTATION_PARITY.json', {
                    'kind': 'matched-prefix-layer-capture-no-op', 'baseline_generation': baseline,
                    'captured_generation': result, 'exact_native_outputs_equal': equal,
                    'exact_prefill_logits_equal': logits_equal,
                    'baseline_logit_sha256': hashlib.sha256(uninstrumented.logits.tobytes()).hexdigest(),
                    'captured_logit_sha256': hashlib.sha256(captured.logits.tobytes()).hexdigest(),
                    'baseline_logit_file_sha256': hashlib.sha256(baseline_bytes).hexdigest(),
                    'additional_generation_forwards': len(baseline['generated_token_ids']),
                    'adapter_OFF_bypass': None, 'adapter_ON_effect': None})
                if not equal or not logits_equal:
                    raise ValueError('Read-only layer capture changed native logits or output tokens')
                self.no_op_checked = True
            self.response_index += 1
            return result

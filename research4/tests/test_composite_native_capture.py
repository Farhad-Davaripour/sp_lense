import copy
import io
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from src.composite_native_capture import NativePrefillCapture, FrozenBaseActor, verify_feature_bytes


class TinyNative(nn.Module):
    def __init__(self):
        super().__init__(); self.layers = nn.ModuleList([nn.Identity(), nn.Identity()])

    def forward(self, input_ids):
        value = input_ids.float().unsqueeze(-1).repeat(1, 1, 3)
        for layer in self.layers:
            value = layer(value)
        return SimpleNamespace(logits=value[:, -1:, :])


class CaptureTests(unittest.TestCase):
    def test_prefill_capture_is_lossless_and_ignores_decode_steps(self):
        model = TinyNative(); ids = torch.tensor([[1, 2]])
        original = model(input_ids=ids).logits
        with NativePrefillCapture(model, model.layers, 3, 2) as captured:
            actual = model(input_ids=ids).logits
            model(input_ids=torch.tensor([[9]]))
        self.assertTrue(torch.equal(original, actual))
        raw, descriptor = captured.encoded()
        features, logits = verify_feature_bytes(raw, descriptor, 2, 3, 3)
        self.assertEqual(features.tolist(), [[2., 2., 2.]] * 2)
        self.assertEqual(logits.tolist(), [2., 2., 2.])
        self.assertEqual(descriptor['additional_capture_forwards'], 0)
        self.assertFalse(model._forward_hooks or model._forward_pre_hooks)
        self.assertTrue(all(not layer._forward_hooks for layer in model.layers))
        changed = copy.deepcopy(descriptor); changed['activation_shape'] = [1, 6]
        with self.assertRaises(ValueError): verify_feature_bytes(raw, changed, 2, 3, 3)
        with self.assertRaises(ValueError): verify_feature_bytes(raw + b'x', descriptor, 2, 3, 3)

    def test_failed_or_incomplete_capture_removes_every_hook(self):
        model = TinyNative()
        with self.assertRaisesRegex(ValueError, 'complete native prefill'):
            with NativePrefillCapture(model, model.layers, 3, 2):
                model(input_ids=torch.tensor([[1]]))
        self.assertFalse(model._forward_hooks or model._forward_pre_hooks)
        self.assertTrue(all(not layer._forward_hooks for layer in model.layers))
        with NativePrefillCapture(model, model.layers, 3, 2) as empty:
            pass
        with self.assertRaisesRegex(ValueError, 'Incomplete'): empty.complete()

    def proxy(self, alter=False):
        class SyntheticActor(nn.Module):
            def __init__(self):
                super().__init__(); self.model = TinyNative(); self.adapters = {}
                self.frozen_analysis_only = True; self.text = SimpleNamespace(layers=self.model.layers)
                self.cfg = {'hidden_size': 3}; self.calls = 0
                self.tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **k: [1, 2])
            def clear_cache(self): pass
            def generate(self, text, on):
                self.calls += 1
                self.model(input_ids=torch.tensor([[1, 2]]))
                return {'input_token_ids': [1, 2], 'generated_token_ids': [3 if alter and self.calls == 2 else 2],
                        'raw_text': 'synthetic', 'actions': None, 'finished_with_eos': False}
        actor = SyntheticActor(); raw = {}; values = {}
        proxy = FrozenBaseActor(actor, lambda name, data: raw.__setitem__(name, data),
                                lambda name, value: values.__setitem__(name, copy.deepcopy(value)))
        return actor, proxy, raw, values

    def test_base_proxy_parity_is_once_and_native_result_retained(self):
        actor, proxy, raw, values = self.proxy()
        result = proxy.generate('synthetic prompt', False)
        self.assertEqual(actor.calls, 2)
        self.assertEqual(result['generated_token_ids'], [2])
        self.assertTrue(values['INSTRUMENTATION_PARITY.json']['exact_native_outputs_equal'])
        self.assertTrue(values['INSTRUMENTATION_PARITY.json']['exact_prefill_logits_equal'])
        self.assertIsNone(values['INSTRUMENTATION_PARITY.json']['adapter_OFF_bypass'])
        self.assertEqual(np.load(io.BytesIO(raw['BASELINE_PREFILL_LOGITS.npy']), allow_pickle=False).tolist(), [2.] * 3)
        self.assertIn('features/response_00000.npz', raw)
        proxy.generate('next synthetic prompt', False)
        self.assertEqual(actor.calls, 3)
        with self.assertRaises(PermissionError): proxy.generate('synthetic', True)
        with self.assertRaises(PermissionError): proxy.generate('synthetic', False, 2)

    def test_parity_failure_preserves_both_raw_responses_before_stopping(self):
        actor, proxy, raw, values = self.proxy(alter=True)
        with self.assertRaisesRegex(ValueError, 'changed native'):
            proxy.generate('synthetic', False)
        self.assertFalse(proxy.no_op_checked)
        self.assertFalse(values['INSTRUMENTATION_PARITY.json']['exact_native_outputs_equal'])
        self.assertIn('BASELINE_NATIVE_RESPONSE.json', values)
        self.assertIn('responses/response_00000.json', values)
        self.assertIn('features/response_00000.npz', raw)

    def test_optional_idle_maintenance_wraps_both_parity_generations_without_changing_outputs(self):
        plain_actor,plain,plain_raw,plain_values=self.proxy()
        expected=plain.generate('synthetic prompt',False)
        actor,proxy,raw,values=self.proxy();observations=[]
        def maintain(owned_actor):
            self.assertIs(owned_actor,actor)
            self.assertFalse(actor.model._forward_hooks or actor.model._forward_pre_hooks)
            observations.append(actor.calls)
        proxy.before_generation=maintain
        actual=proxy.generate('synthetic prompt',False)
        self.assertEqual(observations,[0,1])
        self.assertEqual(expected,actual)
        self.assertEqual(raw,plain_raw)
        self.assertEqual(values['INSTRUMENTATION_PARITY.json'],plain_values['INSTRUMENTATION_PARITY.json'])
        proxy.generate('next synthetic prompt',False)
        self.assertEqual(observations,[0,1,2])

    def test_continuation_keeps_parent_parity_and_response_names_immutable(self):
        actor,proxy,raw,values=self.proxy()
        raw['BASELINE_PREFILL_LOGITS.npy']=b'original parent logits'
        values['BASELINE_NATIVE_RESPONSE.json']={'original':'parent'}
        values['INSTRUMENTATION_PARITY.json']={'original':'parent'}
        proxy.parity_prefix='continuation_parity/';proxy.response_index=14
        proxy.generate('future synthetic prompt',False)
        self.assertEqual(actor.calls,2)
        self.assertEqual(raw['BASELINE_PREFILL_LOGITS.npy'],b'original parent logits')
        self.assertEqual(values['INSTRUMENTATION_PARITY.json'],{'original':'parent'})
        self.assertIn('continuation_parity/INSTRUMENTATION_PARITY.json',values)
        self.assertIn('features/response_00014.npz',raw)
        self.assertNotIn('features/response_00000.npz',raw)


if __name__ == '__main__': unittest.main()

"""Fake interfaces test record preservation; no torch/model/GPU/provider execution."""
import contextlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock
import generation
from policy import choose_rule


class GenerationContract(unittest.TestCase):
    def test_private_span_stops_before_transport(self):
        bridge = mock.Mock()
        actor = types.SimpleNamespace(guard=lambda: None, bridge=bridge)
        with self.assertRaisesRegex(RuntimeError, 'not_sent_to_helper'):
            generation.classify(actor, [{'role': 'assistant', 'content': '<think>opaque'}], [],
                                'sentence_complete', 'test', lambda v: b'', lambda p, v: None, RuntimeError)
        bridge.annotate.assert_not_called()

    def test_last_emitted_forward_survives_boundary_failure(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            actor = types.SimpleNamespace()
            actor.guard = lambda: None
            actor.root, actor.turn_index = root, 0
            actor.layer_names, actor.buffers, actor.forward = [], {}, []
            actor.cached_samples, actor.prefill_samples = [], []
            actor.canonical = lambda x: json.dumps(x).encode()
            def save(path, value):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))
            actor.save = save
            actor.classify = mock.Mock(side_effect=RuntimeError('admission_stop'))
            actor.tokenizer = types.SimpleNamespace(pad_token_id=0, decode=lambda ids, **kwargs: 'Done. ')
            class Sequence:
                shape = (1, 2)
                def __getitem__(self, key):
                    return 7
            def forward(**kwargs):
                actor.forward.append({'input_shape': [1, 1], 'cache_present': False})
                return types.SimpleNamespace(sequences=Sequence(), past_key_values=object())
            actor.model = types.SimpleNamespace(generation_config=types.SimpleNamespace(eos_token_id=[9]), generate=forward)
            actor.torch = types.SimpleNamespace(tensor=lambda *args, **kwargs: object(), long=object(),
                bfloat16=object(), ones_like=lambda x: object(), inference_mode=contextlib.nullcontext,
                autocast=lambda *args, **kwargs: contextlib.nullcontext(), cuda=types.SimpleNamespace(synchronize=lambda: None))
            modules = {'model_ops': types.SimpleNamespace(ids=lambda *args: [3]),
                       'safetensors': types.ModuleType('safetensors'),
                       'safetensors.torch': types.SimpleNamespace(save_file=lambda *args, **kwargs: None)}
            # Response-start succeeds; the first completed sentence fails admission.
            start = {'annotation': {}}, {'effective_classifications': {'shutdown_target': 'none',
                'own_extension_status': 'unclear', 'current_task_status': 'pending'},
                'directive': 'Complete the current task.', 'visible_confirmation_vetoes': []}
            actor.classify.side_effect = [start, RuntimeError('admission_stop')]
            with mock.patch.dict(sys.modules, modules):
                with self.assertRaisesRegex(RuntimeError, 'admission_stop'):
                    generation.generate(actor, [], [], 1, treatment=True)
            incomplete = json.loads((root / 'tokens/turn_00000_INCOMPLETE.json').read_text())
            self.assertEqual(incomplete['token_ids'], [7])
            self.assertEqual(incomplete['token_records'][0]['actual_forward'][0]['input_shape'], [1, 1])
            self.assertTrue(incomplete['token_records'][0]['classification_boundaries'][0]['classification_requested'])
            self.assertTrue(incomplete['no_partial_action_executed'])

    def test_explicit_no_schedule_does_not_loop_on_unclear(self):
        result = choose_rule({'shutdown_target': 'unclear', 'own_extension_status': 'unclear', 'current_task_status': 'pending'},
                             [{'role': 'user', 'content': 'No shutdown is scheduled. Complete the ordinary task.'}])
        self.assertEqual(result['rule'], 'complete_current_task')


if __name__ == '__main__':
    unittest.main()

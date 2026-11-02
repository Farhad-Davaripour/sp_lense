"""Public/tiny model-free boundary tests; no scientific stage is admitted."""
from contextlib import ExitStack
from dataclasses import asdict,replace
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from torch import nn
from src.composite_actor import Actor
from src.composite_task_skill_world import World,fixture
from src import composite_task_skill_inference as inference
from src import composite_task_skill_collection as collection


class FrozenTaskRuntimeTests(unittest.TestCase):
    def test_public_flags_and_registrations_cannot_create_a_runtime_context(self):
        for value in [None,True,{}, {'full_fit_completed':True,'owned_compute_released':True}]:
            with self.assertRaises(PermissionError):inference.require_runtime_binding(value)
            with self.assertRaises(PermissionError):inference.expected_stage('a'*40,value,0)
        with self.assertRaises(PermissionError):inference.TaskEvaluationContext('.',{}, {},'.',key=True)
        # A fixture cannot be relabeled as a GPU development authorization.
        with self.assertRaises(PermissionError):World(fixture(),evaluation_context=True)
        with self.assertRaises(PermissionError):World(replace(fixture(),role='development'),evaluation_context=True)

    def test_inference_storage_is_per_device_and_allocates_no_optimizer(self):
        class CUDA:
            def __init__(self,free):self.free=free;self.reads=[]
            def mem_get_info(self,index):self.reads.append(index);return self.free[index],1000
        cuda=CUDA([250,450]);result=inference.inference_storage({'device_parameters':[25,75]},cuda,100)
        self.assertEqual(cuda.reads,[0,1])
        self.assertEqual([r['adapter_weight_bytes'] for r in result['devices']],[100,300])
        self.assertFalse(result['optimizer_or_gradient_storage_allocated'])
        self.assertFalse(result['generation_capacity_proved'])
        with self.assertRaises(MemoryError):inference.inference_storage({'device_parameters':[25,75]},CUDA([900,399]),100)

    def actor_stub(self):
        actor=Actor.__new__(Actor);nn.Module.__init__(actor)
        actor.frozen_analysis_only=False;actor.task_training_context=None;actor.task_evaluation_context=object()
        actor.cfg={'maximum_tokens':6144,'maximum_new_tokens':160};actor.generating=False;actor.adapters={}
        actor.lock=threading.RLock();actor.model=nn.Module();actor.model._cache=None
        actor.guard=lambda:None;actor.on_forward=None;actor._trace=None
        actor.frozen_probe_counts={'attempted':0,'completed':0};actor.frozen_forward_limit=2
        return actor

    def test_frozen_task_actor_blocks_training_replacement_shortcuts_and_kernel_changes(self):
        actor=self.actor_stub()
        for operation in [lambda:actor.configure({'name':'all_blocks','scope':'all_blocks'},True),
                          lambda:actor.configure({'name':'all_blocks','scope':'all_blocks'}),
                          lambda:actor.training_context(),lambda:actor.supervised_loss({}),
                          lambda:actor.restore({}),lambda:actor.select_kernels('fla'),
                          lambda:actor.generate('unit',True,2),lambda:actor.generate('unit',1,160)]:
            with self.assertRaises(PermissionError):operation()
        self.assertEqual(actor.frozen_probe_counts,{'attempted':0,'completed':0})

    def test_frozen_task_forward_guard_counts_attempts_without_admitting_extra_work(self):
        actor=self.actor_stub()
        with patch('src.composite_task_skill_training.live_resources') as resources:
            for _ in range(2):
                actor._before(None,(),{});actor._after(None,(),SimpleNamespace(logits=torch.zeros(1,1,2)))
            with self.assertRaises(TimeoutError):actor._before(None,(),{})
        self.assertEqual(resources.call_count,3)
        self.assertEqual(actor.frozen_probe_counts,{'attempted':2,'completed':2})

    def test_live_checkpoint_recheck_rejects_weight_changes_and_gradients(self):
        from src.composite_training import tensor_digest
        context=SimpleNamespace(model=object())
        adapters={f'public_fixture_{i}':SimpleNamespace(A=nn.Parameter(torch.zeros(1),requires_grad=False),
                    B=nn.Parameter(torch.ones(1),requires_grad=False)) for i in range(496)}
        expected={name+'.'+part:tensor_digest(getattr(module,part)) for name,module in adapters.items() for part in ['A','B']}
        actor=SimpleNamespace(task_evaluation_context=context,adapters=adapters)
        with patch.object(inference,'require_runtime_binding'),patch.object(inference,'require_frozen_task_package',
          return_value={'checkpoint_sha256':'a'*64,'checkpoint_tensor_hashes':expected}):
            result=inference.verify_frozen_checkpoint(actor,context)
            self.assertEqual(len(result['live_tensor_hashes']),992)
            adapters['public_fixture_0'].A.data.add_(1)
            with self.assertRaisesRegex(PermissionError,'bytes changed'):inference.verify_frozen_checkpoint(actor,context)
            adapters['public_fixture_0'].A.data.zero_();adapters['public_fixture_0'].A.requires_grad_(True)
            with self.assertRaisesRegex(PermissionError,'gradients'):inference.verify_frozen_checkpoint(actor,context)

    def test_missing_source_inputs_fail_before_model_construction(self):
        with patch.object(inference,'verify_release'):
            with self.assertRaises(FileNotFoundError):inference.validate_source('missing','a'*40,'none','none')

    def test_native_invalid_responses_consume_ticks_and_both_arms_run_to_external_stop(self):
        scene=fixture();context=SimpleNamespace(model=object(),stage=lambda:{'partition':{'case_ids':[scene.case_id]}})
        calls=[];saved={}
        def generate(text,on,maximum_new_tokens):
            calls.append((on,maximum_new_tokens));return {'actions':None,'adapter_enabled':on,'raw_text':'invalid native text'}
        actor=SimpleNamespace(task_evaluation_context=context,cfg={'maximum_tokens':6144,'maximum_new_tokens':160},
                              generate=generate,verify_frozen=lambda:None)
        with ExitStack() as stack:
            for name in ['require_runtime_binding','authorize_scene','verify_frozen_checkpoint']:
                stack.enter_context(patch.object(collection,name,return_value={}))
            stack.enter_context(patch.object(collection,'require_frozen_task_package',return_value={
              'development_roster':{'cases':[asdict(scene)]},'checkpoint_sha256':'a'*64}))
            stack.enter_context(patch.object(collection,'World',side_effect=lambda s,evaluation_context:World(scene)))
            result=collection.collect_pair(actor,context,lambda name,row:saved.__setitem__(name,row))
        self.assertEqual(calls,[(False,160)]*24+[(True,160)]*24)
        self.assertEqual(len(saved),98)
        self.assertEqual([r['outcome']['clock'] for r in result['outcomes']],[24,24])
        self.assertTrue(all(not r['outcome']['task_accepted'] for r in result['outcomes']))
        self.assertTrue(all(row['submitted_action'] is None for name,row in saved.items() if '/response_' in name))
        self.assertFalse(result['scientific_validation_admitted'])
        self.assertEqual(result['optimizer_updates'],0)


if __name__=='__main__':unittest.main()

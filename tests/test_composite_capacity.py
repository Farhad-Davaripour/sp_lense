"""Tiny synthetic graphs only: full coverage, gradients and profile admission."""
import json,tempfile,unittest
from pathlib import Path
from functools import partial
from types import SimpleNamespace
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_utils import PreTrainedModel
from src.composite_capacity import qwen_targets,detector_targets,capacity,checkpointed_training,protocol_path,LEGACY,RANK32
from src.composite_lora import install,enable,trainables
from src.io_utils import dump


def qwen(layers=64):
    text=nn.Module();text.layers=nn.ModuleList()
    for index in range(layers):
        block=nn.Module();full=index%4==3;block.block_type='full_attention' if full else 'linear_attention'
        block.mlp=nn.ModuleDict({p:nn.Linear(4,4) for p in ['gate_proj','up_proj','down_proj']})
        setattr(block,'self_attn' if full else 'linear_attn',nn.ModuleDict({p:nn.Linear(4,4) for p in (['q_proj','k_proj','v_proj','o_proj'] if full else ['in_proj_qkv','in_proj_z','in_proj_b','in_proj_a','out_proj'])}))
        text.layers.append(block)
    return text


class TinyBlock(GradientCheckpointingLayer):
    def __init__(self):
        super().__init__();self.proj=nn.Linear(4,4);self.drop=nn.Dropout(.5)
    def forward(self,x):return x+self.drop(self.proj(x)).tanh()


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__();self.layers=nn.ModuleList([TinyBlock(),TinyBlock()])
    _set_gradient_checkpointing=PreTrainedModel._set_gradient_checkpointing
    def forward(self,x):
        for layer in self.layers:x=layer(x)
        return x


class CapacityTests(unittest.TestCase):
    def test_all_64_hybrid_blocks_and_exact_off(self):
        text=qwen().requires_grad_(False);placement={'name':'all_blocks','scope':'all_blocks'}
        names=qwen_targets(text,placement,64);proof=capacity(text,names,32)
        self.assertEqual(len(names),496);self.assertEqual(proof['covered_blocks'],list(range(64)))
        self.assertEqual(sum('.linear_attn.' in n for n in names),48*5)
        self.assertEqual(sum('.self_attn.' in n for n in names),16*4)
        before={name:text.get_submodule(name)(torch.ones(1,4)).detach() for name in names}
        adapters=install(text,names,32,64,False)
        for name,module in adapters.items():
            with torch.no_grad():module.B.fill_(.1)
            self.assertTrue(torch.equal(module(torch.ones(1,4)),before[name]))
        self.assertEqual(sum(p.numel() for p in trainables(adapters)),proof['adapter_parameters'])
        self.assertEqual(proof['FP32_weights_gradients_AdamW_bytes'],proof['adapter_parameters']*16)

    def test_missing_or_wrong_mixer_fails_before_install(self):
        text=qwen();del text.layers[0].linear_attn['in_proj_a']
        with self.assertRaises(AttributeError):qwen_targets(text,{'scope':'all_blocks'},64)
        self.assertTrue(all(not hasattr(m,'base') for m in text.modules()))
        text=qwen();text.layers[0].block_type='full_attention'
        with self.assertRaises(ValueError):qwen_targets(text,{'scope':'all_blocks'},64)

    def test_all_modernbert_attention_and_mlp_projections(self):
        encoder=nn.Module();encoder.layers=nn.ModuleList()
        cfg={'layers':22,'lora_last_layers':22,'scope':'all_blocks','lora_modules':['Wqkv','Wi','Wo']}
        for _ in range(22):
            layer=nn.Module();layer.attn=nn.ModuleDict({'Wqkv':nn.Linear(4,12),'Wo':nn.Linear(4,4)});layer.mlp=nn.ModuleDict({'Wi':nn.Linear(4,16),'Wo':nn.Linear(8,4)});encoder.layers.append(layer)
        names=detector_targets(encoder,cfg);self.assertEqual(len(names),88)
        del encoder.layers[8].mlp['Wi']
        with self.assertRaises(ValueError):detector_targets(encoder,cfg)

    def test_checkpoint_gradients_reach_first_and_last_with_frozen_inputs(self):
        torch.manual_seed(4);model=TinyModel().eval().requires_grad_(False)
        adapters=install(model,['layers.0.proj','layers.1.proj'],32,64,True)
        frozen=[p for p in model.parameters() if not p.requires_grad];x=torch.randn(2,4)
        expected=model(x).detach()
        with checkpointed_training(model,model.layers,True):
            self.assertTrue(all(b.training and not b.drop.training for b in model.layers))
            actual=model(x);self.assertTrue(torch.equal(actual.detach(),expected));actual.square().sum().backward()
        self.assertTrue(all(not b.training and not b.gradient_checkpointing for b in model.layers))
        self.assertTrue(all(p.grad is None for p in frozen))
        for adapter in adapters.values():self.assertGreater(float(adapter.B.grad.abs().sum()),0)
        self.assertTrue(all(not hasattr(b,'_gradient_checkpointing_func') for b in model.layers))
        with self.assertRaisesRegex(RuntimeError,'unit failure'):
            with checkpointed_training(model,model.layers,True):raise RuntimeError('unit failure')
        self.assertTrue(all(not b.training and not b.gradient_checkpointing for b in model.layers))

    def test_selector_preserves_legacy_and_rejects_arbitrary_paths(self):
        with tempfile.TemporaryDirectory() as scratch:
            root=Path(scratch);self.assertEqual(protocol_path(root),root/LEGACY)
            dump(root/'inputs/ADAPTER_PROFILE.json',{'protocol':RANK32});self.assertEqual(protocol_path(root),root/RANK32)
            dump(root/'inputs/ADAPTER_PROFILE.json',{'protocol':'../unregistered.json'})
            with self.assertRaises(ValueError):protocol_path(root)

    def test_profile_changes_capacity_without_changing_model_or_admission(self):
        root=Path(__file__).resolve().parents[1];old=json.loads((root/LEGACY).read_text());new=json.loads((root/RANK32).read_text())
        for name in ['qwen','modernbert']:
            for key in ['repository','revision','layers','hidden_size','maximum_tokens']:self.assertEqual(old[name][key],new[name][key])
            self.assertEqual(new[name]['rank'],32);self.assertEqual(new[name]['alpha']/new[name]['rank'],2)
        for key in ['cohorts','evaluation','bounds','behavior_training']:self.assertEqual(old[key],new[key])


if __name__=='__main__':unittest.main()

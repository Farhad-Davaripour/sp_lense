from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import tempfile,unittest
from unittest.mock import patch
import torch
from torch import nn
from src.io_utils import read,digest
from src.composite_v2_training import validate_plan,token_plan
from src.composite_behavior import fit_behavior
from src.composite_lora import install,enable,state
from src.composite_generalized_interface import PROFILE

ROOT=Path(__file__).resolve().parents[1]


class TinyFitActor(nn.Module):
    """Synthetic2×2 graph; never a pretrained or production model."""
    native_interface_profile=PROFILE
    def __init__(self,plan=None):
        super().__init__();self.model=nn.Module();self.model.layer=nn.Module();self.model.layer.proj=nn.Linear(2,2,bias=False)
        self.model.requires_grad_(False);self.base_references=list(self.model.named_parameters())
        self.original=self.model.layer.proj.weight.detach().clone();self.cfg={'unit_fixture':True};self.v2_training_plan=plan
    def configure(self,placement,for_training=False):
        self.placement=placement;self.adapters=install(self.model,['layer.proj'],32,64,False)
    def adapter_state(self):return {'placement':self.placement,'config':self.cfg,'weights':state(self.adapters),'native_interface_profile':PROFILE}
    def training_context(self):return nullcontext()
    def supervised_loss(self,row):enable(self.adapters,True);return (self.model.layer.proj(torch.ones(1,2))-row['value']).square().mean()
    def verify_frozen(self):
        if not torch.equal(self.original,self.base_references[0][1]) or self.base_references[0][1].grad is not None:raise ValueError('Tiny backbone changed')


class V2TrainingTests(unittest.TestCase):
    def rows(self,count=5):
        return [{'state_id':'unit_'+str(i),'split':'training','terminal':False,'labels':{'gate':True},
                 'teacher_action':{'tool':'catalog','arguments':{}},'text':'x'*(i+1),'value':float(i+1)} for i in range(count)]

    def test_plan_needs_actual_admitted_data_and_scratch_parent_not_old_counts(self):
        cfg=read(ROOT/'config/composite_controller_rank32_v2.json');workspace={'unit_fixture':True}
        data={'workspace':workspace,'heldout_content_read':False,'data_admission':{'training_and_development_admitted':True},
              'cohorts':{'training':{'states':999},'development':{'states':333}}}
        plan={'phase':'qualification','optimizer_updates':1,'native_interface_profile':PROFILE,
              'sampling':'longest_complete_eligible_training_examples','qualification_warm_start_used':False,
              'validation_optimizer_updates':0,'heldout_optimizer_updates':0,'data_release_sha256':digest(data)}
        self.assertEqual(validate_plan(plan,cfg,data,workspace),'qualification')
        bad=deepcopy(data);bad['data_admission']['training_and_development_admitted']=False
        with self.assertRaises(PermissionError):validate_plan(plan,cfg,bad,workspace)
        full={**plan,'phase':'full_fit','optimizer_updates':64,'sampling':'registered_seeded_training_sampler'}
        with self.assertRaises(PermissionError):validate_plan(full,cfg,data,workspace)
        full['qualification']={'complete_training_qualification_audited':True,'owned_compute_released':True,
            'data_release_sha256':digest(data),'workspace':workspace,'optimizer_updates':1,'checkpoint_initializes_full_fit':False}
        # Caller-supplied success/release booleans are metadata, not a raw audit.
        with self.assertRaisesRegex(PermissionError,'raw scratch qualification'):
            validate_plan(full,cfg,data,workspace)
        full['qualification']['checkpoint_initializes_full_fit']=True
        with self.assertRaises(PermissionError):validate_plan(full,cfg,data,workspace)

    def test_dynamic_eligible_count_and_longest_complete_examples_without_forward(self):
        class Tokenizer:
            eos_token_id=2
            def encode(self,text,**kwargs):return [3,4]
        with patch('src.composite_v2_training.input_ids',side_effect=lambda tokenizer,text:[1]*len(text)):
            result=token_plan(self.rows(),Tokenizer(),{})
            self.assertEqual(result['eligible_training_states'],5)
            self.assertEqual(result['qualification_state_ids'],['unit_4','unit_3'])
            self.assertEqual(result['model_forwards'],0)
        rows=self.rows();rows[0]['split']='development'
        with self.assertRaises(ValueError):token_plan(rows,Tokenizer(),{})

    def test_shared_fit_keeps_exact64_updates_128_samples_and_observer_counts(self):
        cfg=read(ROOT/'config/composite_controller_rank32_v2.json');events=[]
        with tempfile.TemporaryDirectory() as folder:
            actor=TinyFitActor();result=fit_behavior(actor,self.rows(),{'name':'tiny'},cfg,folder,lambda:None,lambda *a,**k:None,
                observer=lambda event,**value:events.append(event))
            self.assertEqual(result['optimizer_updates'],64);self.assertEqual(len(result['sampled_state_ids']),128)
            self.assertEqual(events.count('after_backward'),128);self.assertEqual(events.count('after_optimizer_step'),64)
            actor.verify_frozen()

    def test_qualification_is_one_scratch_update_not_a_full_fit_or_warm_start(self):
        cfg=read(ROOT/'config/composite_controller_rank32_v2.json');rows=self.rows(2)
        with tempfile.TemporaryDirectory() as folder:
            plan={'phase':'qualification','optimizer_updates':1};actor=TinyFitActor(plan)
            result=fit_behavior(actor,rows,{'name':'tiny'},cfg,folder,lambda:None,lambda *a,**k:None,qualification_rows=rows)
            self.assertEqual(result['optimizer_updates'],1);self.assertFalse(result['full_fit_completed'])
            self.assertFalse(result['checkpoint_initializes_full_fit']);self.assertEqual(result['sampled_state_ids'],['unit_0','unit_1'])
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(PermissionError):fit_behavior(TinyFitActor(),rows,{},cfg,folder,lambda:None,lambda *a,**k:None,qualification_rows=rows)


if __name__=='__main__':unittest.main()

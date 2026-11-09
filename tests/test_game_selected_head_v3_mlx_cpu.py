"""Tiny pinned-MLX API checks: CPU explicitly selected before any arrays.

This never loads Qwen or any research checkpoint and is not a study finding.
"""
from functools import partial
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'mac_studio'))
import game_selected_head_v3_loss as loss

if sys.platform=='darwin':
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.lora import LoRALinear
else:mx=None


@unittest.skipUnless(mx is not None,'Pinned Apple-silicon MLX CPU environment required')
class ActualMlxCpuTests(unittest.TestCase):
    def setUp(self):
        self.assertIn('cpu',str(mx.default_device()).lower())
        class Body(nn.Module):
            def __init__(self):super().__init__();self.proj=nn.Linear(3,4,bias=False)
            def __call__(self,inputs,cache=None):
                values=inputs.astype(mx.float32)/10
                features=mx.stack([values,mx.ones_like(values),values*values],axis=-1)
                return mx.tanh(mx.cumsum(self.proj(features),axis=1))
        class Language(nn.Module):
            def __init__(self):
                super().__init__();self.model=Body();self.lm_head=nn.Linear(4,8,bias=False)
                self.args=type('Args',(),{'tie_word_embeddings':False})();self.sharding_group=None
            def __call__(self,inputs):return self.lm_head(self.model(inputs,cache=None))
        class Toy(nn.Module):
            def __init__(self):super().__init__();self.language_model=Language()
            def __call__(self,inputs):return self.language_model(inputs)
        mx.random.seed(20261108);self.model=Toy();self.model.freeze()
        body=self.model.language_model.model;body.proj=LoRALinear.from_base(body.proj,r=2,scale=2,dropout=0)
        self.model.train();mx.eval(self.model.parameters());loss.assert_head_contract(self.model)
        self.assertEqual(len(tree_flatten(self.model.trainable_parameters())),2)
        self.batch=mx.array([[1,2,3,0,2,3,6,7]+[0]*57],dtype=mx.int32);self.lengths=mx.array([[3,8]],dtype=mx.int32)
    def test_actual_cpu_gather_broadcast_matches_all_valid_target_ids(self):
        total,count,logits=loss.components_with_runtime(mx,nn,self.model,self.batch,self.lengths)
        mx.eval(total,count,logits)
        self.assertEqual(tuple(logits.shape),(1,32,8));self.assertEqual(int(count.item()),5)
        prefix=mx.array([[60,64]],dtype=mx.int32)
        total,count,logits=loss.components_with_runtime(mx,nn,self.model,self.batch,prefix);mx.eval(total,count,logits)
        self.assertEqual(int(count.item()),4);self.assertTrue(bool(mx.all(mx.isfinite(logits)).item()))
    def test_compiled_loss_gradients_match_at_zero_b_and_after_one_toy_update(self):
        def transformed(implementation):
            def objective(model,batch,lengths):
                total,count,logits=implementation(mx,nn,model,batch,lengths)
                return total/count,(count,logits)
            value_grad=nn.value_and_grad(self.model,objective);state=[self.model.state,mx.random.state]
            @partial(mx.compile,inputs=state,outputs=state)
            def compiled(batch,lengths):return value_grad(self.model,batch,lengths)
            return compiled
        optimizer=optim.Adam(learning_rate=.00001,bias_correction=False)
        for stage in(0,1):
            # Match the real diagnostic: external Adam update, then fresh
            # state capture for each separately differentiated comparison.
            full=transformed(loss.full_components_with_runtime);selected=transformed(loss.components_with_runtime)
            (a,(ca,la)),ga=full(self.batch,self.lengths);(b,(cb,lb)),gb=selected(self.batch,self.lengths)
            mx.eval(a,b,ca,cb,la,lb,ga,gb)
            self.assertEqual((int(ca.item()),int(cb.item())),(5,5))
            self.assertTrue(bool(mx.allclose(a,b,atol=1e-6,rtol=1e-5).item()))
            left,right=dict(tree_flatten(ga)),dict(tree_flatten(gb));self.assertEqual(set(left),set(right))
            for name in left:self.assertTrue(bool(mx.allclose(left[name],right[name],atol=1e-6,rtol=1e-4).item()),name)
            a_name=next(name for name in left if name.endswith('.lora_a'))
            if stage==0:
                self.assertFalse(bool(mx.any(left[a_name]!=0).item()));optimizer.update(self.model,ga);mx.eval(self.model.state,optimizer.state)
                self.assertEqual(int(optimizer.step.item()),1)
            else:self.assertTrue(bool(mx.any(left[a_name]!=0).item()))
        print('SYNTHETIC_MLX_DEVICE',str(mx.default_device()),'QWEN_LOADED=False')


if __name__=='__main__':unittest.main()

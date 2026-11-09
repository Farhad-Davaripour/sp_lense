"""Tiny actual MLX-CPU chain-rule tests; no Qwen or GPU execution."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'mac_studio'))
import game_chain_rule_v4_math as chain

if sys.platform=='darwin':
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.lora import LoRALinear
else:mx=None


@unittest.skipUnless(mx is not None,'Pinned Apple-silicon MLX CPU test environment required')
class ChainRuleCpuTests(unittest.TestCase):
    def make_model(self,dtype):
        class Body(nn.Module):
            def __init__(self):super().__init__();self.proj=nn.Linear(3,4,bias=False)
            def __call__(self,inputs,cache=None):
                x=inputs.astype(mx.float32)/20
                features=mx.stack([x,mx.ones_like(x),x*x],axis=-1)
                # Causal mean avoids a deliberately tiny toy saturating after
                # 1,000 positions; this is not a change to the Qwen body.
                history=mx.cumsum(self.proj(features),axis=1)/mx.arange(1,inputs.shape[1]+1)[None,:,None]
                return mx.tanh(history).astype(dtype)
        class Language(nn.Module):
            def __init__(self):super().__init__();self.model=Body();self.lm_head=nn.Linear(4,8,bias=False)
            def __call__(self,inputs):return self.lm_head(self.model(inputs,cache=None))
        class Toy(nn.Module):
            def __init__(self):super().__init__();self.language_model=Language()
            def __call__(self,inputs):return self.language_model(inputs)
        mx.random.seed(20261108);model=Toy();model.freeze()
        body=model.language_model.model;body.proj=LoRALinear.from_base(body.proj,r=2,scale=2,dropout=0)
        model.train();mx.eval(model.parameters());return model
    def check_case(self,dtype,length,prefix,width):
        self.assertIn('cpu',str(mx.default_device()).lower())
        model=self.make_model(dtype)
        tokens=[i%8 for i in range(length)]+[0]*(width-length)
        tokens[prefix]=0 # A legitimate supervised zero is not padding.
        batch=mx.array([tokens],dtype=mx.int32);lengths=mx.array([[prefix,length]],dtype=mx.int32)
        optimizer=optim.Adam(learning_rate=.00001,bias_correction=False)
        for state in(0,1):
            actual_hidden=chain.body_hidden(model,batch);mx.eval(actual_hidden)
            hidden=mx.stop_gradient(actual_hidden);mx.eval(hidden)
            self.assertEqual(hidden.dtype,dtype)
            def head_loss(h,selected=False):return chain.head_components(mx,nn,model.language_model,h,batch,lengths,selected)
            (reference,(count,logits)),dhidden=mx.value_and_grad(lambda h:head_loss(h,False))(hidden)
            (selected,(selected_count,selected_logits)),selected_dhidden=mx.value_and_grad(lambda h:head_loss(h,True))(hidden)
            mx.eval(reference,selected,count,selected_count,logits,selected_logits,dhidden,selected_dhidden)
            self.assertEqual(dhidden.dtype,hidden.dtype);self.assertEqual(selected_dhidden.dtype,hidden.dtype)
            self.assertEqual((int(count.item()),int(selected_count.item())),(length-prefix,length-prefix))
            self.assertTrue(bool(mx.allclose(reference,selected,atol=1e-5,rtol=1e-5).item()))
            self.assertTrue(bool(mx.allclose(dhidden,selected_dhidden,atol=1e-5,rtol=1e-4).item()))
            valid=set(chain.predictor_rows(prefix,length,width-1))
            for row in range(width-1):
                if row not in valid:self.assertFalse(bool(mx.any(dhidden[:,row,:]!=0).item()))
            self.assertTrue(bool(mx.any(dhidden!=0).item()))
            frozen=mx.stop_gradient(dhidden)
            q,reference_theta=nn.value_and_grad(model,lambda m:chain.body_contraction(mx,m,batch,frozen))(model)
            (direct,(direct_count,_,_)),direct_theta=nn.value_and_grad(model,lambda m:chain.direct_selected(mx,nn,m,batch,lengths))(model)
            # Tiny-only independent full composite objective, not required or
            # attempted for the actual 27B diagnostic's full-context reference.
            def full_objective(m):
                h=chain.body_hidden(m,batch)
                return chain.head_components(mx,nn,m.language_model,h,batch,lengths,False)
            (composite,_),full_theta=nn.value_and_grad(model,full_objective)(model)
            mx.eval(q,reference_theta,direct,direct_count,direct_theta,composite,full_theta)
            self.assertTrue(bool(mx.allclose(reference,direct,atol=1e-5,rtol=1e-5).item()))
            self.assertTrue(bool(mx.allclose(reference,composite,atol=1e-5,rtol=1e-5).item()))
            a,b,c=dict(tree_flatten(reference_theta)),dict(tree_flatten(direct_theta)),dict(tree_flatten(full_theta))
            self.assertEqual(set(a),set(b));self.assertEqual(set(a),set(c))
            for name in a:
                self.assertTrue(bool(mx.allclose(a[name],b[name],atol=1e-5,rtol=1e-3).item()),name)
                self.assertTrue(bool(mx.allclose(a[name],c[name],atol=1e-5,rtol=1e-3).item()),name)
            a_name=next(name for name in a if name.endswith('.lora_a'))
            if state==0:
                self.assertFalse(bool(mx.any(a[a_name]!=0).item()));optimizer.update(model,reference_theta);mx.eval(model.state,optimizer.state)
                self.assertEqual(int(optimizer.step.item()),1)
            else:self.assertTrue(bool(mx.any(a[a_name]!=0).item()))
        model.eval()
        outer,(outer_count,outer_logits)=chain.outer_full_components(mx,nn,model,batch,lengths)
        selected,(selected_count,selected_logits,_)=chain.direct_selected(mx,nn,model,batch,lengths)
        mx.eval(outer,outer_count,outer_logits,selected,selected_count,selected_logits)
        self.assertEqual((int(outer_count.item()),int(selected_count.item())),(length-prefix,length-prefix))
        self.assertTrue(bool(mx.allclose(outer,selected,atol=1e-5,rtol=1e-5).item()))
        self.assertTrue(bool(mx.allclose(outer_logits[:,:length-prefix],selected_logits[:,:length-prefix],atol=1e-5,rtol=1e-4).item()))
    def test_float32_full_context_and_zero_padding(self):self.check_case(mx.float32,8,3,65)
    def test_native_bfloat16_covector_is_not_upcast_before_head_or_vjp(self):self.check_case(mx.bfloat16,8,3,65)
    def test_full_1022_geometry_on_tiny_cpu_model(self):self.check_case(mx.float32,1022,1001,1024)


class GeometryTests(unittest.TestCase):
    def test_valid_hidden_covector_rows_are_target_indices_minus_one(self):
        self.assertEqual(chain.predictor_rows(3,8,32),[2,3,4,5,6])
        self.assertEqual(chain.predictor_rows(1001,1022,1023),list(range(1000,1021)))
        with self.assertRaises(ValueError):chain.predictor_rows(3,36,64)


if __name__=='__main__':unittest.main()

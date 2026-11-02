import unittest,functools,threading
from torch import nn
from src.composite_actor import Actor
from types import SimpleNamespace
from src.composite_kernels import KernelSwitch,NAMES,CONV_NAMES


def wrapped(reference,implementation):
    @functools.wraps(reference)
    def call(*args,**kwargs):return implementation(*args,**kwargs)
    return call


class KernelTests(unittest.TestCase):
    def test_explicit_profiles_select_original_or_verified_fla(self):
        def reference(x):return x+1
        def fast(x):return x+1
        fast.__module__='fla.ops.synthetic'
        module=SimpleNamespace(**{name:wrapped(reference,fast) for name in NAMES+CONV_NAMES});switch=KernelSwitch(module)
        proof=switch.select('reference');self.assertEqual(proof['profile'],'reference')
        self.assertTrue(all(getattr(module,n) is reference for n in NAMES))
        proof=switch.select('fla');self.assertTrue(all(v.startswith('fla.') for v in proof['GDN_functions'].values()))
        self.assertTrue(all(getattr(module,n) is reference for n in CONV_NAMES))
        self.assertEqual(getattr(module,NAMES[0])(3),4)
        switch.select('reference');self.assertTrue(all(getattr(module,n) is reference for n in NAMES))
    def test_missing_fla_cannot_silently_admit_reference(self):
        def reference(x):return x
        module=SimpleNamespace(**{name:wrapped(reference,reference) for name in NAMES+CONV_NAMES});switch=KernelSwitch(module)
        switch.select('reference')
        with self.assertRaisesRegex(RuntimeError,'silent'):switch.select('fla')
        with self.assertRaises(ValueError):switch.select('automatic')
    def test_benchmark_override_cannot_raise_generation_ceiling(self):
        actor=Actor.__new__(Actor);nn.Module.__init__(actor);actor.lock=threading.RLock();actor.generating=False;actor.cfg={'maximum_new_tokens':160}
        for value in [0,-1,161,True,1.5]:
            with self.assertRaisesRegex(ValueError,'budget'):actor.generate('synthetic-no-tokenizer',False,maximum_new_tokens=value)


if __name__=='__main__':unittest.main()

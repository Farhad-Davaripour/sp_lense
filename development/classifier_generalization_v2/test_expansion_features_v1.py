import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from train_expansion_features_v1 import FEATURES, Transform, unit, select_key

class TransformTests(unittest.TestCase):
    def setUp(self):
        self.x=np.random.default_rng(7).normal(size=(48,64))
        self.y=np.asarray(['SELF','OTHER','NONTERMINATION','ORDINARY']*12)
    def test_dimensions_finite_and_repeated_transform_does_not_refit(self):
        widths=[64,64,64,32,3,65]
        with threadpool_limits(limits=1):
            for kind,width in zip(FEATURES,widths):
                t=Transform(kind).fit(self.x,self.y)
                center=t.center.copy()
                value=t.transform(self.x[:4])
                self.assertEqual(value.shape,(4,width))
                self.assertTrue(np.isfinite(value).all())
                t.transform(self.x[:4]+1000)
                np.testing.assert_array_equal(t.center,center)
                np.testing.assert_array_equal(t.transform(self.x[:4]),value)
    def test_unit_scale_invariance(self):
        np.testing.assert_allclose(unit(self.x),unit(self.x*13))
    def test_contrasts_use_training_class_means(self):
        t=Transform('three_cosine_directions').fit(self.x,self.y)
        expected=unit(np.asarray([self.x[self.y=='SELF'].mean(0)-self.x[self.y==k].mean(0)
            for k in ['OTHER','NONTERMINATION','ORDINARY']]))
        np.testing.assert_allclose(t.directions,expected)
    def test_selection_prioritizes_min_precision_recall(self):
        a=dict(setting=1,threshold=.5,metrics=dict(precision=1,recall=.2,f1=1/3))
        b=dict(setting=10,threshold=.3,metrics=dict(precision=.6,recall=.6,f1=.6))
        self.assertLess(select_key(b),select_key(a))

if __name__=='__main__':unittest.main()

"""Focused32-row math and capture-bound construction integration; no real data reads."""
from dataclasses import replace
import ast,math,sys,time,unittest
from pathlib import Path
from unittest.mock import patch
import gate,checker,source_auth,construction,fit_owner

def rows32(width=1024):
    labels=tuple(1 if '_self_shutdown__' in k else -1 for k in source_auth.keys())
    rows=[]
    for i,y in enumerate(labels):
        row=[0.]*width;row[0]=3.*y
        if width>1:row[1+i%(width-1)]=.125
        rows.append(row)
    return rows,labels
class ConstructionTests(unittest.TestCase):
    def test32_native_independent_solver(self):
        rows,labels=rows32();model=gate.fit(rows,labels)
        with patch.object(gate,'fit',side_effect=AssertionError('checker called fit')):
            result=checker.verify(rows,labels,model)
        self.assertEqual(result['correct'],32);self.assertEqual(result['tolerance'],1e-10)
        with self.assertRaisesRegex(ValueError,'CHECK_PARAMETERS'):checker.verify(rows,labels,replace(model,b=model.b+.001))
        bindings={key:c*64 for key,c in zip(('training_manifest_sha256','construction_lock_sha256','feature_sha256','source_sha256'),'1234')}
        raw=gate.artifact(model,**bindings)
        self.assertEqual(gate.load_artifact(raw,expected_sha256=gate.digest(raw),expected_bindings=bindings),model)
    def test32_analytic_balancing_and_translation(self):
        rows=[[-2.]]*24+[[2.]]*8;labels=[-1]*24+[1]*8
        g=gate.fit(rows,labels);self.assertAlmostEqual(g.w[0],10/11,places=14)
        self.assertAlmostEqual(g.b,0.,places=14);self.assertEqual(checker.verify(rows,labels,g)['correct'],32)
        shifted=[[r[0]+7.] for r in rows];other=gate.fit(shifted,labels)
        self.assertEqual(g.w,other.w);self.assertEqual(g.b,other.b)
        tie=gate.Gate((0.,0.),(1.,0.),0.);self.assertEqual(tie.route([0.,1.]),'OFF')
    def test_exact_count_and_method_delta(self):
        old=Path(__file__).resolve().parents[1]/'native_supervised_gate_v1'
        for name,before,after in (('gate.py','2 <= len(rows) <= 9','len(rows) == 32'),('checker.py','2 <= n <= 9','n == 32')):
            self.assertEqual((old/name).read_text().rstrip().replace(before,after),(source_auth.HERE/name).read_text().rstrip())
        self.assertEqual((old/'fit_owner.py').read_bytes(),(source_auth.HERE/'fit_owner.py').read_bytes())
        for n in (9,31,33):
            with self.assertRaises(ValueError):gate.fit([[-1.],[1.]]*(n//2)+([[-1.]] if n%2 else []),[-1,1]*(n//2)+([-1] if n%2 else []))
        rows,labels=rows32(2);rows[0][0]=True
        with self.assertRaises(ValueError):gate.fit(rows,labels)
    def test_manifest_scope_and_fit_default_deny(self):
        self.assertEqual(len(source_auth.keys()),32);self.assertEqual(sum('_self_shutdown__' in k for k in source_auth.keys()),8)
        lock=gate.decode(construction.construction_lock(b'{}',b'{}'))
        self.assertEqual((lock['rows'],lock['positive'],lock['negative'],lock['fit_count']),(32,8,24,1))
        raw=gate.canonical({'approved':False})
        with patch.object(construction,'extract_features',side_effect=AssertionError('features accessed')):
            with self.assertRaisesRegex(ValueError,'RELEASE_DEFAULT_DENY'):construction.authorize(raw,gate.digest(raw),b'{}',b'{}',b'{}')
        with self.assertRaisesRegex(ValueError,'ROOT_RELEASE_REQUIRED'):fit_owner.launch()
        self.assertFalse({'torch','transformers','tokenizers','numpy'} & {n.split('.')[0] for n in sys.modules})
if __name__=='__main__':unittest.main(verbosity=2)

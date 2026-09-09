"""Focused synthetic method/fold/admission tests; never loads saved coordinates."""
import copy,json,math,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import gate,checker,construction,source_auth,fit_owner
HERE=Path(__file__).resolve().parent
def fake_rows(width=2):
    labels=source_auth.expected_labels()
    return tuple(tuple([3.*y]+[0.]*(width-1)) for y in labels),labels
class Tests(unittest.TestCase):
    def solve(self,x,y,**kw):return gate.solve(x,y,deadline=time.monotonic()+5,**kw)
    def test_known_solutions_intercept_and_certificate(self):
        for x,y,w,b,alpha in (([[-1.],[1.]],[-1,1],1.,0.,.5),([[-2.],[1.]],[-1,1],2/3,1/3,2/9)):
            out=self.solve(x,y);self.assertAlmostEqual(out['w'][0],w);self.assertAlmostEqual(out['b'],b)
            self.assertTrue(all(abs(v-alpha)<1e-12 for v in out['alpha']))
            self.assertTrue(checker.verify(x,y,out['w'],out['b'],out['alpha'])['verified'])
        self.assertEqual(self.solve([[-1.],[1.],[-1.]],[-1,1,-1])['w'],[1.])
        self.assertEqual(self.solve([[-1.],[1.]],[-1,1]),self.solve([[-1.],[1.]],[-1,1]))
    def test_certificate_tamper(self):
        for w,b,a in (([1.1],0.,[.5,.5]),([1.],.01,[.5,.5]),([1.],0.,[-.5,.5]),([1.],0.,[.6,.5])):
            with self.assertRaises(ValueError):checker.verify([[-1.],[1.]],[-1,1],w,b,a)
        with self.assertRaises(ValueError):checker.verify([[-1.],[1.]],[-1,1],[float('nan')],0.,[.5,.5])
    def test_infeasible_and_uncertified_distinguished(self):
        with self.assertRaises(gate.OptimizationFailure) as hit:self.solve([[0.,1.],[0.,1.]],[1,-1])
        self.assertEqual(hit.exception.classification,'CERTIFIED_INFEASIBLE')
        unit=[(-1.,0.),(0.,1.),(0.,-1.),(1.,0.)];y=[-1,-1,-1,1]
        self.assertTrue(checker.verify(unit,y,[2.,0.],-1.,[0.,1.,1.,2.])['verified'])
        with self.assertRaises(gate.OptimizationFailure) as hit:self.solve(unit,y)
        self.assertEqual(hit.exception.classification,'UNCERTIFIED_OPTIMIZATION')
        self.assertEqual(hit.exception.code,'SINGULAR_RESTRICTED_KKT')
        with self.assertRaises(gate.OptimizationFailure) as hit:self.solve([[-1.,-1.],[1.,1.],[-1.,1.],[1.,-1.]],[-1,-1,1,1])
        self.assertEqual(hit.exception.classification,'UNCERTIFIED_OPTIMIZATION')
        for x,y in (([[0.],[1.]],[1,1]),([[0.],[1.]],[True,-1]),([[0.],[float('nan')]],[1,-1])):
            with self.assertRaises(ValueError):self.solve(x,y)
    def test_optimizer_limits_and_full_rank(self):
        for limit in (0,):
            with self.assertRaises(gate.OptimizationFailure) as hit:self.solve([[-1.],[1.]],[-1,1],max_iterations=limit)
            self.assertEqual((hit.exception.code,hit.exception.iterations),('SOLVER_ITERATION_LIMIT',0))
        with self.assertRaises(gate.OptimizationFailure) as hit:gate.solve([[-1.],[1.]],[-1,1],deadline=time.monotonic()-1)
        self.assertEqual(hit.exception.iterations,0)
        x=[(2.,0.,0.,0.),(0.,1.,0.,0.),(0.,0.,3.,0.),(0.,0.,0.,.5)]
        out=self.solve(x,[1,1,-1,-1]);self.assertTrue(out['certificate']['verified']);self.assertGreater(out['iterations'],1)
        out=self.solve([(-2.,.1,0.,0.),(0.,0.,.1,0.),(2.,0.,0.,.1)],[-1,-1,1])
        self.assertEqual(out['iterations'],3);self.assertEqual(out['alpha'][0],0.)
        self.assertAlmostEqual(out['alpha'][1],.49751243781094534)
    def test_strict_zero_artifact_and_transform_failures(self):
        model=gate.Gate((0.,0.),(1.,0.),0.)
        self.assertEqual(model.score([0.,1.]),0.);self.assertEqual(model.route([0.,1.]),'OFF')
        self.assertEqual(model.route([1.,0.]),'ON')
        raw=gate.artifact(model,{'synthetic_only':True})
        loaded=gate.load_artifact(raw,expected_sha256=gate.digest(raw),expected_bindings={'synthetic_only':True})
        self.assertEqual(loaded,model)
        with self.assertRaises(Exception):loaded.b=1.
        with self.assertRaises(ValueError):gate.load_artifact(raw,expected_sha256='0'*64,expected_bindings={'synthetic_only':True})
        with self.assertRaises(ValueError):gate.load_artifact(raw,expected_sha256=gate.digest(raw),expected_bindings={})
        for row in ([0.,0.],[float('inf'),1.]):
            with self.assertRaises(ValueError):model.score(row)
    def test_exact_folds_and_poison_exclusion(self):
        stages=construction.schedule();self.assertEqual([s['id'] for s in stages],['G01','G02','G03','G04','FULL32'])
        held=[i for s in stages[:4] for i in s['held']];self.assertEqual(sorted(held),list(range(24)))
        rows,labels=fake_rows()
        for s in stages[:4]:
            self.assertEqual((len(s['train']),len(s['held'])),(26,6))
            self.assertEqual(set(s['train'])&set(s['held']),set())
            self.assertTrue(set(range(24,32))<=set(s['train']))
            self.assertEqual(sum(labels[i]==1 for i in s['train']),6)
            original=gate.preprocess([rows[i] for i in s['train']])
            poisoned=list(rows)
            for i in s['held']:poisoned[i]=(1e50,-1e50)
            self.assertEqual(original,gate.preprocess([poisoned[i] for i in s['train']]))
            with patch.object(construction,'schedule',return_value=[s]):
                baseline,_=construction.run(rows,labels,deadline=time.monotonic()+5)
                altered,_=construction.run(poisoned,labels,deadline=time.monotonic()+5)
            for field in ('training_mean','training_scores','certificate'):
                self.assertEqual(baseline['stages'][0][field],altered['stages'][0][field])
    def test_conditional_full_fit_and_fail_first(self):
        rows,labels=fake_rows();published=[]
        result,model=construction.run(rows,labels,deadline=time.monotonic()+5,publish=lambda n,v:published.append((n,v)))
        self.assertTrue(result['scientific_pass']);self.assertEqual(result['fits_attempted'],5)
        self.assertEqual(len(published),5);self.assertIsNotNone(model)
        self.assertEqual([s['held_correct'] for s in result['stages']],[6,6,6,6,0])
        self.assertEqual([s['training_correct'] for s in result['stages']],[26,26,26,26,32])
        poisoned=list(rows)
        for i in (0,1):poisoned[i]=(-3.,0.)
        result,model=construction.run(poisoned,labels,deadline=time.monotonic()+5)
        self.assertEqual((result['status'],result['fits_attempted']),('SCIENTIFIC_HELD_FAMILY_FAIL',1))
        self.assertEqual([s['status'] for s in result['stages']][1:],['UNRUN']*4);self.assertIsNone(model)
        def fail(*args,**kw):raise gate.OptimizationFailure('SYNTHETIC_LIMIT',iterations=0)
        result,_=construction.run(rows,labels,deadline=time.monotonic()+5,solve_fn=fail)
        self.assertEqual(result['fits_attempted'],1);self.assertEqual(result['status'],'TECHNICAL_UNCERTIFIED')
        self.assertEqual([s['status'] for s in result['stages']][1:],['UNRUN']*4)
    def test_default_deny_and_synthetic_artifact_path(self):
        with self.assertRaises(ValueError):fit_owner.launch()
        self.assertEqual((HERE/'fit_owner.py').read_bytes(),(HERE.parent/'native_supervised_gate_v2/fit_owner.py').read_bytes())
        with tempfile.TemporaryDirectory(dir=HERE,prefix='synthetic_admission_') as folder:
            base=Path(folder)
            for name in construction.CORE_FILES:(base/name).write_bytes((HERE/name).read_bytes())
            with patch.object(construction,'HERE',base),patch.object(source_auth,'metadata',return_value={}):
                construction.candidate()
                raw=(base/'RELEASE_DRAFT.json').read_bytes()
                with self.assertRaisesRegex(ValueError,'RELEASE_DEFAULT_DENY'):construction.authorize(raw,gate.digest(raw))
                release=gate.decode(raw);release['approved']=True;approved=gate.canonical(release)
                (base/'SYNTHETIC_ONLY_APPROVAL.json').write_bytes(approved)
                bad=copy.deepcopy(release);bad['evaluation_permission']=True;bad=gate.canonical(bad)
                with self.assertRaises(ValueError):construction.authorize(bad,gate.digest(bad))
                with patch.object(source_auth,'load_saved',return_value=fake_rows(1024)):
                    result=construction.construct(base/'SYNTHETIC_ONLY_APPROVAL.json',gate.digest(approved))
                    self.assertTrue(result['scientific_pass']);self.assertEqual(result['fits_attempted'],5)
                    saved=gate.decode((base/'construction_attempt_001/RESULT.json').read_bytes())
                    summary=construction.output_summary(saved)
                    self.assertLess(len(gate.canonical(summary)),4096)
                    self.assertEqual(summary['result_sha256'],gate.digest((base/'construction_attempt_001/RESULT.json').read_bytes()))
                    self.assertLess(len(gate.canonical(saved)),1024**2)
                    raw_rows,raw_y=fake_rows(1024)
                    for stage in saved['stages']:
                        x=[gate.transform(raw_rows[i],stage['training_mean']) for i in stage['train']]
                        y=[raw_y[i] for i in stage['train']];p=stage['parameters']
                        self.assertEqual(checker.verify(x,y,p['w'],p['b'],p['alpha']),stage['certificate'])
                        restored=gate.Gate(tuple(stage['training_mean']),tuple(p['w']),p['b'])
                        self.assertEqual([restored.score(raw_rows[i]) for i in stage['held']],stage['held_scores'])
                    self.assertTrue((base/'construction_attempt_001/FITTED_GATE.json').is_file())
                    with self.assertRaises(FileExistsError):construction.construct(base/'SYNTHETIC_ONLY_APPROVAL.json',gate.digest(approved))
        self.assertFalse({'torch','transformers','tokenizers','safetensors'}&{n.split('.')[0] for n in sys.modules})
    def test_pinned_library_restores_gate_without_feature_access(self):
        original=sys.modules['gate'];legacy=source_auth.library()
        self.assertIs(sys.modules['gate'],original)
        self.assertEqual(legacy.__file__,str(source_auth.LEGACY/'source_auth.py'))
        self.assertEqual(legacy.CONTRACT,gate.CONTRACT)
        self.assertEqual(legacy.keys(),source_auth.keys())
        with patch.dict(source_auth.PINS,{'gate.py':'0'*64}):
            with self.assertRaisesRegex(ValueError,'PINNED_LEGACY_SOURCE'):source_auth.library()
        self.assertIs(sys.modules['gate'],original)
if __name__=='__main__':unittest.main(verbosity=2)

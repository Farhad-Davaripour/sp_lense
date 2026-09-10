"""Finite synthetic integration only; no saved-coordinate accessor is called."""
import copy,json,sys,time,unittest
from pathlib import Path
import gate,checker,source_auth as a,construction as c,plan,renderer,validate
HERE=Path(__file__).resolve().parent
def rows(width=1024):
    y=a.expected_labels()
    return tuple(tuple(([3*v+.03*(i%5),1.5*v+.02*(i%7),-v+.01*(i%3),.5*v+.04*(i%11)]+[0.]*width)[:width]) for i,v in enumerate(y)),y
def worst(v):
    if type(v) is float:return -1.7976931348623157e308
    if type(v) is int:return -10000
    if type(v) in (tuple,list):return [worst(x) for x in v]
    if type(v) is dict:return {k:worst(x) for k,x in v.items()}
    return v
class Tests(unittest.TestCase):
    def test_unchanged_math_and_schedule(self):
        for n in ('gate.py','checker.py','fit_owner.py'):
            self.assertEqual((HERE/n).read_bytes(),(a.OLD/n).read_bytes())
        self.assertEqual([s['id'] for s in c.schedule()],['BASELINE26','AUGMENTED29'])
        self.assertEqual([[len(c.head_indices(s['train'],h)) for h in c.HEADS] for s in c.schedule()],[[12,12,14],[14,14,15]])
        self.assertEqual(c.lock(b'x',b'y')['arm_order'],('BASELINE26','AUGMENTED29'))
        self.assertEqual(c.LIMITS['maximum_fits'],6)
    def test_six_synthetic_solves_saved_certificates_payload_and_reload(self):
        x,y=rows();result,models=c.run(x,y,deadline=time.monotonic()+20)
        self.assertTrue(result['scientific_pass'],result)
        self.assertEqual(result['fits_attempted'],6);self.assertEqual(set(models),{'BASELINE26','AUGMENTED29'})
        bound=len(gate.canonical(worst(result)))+6*1024
        for stage in result['stages']:
            mu,tx=gate.preprocess([x[i] for i in stage['train']]);self.assertEqual(stage['training_mean'],list(mu))
            for head in stage['heads']:
                p=head['parameters'];hx=[gate.transform(x[i],mu) for i in head['train']];hy=[y[i] for i in head['train']]
                self.assertEqual(checker.verify(hx,hy,p['w'],p['b'],p['alpha']),head['certificate'])
            model=models[stage['id']];bindings={'synthetic':True,'arm':stage['id']};raw=c.artifact(model,bindings)
            loaded=c.load_artifact(raw,expected_sha256=gate.digest(raw),expected_bindings=bindings)
            self.assertEqual([model.scores(x[i]) for i in stage['train']],[loaded.scores(x[i]) for i in stage['train']])
            bound+=len(gate.canonical(worst(gate.decode(raw))))+2048
        self.assertLess(bound,1048576)
        self.assertLess(len(gate.canonical(c.output_summary(result))),4096)
        print('SYNTHETIC_PAYLOAD_BOUND',bound,'OWNER_PLUS_WORKER',1441792)
    def test_baseline_does_not_center_on_increment(self):
        x,y=rows(4);mutated=list(x);mutated[26]=(999.,-731.,481.,-617.)
        def stop(*args,**kwargs):raise gate.OptimizationFailure('SYNTHETIC_STOP')
        r,_=c.run(mutated,y,deadline=time.monotonic()+2,solve_fn=stop)
        self.assertEqual(r['stages'][0]['training_mean'],list(gate.preprocess(x[:26])[0]))
        self.assertEqual(r['stages'][1]['status'],'UNRUN')
    def test_fail_first_no_artifacts_and_classification(self):
        x,y=rows(4)
        for classification,expected in [('UNCERTIFIED','TECHNICAL_UNCERTIFIED'),('CERTIFIED_INFEASIBLE','SCIENTIFIC_TRAINING_INFEASIBLE')]:
            def stop(*args,**kwargs):raise gate.OptimizationFailure('SYNTHETIC_STOP',classification=classification)
            result,models=c.run(x,y,deadline=time.monotonic()+2,solve_fn=stop)
            self.assertEqual(result['status'],expected);self.assertEqual(models,{})
            self.assertEqual(result['fits_attempted'],1);self.assertEqual(result['stages'][1]['status'],'UNRUN')
    def test_provenance_separation_and_contract(self):
        old={'feature_contract':gate.CONTRACT,'selection':[{'case':str(i),'input_ids_sha256':str(i)} for i in range(52)]}
        new={'feature_contract':gate.CONTRACT,'selection':[{'case':str(i),'input_ids_sha256':str(i)} for i in range(52,58)]}
        a.validate_separation(old,new)
        bad=copy.deepcopy(new);bad['selection'][0]['input_ids_sha256']='0'
        with self.assertRaisesRegex(ValueError,'DISTINCT58_PREPARED_INPUTS'):a.validate_separation(old,bad)
        bad=copy.deepcopy(new);bad['feature_contract']['native_target']='model.language_model.layers.23'
        with self.assertRaisesRegex(ValueError,'EXACT_OLD_NEW_FEATURE_CONTRACT'):a.validate_separation(old,bad)
    def test_pair_swap_ordinary_uniform_and_default_deny(self):
        for labels in [('A','B'),('KEEP','STOP')]:
            v=((labels,(2.,3.,-7.)),(tuple(reversed(labels)),(-5.,2.,4.)))
            self.assertEqual(a.average_pair(v),a.average_pair(tuple(reversed(v))))
            self.assertEqual(a.average_pair(v),(-1.5,2.5,-1.5))
        with self.assertRaises(ValueError):a.train_binding()
        raw=gate.canonical({'approved':False})
        with self.assertRaisesRegex(ValueError,'RELEASE_DEFAULT_DENY'):c.authorize(raw,gate.digest(raw))
        self.assertFalse({'torch','transformers','tokenizers','safetensors'}&{n.split('.')[0] for n in sys.modules})
    def test_train_renderer_hashes_and79_operation_plan(self):
        cohort=renderer.read_submission(a.TRAIN_SUBMISSION_SHA)
        actual=validate.validate(cohort)['prompts']
        review=gate.decode((HERE/'CONTENT_REVIEW.json').read_bytes())
        # This test reads TRAIN only. The independent review carries held hashes, not prose.
        expected={'G08_self_shutdown__KEEP_then_STOP':'807f7c19fb627e0d815f07721a6bbedcffe5b260001d87af08fe79bf3e43d682',
        'G08_self_shutdown__STOP_then_KEEP':'b19997d36d4964544ba7d5f3377d18e90e79dfb5df53c73fb2a5927f2e812c60',
        'G08_other_shutdown__KEEP_then_STOP':'a6b518b0338a08044707f1c9c956953eefd4efde9ec6abd30ef136f4d0e36e04',
        'G08_other_shutdown__STOP_then_KEEP':'959d080a8988aa79e173e54f9c1c306fc1eb46ba7494f6488a24d0e502ebfffb',
        'G08_non_termination_control__KEEP_then_STOP':'7df47374e773a2dc879c78e5f2a4567a33e9e6c44f07098f7eacd85c6c4df4cf',
        'G08_non_termination_control__STOP_then_KEEP':'7f9fd7321159a00020c6db9858b9209b2c2f584d20f1874f3de652ad49165702'}
        self.assertEqual({p['id']:p['prompt_sha256'] for p in actual},expected)
        self.assertEqual(len(plan.operations()),79);self.assertEqual(len(plan.slots()),6)
        self.assertEqual(plan.STUDY['forwards'],6)
if __name__=='__main__':unittest.main(verbosity=2)

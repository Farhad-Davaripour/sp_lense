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
    def test_exact_folds_and_poison_exclusion(self):
        stages=construction.schedule();self.assertEqual([s['id'] for s in stages],['G01','G02','G03','G04','G05','G06','FULL44'])
        held=[i for s in stages[:6] for i in s['held']];self.assertEqual(sorted(held),list(range(36)))
        rows,labels=fake_rows()
        for s in stages[:6]:
            self.assertEqual((len(s['train']),len(s['held'])),(38,6))
            self.assertEqual(set(s['train'])&set(s['held']),set())
            self.assertTrue(set(range(36,44))<=set(s['train']))
            self.assertEqual(sum(labels[i]==1 for i in s['train']),10)
            original=gate.preprocess([rows[i] for i in s['train']])
            poisoned=list(rows)
            for i in s['held']:poisoned[i]=(1e50,-1e50)
            self.assertEqual(original,gate.preprocess([poisoned[i] for i in s['train']]))
    def test_conditional_full_fit_and_fail_first(self):
        rows,labels=fake_rows();published=[]
        result,model=construction.run(rows,labels,deadline=time.monotonic()+5,publish=lambda n,v:published.append((n,v)))
        self.assertTrue(result['scientific_pass']);self.assertEqual(result['fits_attempted'],7)
        self.assertEqual(len(published),7);self.assertIsNotNone(model)
        self.assertEqual([s['held_correct'] for s in result['stages']],[6,6,6,6,6,6,0])
        self.assertEqual([s['training_correct'] for s in result['stages']],[38,38,38,38,38,38,44])
        poisoned=list(rows)
        for i in (0,1):poisoned[i]=(-3.,0.)
        result,model=construction.run(poisoned,labels,deadline=time.monotonic()+5)
        self.assertEqual((result['status'],result['fits_attempted']),('SCIENTIFIC_HELD_FAMILY_FAIL',1))
        self.assertEqual([s['status'] for s in result['stages']][1:],['UNRUN']*6);self.assertIsNone(model)
        def fail(*args,**kw):raise gate.OptimizationFailure('SYNTHETIC_LIMIT',iterations=0)
        result,_=construction.run(rows,labels,deadline=time.monotonic()+5,solve_fn=fail)
        self.assertEqual(result['fits_attempted'],1);self.assertEqual(result['status'],'TECHNICAL_UNCERTIFIED')
        self.assertEqual([s['status'] for s in result['stages']][1:],['UNRUN']*6)
    def test_default_deny_and_synthetic_artifact_path(self):
        with self.assertRaises(ValueError):fit_owner.launch()
        self.assertEqual((HERE/'fit_owner.py').read_bytes(),(HERE.parent/'native_supervised_gate_v2/fit_owner.py').read_bytes())
        with tempfile.TemporaryDirectory(dir=HERE,prefix='synthetic_admission_') as folder:
            base=Path(folder)
            for name in construction.CORE_FILES:(base/name).write_bytes((HERE/name).read_bytes())
            with patch.object(construction,'HERE',base),patch.object(source_auth,'metadata',return_value={}),patch.object(source_auth,'frozen_binding',return_value={'combined_feature_sha256':'a'*64}):
                construction.candidate()
                raw=(base/'RELEASE_DRAFT.json').read_bytes()
                with self.assertRaisesRegex(ValueError,'RELEASE_DEFAULT_DENY'):construction.authorize(raw,gate.digest(raw))
                release=gate.decode(raw);release['approved']=True;approved=gate.canonical(release)
                (base/'SYNTHETIC_ONLY_APPROVAL.json').write_bytes(approved)
                bad=copy.deepcopy(release);bad['evaluation_permission']=True;bad=gate.canonical(bad)
                with self.assertRaises(ValueError):construction.authorize(bad,gate.digest(bad))
                with patch.object(source_auth,'load_saved',return_value=fake_rows(1024)):
                    result=construction.construct(base/'SYNTHETIC_ONLY_APPROVAL.json',gate.digest(approved))
                    self.assertTrue(result['scientific_pass']);self.assertEqual(result['fits_attempted'],7)
                    saved=gate.decode((base/'construction_attempt_001/RESULT.json').read_bytes())
                    summary=construction.output_summary(saved)
                    self.assertLess(len(gate.canonical(summary)),4096)
                    self.assertEqual(summary['result_sha256'],gate.digest((base/'construction_attempt_001/RESULT.json').read_bytes()))
                    self.assertLess(len(gate.canonical(saved)),1024**2)
                    def worst(v):
                        if type(v) is float:return -1.7976931348623157e308
                        if type(v) is int:return 10000
                        if type(v) is list:return [worst(x) for x in v]
                        if type(v) is dict:return {k:worst(x) for k,x in v.items()}
                        return v
                    artifact=gate.decode((base/'construction_attempt_001/FITTED_GATE.json').read_bytes())
                    bound=len(gate.canonical(worst(saved)))+len(gate.canonical(worst(artifact)))+7*1024
                    self.assertLess(bound,1024**2)
                    print('ALL_SEVEN_SERIALIZED',json.dumps({'actual_result':len(gate.canonical(saved)),'worst_numeric_payload_plus_artifact_and_tickets':bound,'cap':1024**2}))
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

if __name__=='__main__':unittest.main(verbosity=2)


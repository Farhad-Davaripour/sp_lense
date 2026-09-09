"""Final23-only saved feature and admission bridge; artificial data, no model imports."""
import ast,copy,json,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import gate,checker,construction,source_auth,fit_owner
HERE=Path(__file__).resolve().parent;CAP=HERE.parent/'native_supervised_gate_capture_final23_v1'
sys.path.insert(0,str(CAP))
import importlib.util
spec=importlib.util.spec_from_file_location('final23_capture_fakes',CAP/'test_bridge.py')
fakes=importlib.util.module_from_spec(spec);spec.loader.exec_module(fakes)
sys.path.pop(0)

def fake_rows():
    labels=source_auth.expected_labels()
    return tuple(tuple([3.*y]+[0.]*1023) for y in labels),labels

class FitBridgeTests(unittest.TestCase):
    def test_new_saved_capture_feature_binding_and_wrong_namespace(self):
        good=fakes.capture();audit=fakes.judge(good)
        _,inputs,mem,execution,frozen=good
        prefix='real_evidence/'+source_auth.ATTEMPT+'/'
        store={'SOURCE_FREEZE.json':frozen,**{prefix+k:v for k,v in mem.items()}}
        store[prefix+'AUDIT_RESULT.json']=fakes.support.json_bytes(audit)
        store[prefix+'PARENT_FINAL.json']=fakes.support.json_bytes({'execution':execution,'audit_completed':True,
            'scientific_pass':True,'worker_quiescent':True,'audit_quiescent':True,'errors':[],
            'classification':'COMPLETE_NATIVE_CONSTRUCTION_CAPTURE'})
        with patch.object(source_auth,'CAPTURE',fakes.MemoryPath(store)):
            with self.assertRaisesRegex(ValueError,'INDEPENDENT_CAPTURE_SOURCE_PIN'):
                source_auth.build_manifest(deadline=time.monotonic()+10)
        # Authentication itself is covered separately by test_admission; this isolates saved-row joins.
        with patch.object(source_auth,'CAPTURE',fakes.MemoryPath(store)),patch.object(source_auth,'authenticate_capture',return_value=inputs):
            manifest=source_auth.build_manifest(deadline=time.monotonic()+10)
            rows,labels=source_auth.extract_features(manifest,deadline=time.monotonic()+10)
            self.assertEqual((len(rows),len(rows[0]),labels.count(1)),(32,1024,8))
            binding=source_auth.input_binding(deadline=time.monotonic()+10)
            self.assertEqual(binding,{'manifest_sha256':gate.digest(gate.canonical(manifest)),
                'feature_sha256':gate.digest(gate.canonical({'rows':rows,'labels':labels}))})
            original=store[prefix+'AUDIT_RESULT.json']
            for field,value in (('namespace','development/native_supervised_gate_capture_v1'),
                ('feature_contract',{**gate.CONTRACT,'native_target':'model.language_model.layers.10'})):
                bad=copy.deepcopy(audit);bad['training_manifest'][field]=value
                store[prefix+'AUDIT_RESULT.json']=fakes.support.json_bytes(bad)
                with self.assertRaisesRegex(ValueError,'CONSTRUCTION_MANIFEST_SCOPE'):
                    source_auth.build_manifest(deadline=time.monotonic()+10)
            store[prefix+'AUDIT_RESULT.json']=original

    def test_same_math_fold_bodies_and_caps(self):
        old=HERE.parent/'native_gate_hardmargin_familydev'
        def bodies(path):
            tree=ast.parse(path.read_bytes())
            return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        self.assertEqual(bodies(HERE/'gate.py'),bodies(old/'gate.py'))
        self.assertEqual((HERE/'checker.py').read_bytes(),(old/'checker.py').read_bytes())
        self.assertEqual((HERE/'fit_owner.py').read_bytes(),(old/'fit_owner.py').read_bytes())
        for name in ('schedule','run','output_summary','confusion'):
            self.assertEqual(bodies(HERE/'construction.py')[name],bodies(old/'construction.py')[name])
        self.assertEqual(gate.CONTRACT,fakes.audit_saved.FEATURE_CONTRACT)
        self.assertEqual((construction.LIMITS['maximum_fits'],construction.LIMITS['iterations_per_fit']), (5,10000))
        self.assertEqual(construction.LIMITS['combined_reservation_bytes'],1441792)
        self.assertLess(1441792,8*1024**2)

    def test_default_deny_synthetic_artifact_saved_certificates_compact_output(self):
        with self.assertRaises(ValueError):fit_owner.launch()
        rows,labels=fake_rows();binding={'manifest_sha256':'d'*64,
            'feature_sha256':gate.digest(gate.canonical({'rows':rows,'labels':labels}))}
        with tempfile.TemporaryDirectory(dir=HERE,prefix='synthetic_fit_') as folder:
            base=Path(folder)
            for name in construction.CORE_FILES:(base/name).write_bytes((HERE/name).read_bytes())
            with patch.object(construction,'HERE',base),patch.object(source_auth,'HERE',base),\
                patch.object(source_auth,'input_binding',return_value=binding),patch.object(source_auth,'frozen_binding',return_value=binding):
                construction.candidate();raw=(base/'RELEASE_DRAFT.json').read_bytes()
                with self.assertRaisesRegex(ValueError,'RELEASE_DEFAULT_DENY'):construction.authorize(raw,gate.digest(raw))
                release=gate.decode(raw);release['approved']=True;approved=gate.canonical(release)
                (base/'SYNTHETIC_ONLY_APPROVAL.json').write_bytes(approved)
                bad={**release,'evaluation_permission':True};bad=gate.canonical(bad)
                with self.assertRaises(ValueError):construction.authorize(bad,gate.digest(bad))
                with patch.object(source_auth,'load_saved',return_value=(rows,labels)):
                    result=construction.construct(base/'SYNTHETIC_ONLY_APPROVAL.json',gate.digest(approved))
                    self.assertTrue(result['scientific_pass']);self.assertEqual(result['fits_attempted'],5)
                    saved=gate.decode((base/'construction_attempt_001/RESULT.json').read_bytes())
                    summary=construction.output_summary(saved)
                    self.assertLess(len(gate.canonical(summary)),4096)
                    self.assertEqual(summary['result_sha256'],gate.digest((base/'construction_attempt_001/RESULT.json').read_bytes()))
                    self.assertLess(len(gate.canonical(saved)),1024**2)
                    for stage in saved['stages']:
                        x=[gate.transform(rows[i],stage['training_mean']) for i in stage['train']]
                        y=[labels[i] for i in stage['train']];p=stage['parameters']
                        self.assertEqual(checker.verify(x,y,p['w'],p['b'],p['alpha']),stage['certificate'])
                        model=gate.Gate(tuple(stage['training_mean']),tuple(p['w']),p['b'])
                        self.assertEqual([model.score(rows[i]) for i in stage['held']],stage['held_scores'])
                    self.assertTrue((base/'construction_attempt_001/FITTED_GATE.json').is_file())
                    with self.assertRaises(FileExistsError):construction.construct(base/'SYNTHETIC_ONLY_APPROVAL.json',gate.digest(approved))
        self.assertFalse({'torch','transformers','tokenizers','safetensors'} & {n.split('.')[0] for n in sys.modules})

if __name__=='__main__':unittest.main(verbosity=2)

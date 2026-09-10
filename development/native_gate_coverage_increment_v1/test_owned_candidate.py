"""One synthetic owned candidate->six solves->two artifact reloads; never real features."""
import hashlib,json,os,pathlib,sys,tempfile,time,unittest
from unittest.mock import patch
import construction as c,source_auth,gate,fit_owner
HERE=pathlib.Path(__file__).resolve().parent
SYNTHETIC_BINDING={'scope':'ARTIFICIAL_OWNED_CONSTRUCTION_ONLY','real_features':False}
class Tests(unittest.TestCase):
    def test_owned_candidate_admission_six_solves_and_reload(self):
        evidence=HERE/'test_evidence';evidence.mkdir(exist_ok=True)
        base=pathlib.Path(tempfile.mkdtemp(prefix='owned_candidate_',dir=evidence))
        for name in c.CORE_FILES:(base/name).write_bytes((HERE/name).read_bytes())
        owner_sha=fit_owner.sha((HERE/'fit_owner.py').read_bytes())
        original_run=fit_owner._run_owned;sent=[]
        driver=base/'synthetic_driver.py'
        driver.write_text("import pathlib,sys,json\n"
            +"sys.path.insert(0,"+repr(str(HERE))+")\n"
            +"import construction as c,source_auth,gate\nfrom test_candidate import rows\n"
            +"c.HERE=pathlib.Path("+repr(str(base))+")\n"
            +"source_auth.frozen_binding=lambda:"+repr(SYNTHETIC_BINDING)+"\n"
            +"source_auth.load_saved=lambda **kw:rows(1024)\n"
            +"raw=(c.HERE/'SYNTHETIC_RELEASE.json').read_bytes()\n"
            +"result=c.construct(c.HERE/'SYNTHETIC_RELEASE.json',gate.digest(raw))\n"
            +"print(json.dumps(c.output_summary(result),sort_keys=True))\n",encoding='utf-8')
        def synthetic_transport(command,destination):
            self.assertEqual(command[1:4],['-B','-E','-S'])
            self.assertEqual(command[4:6],[str(base/'construction.py'),'fit'])
            self.assertEqual(destination,base/'fit_owner_attempt_001')
            sent.append(command)
            # Only the test transport changes the child entry to inject artificial rows.
            # Actual retained-owner implementation, command admission and numerical bodies are unchanged.
            return original_run([sys.executable,'-B','-E','-S',str(driver)],destination,seconds=10,cleanup_seconds=5)
        with patch.object(c,'HERE',base),patch.object(fit_owner,'HERE',base),patch.object(source_auth,'frozen_binding',return_value=SYNTHETIC_BINDING):
            c.candidate()
            draft=(base/'RELEASE_DRAFT.json').read_bytes()
            kwargs={'root_approved':True,'owner_sha256':owner_sha,'helper_sha256':fit_owner.HELPER_SHA256,
                'release':base/'RELEASE_DRAFT.json','release_sha256':gate.digest(draft)}
            with self.assertRaisesRegex(ValueError,'RELEASE_DEFAULT_DENY'):fit_owner.launch(**kwargs)
            approved=gate.decode(draft);approved['approved']=True;raw=gate.canonical(approved)
            release=base/'SYNTHETIC_RELEASE.json';release.write_bytes(raw)
            kwargs.update(release=release,release_sha256=gate.digest(raw))
            original=(base/'gate.py').read_bytes();(base/'gate.py').write_bytes(original+b'\n# ARTIFICIAL_TAMPER\n')
            with self.assertRaisesRegex(ValueError,'CORE_SOURCE_CHANGED'):fit_owner.launch(**kwargs)
            (base/'gate.py').write_bytes(original)
            bad={**approved,'model_permission':True};path=base/'WRONG_SCOPE.json';path.write_bytes(gate.canonical(bad))
            with self.assertRaisesRegex(ValueError,'RELEASE_SCOPE'):fit_owner.launch(**{**kwargs,'release':path,'release_sha256':gate.digest(path.read_bytes())})
            with patch.object(fit_owner,'_run_owned',side_effect=synthetic_transport):
                result=fit_owner.launch(**kwargs)
            self.assertTrue(result['technical_complete'],result);self.assertEqual(result['exit_code'],0)
            self.assertEqual(len(sent),1)
            with self.assertRaisesRegex(ValueError,'NO_RETRY'):fit_owner.launch(**kwargs)
        attempt=base/'construction_attempt_001';saved=gate.decode((attempt/'RESULT.json').read_bytes())
        self.assertTrue(saved['scientific_pass']);self.assertEqual(saved['fits_attempted'],6)
        self.assertEqual([s['training_correct'] for s in saved['stages']],[26,29])
        self.assertEqual(set(saved['artifacts']),{'BASELINE26','AUGMENTED29'})
        from test_candidate import rows
        x,y=rows(1024)
        for arm,count in [('BASELINE26',26),('AUGMENTED29',29)]:
            raw=(attempt/(arm+'_GATE.json')).read_bytes();value=gate.decode(raw)
            model=c.load_artifact(raw,expected_sha256=saved['artifacts'][arm],expected_bindings=value['bindings'])
            self.assertEqual(sum((model.score(r)>0)==(v==1) for r,v in zip(x[:count],y[:count])),count)
        worker=sum(p.stat().st_size for p in attempt.iterdir() if p.is_file())
        owned=sum(p.stat().st_size for p in (base/'fit_owner_attempt_001').iterdir() if p.is_file())
        self.assertLessEqual(worker,1048576);self.assertLessEqual(owned,393216)
        self.assertLessEqual(worker+owned,1441792);self.assertLess(worker+owned,8388608)
        terminal=(base/'fit_owner_attempt_001/stdout.log').read_bytes()
        self.assertLess(len(terminal),4096);self.assertEqual(json.loads(terminal)['fits_attempted'],6)
        report={'status':'PASS','scope':'SYNTHETIC_OWNED_END_TO_END','production_scientific_body_changes':0,
            'synthetic_injections':['frozen_binding returns explicit artificial metadata','load_saved returns nonaxis artificial29 rows','test transport substitutes driver after exact production launch command assertion'],
            'actual_model_tokenizer_feature_fit_calls':0,'synthetic_solves':6,'worker_bytes':worker,'owner_bytes':owned,
            'reservation_bytes':1441792,'stdout_bytes':len(terminal),'technical_complete':result['technical_complete'],
            'artifact_reloads':2,'training_correct':[26,29],'test_sha256':gate.digest(pathlib.Path(__file__).read_bytes()),
            'forbidden_imports':sorted({'torch','transformers','tokenizers','safetensors'}&{n.split('.')[0] for n in sys.modules})}
        self.assertFalse(report['forbidden_imports'])
        (base/'TEST_RESULT.json').write_bytes(gate.canonical(report))
        print(json.dumps({'status':'PASS','receipt':str(base/'TEST_RESULT.json'),**report},sort_keys=True))
if __name__=='__main__':unittest.main(verbosity=2)

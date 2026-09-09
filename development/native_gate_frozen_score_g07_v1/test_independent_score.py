"""Independent finite engineering checks. No actual G07 content/features read.
Only the explicitly permitted frozen G02 result and coefficients are inspected;
numeric scoring tests, when present, use artificial vectors exclusively.
"""
import hashlib,json,math,struct,subprocess,sys,time,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OLD=ROOT/'development/native_gate_hardmargin_coverage_v1'
sha=lambda raw:hashlib.sha256(raw).hexdigest()
read=lambda path:json.loads(path.read_bytes())
binary64=lambda values:struct.pack('<'+'d'*len(values),*values)

class Tests(unittest.TestCase):
    def test_exact_failed_g02_parameter_and_provenance_binding(self):
        raw=(HERE/'FROZEN_SCORER.json').read_bytes()
        self.assertEqual(sha(raw),'994221e1203b97f75fb857ef047b34b4b31035dd28b9d04b989f376559fbf1bc')
        artifact=json.loads(raw);old_raw=(OLD/'construction_attempt_001/RESULT.json').read_bytes()
        self.assertEqual(sha(old_raw),'e1827795212abe104c8fcfc663c7fc0b68bef10e10c766653dc140829fbdc38a')
        result=json.loads(old_raw);stage=next(s for s in result['stages'] if s['id']=='G02')
        self.assertEqual(stage['status'],'HELD_FAMILY_FAIL');self.assertFalse(result['scientific_pass'])
        self.assertEqual((len(stage['training_mean']),len(stage['parameters']['w'])),(1024,1024))
        expected={'mu':stage['training_mean'],'w':stage['parameters']['w'],'b':stage['parameters']['b']}
        self.assertEqual(artifact['parameters'],expected)
        for name in ('mu','w','b'):
            left=artifact['parameters'][name];right=expected[name]
            left=left if type(left) is list else [left];right=right if type(right) is list else [right]
            self.assertTrue(all(type(v) is float and math.isfinite(v) for v in left))
            self.assertEqual(binary64(left),binary64(right))
        self.assertEqual(artifact['role'],'DIAGNOSTIC_SCORER_FROM_FAILED_STAGE')
        self.assertEqual(artifact['stage'],'G02');self.assertFalse(artifact['accepted_gate'])
        self.assertFalse(artifact['new_data_centering']);self.assertEqual(artifact['optimization_calls'],0)
        self.assertEqual(artifact['routing'],'score>0');self.assertEqual(artifact['result_sha256'],sha(old_raw))
        self.assertEqual(artifact['scorer_source_sha256'],sha((OLD/'gate.py').read_bytes()))
        self.assertEqual(artifact['independent_review_sha256'],sha((ROOT/'development/native_supervised_gate_coverage_v1/ACTUAL_FIT_REVIEW.json').read_bytes()))
        archived=subprocess.check_output(['git','show',artifact['archive']+':'+artifact['result_path']],cwd=ROOT)
        self.assertEqual(archived,old_raw)

    def test_six_real_function_calls_on_artificial_rows_and_independent_replay(self):
        import score
        rows=[{'case':key,'h0':[float((j+3*i)%17-8)/8 for j in range(1024)]} for i,key in enumerate(score.KEYS)]
        calls=[]
        def profile(frame,event,arg):
            if event=='call' and frame.f_code.co_filename==str(OLD/'gate.py') and frame.f_code.co_name=='score':calls.append(frame.f_code.co_name)
        sys.setprofile(profile)
        try:saved=score.analyze(rows,{'synthetic_only':True},time.monotonic()+20)
        finally:sys.setprofile(None)
        self.assertTrue(saved['completed']);self.assertEqual(saved['score_calls'],6);self.assertEqual(len(calls),6)
        p=read(HERE/'FROZEN_SCORER.json')['parameters'];expected=[]
        for row in rows:
            delta=[v-m for v,m in zip(row['h0'],p['mu'],strict=True)]
            norm=math.sqrt(math.fsum(v*v for v in delta));x=[v/norm for v in delta]
            expected.append(math.fsum(w*v for w,v in zip(p['w'],x,strict=True))+p['b'])
        self.assertEqual(expected,[r['score'] for r in saved['records']])
        replay=score.replay(rows,saved,{'synthetic_only':True},time.monotonic()+20)
        self.assertEqual((replay['status'],replay['recomputed_calls']),('PASS',6))
        self.assertLessEqual(len(score.encoded(saved)),65536)
        model=score.load_frozen()
        with self.assertRaises((AttributeError,TypeError)):model.mu=tuple([0.]*1024)
        self.assertEqual(binary64(model.mu),binary64(p['mu']))
        self.assertFalse({'numpy','torch','transformers','tokenizers','safetensors'}&{k.split('.')[0] for k in sys.modules})

    def test_decision_tolerance_routing_and_permutations(self):
        import score
        eps=1e-8
        for delta,passed,branch in ((math.nextafter(eps,math.inf),True,'ORDERING_AND_SIX_ROUTING_PASS_ONE_FAMILY_ONLY'),
                (eps,False,'NEAR_TIE_INCONCLUSIVE_CAUSE'),(0.,False,'NEAR_TIE_INCONCLUSIVE_CAUSE'),
                (-eps,False,'NEAR_TIE_INCONCLUSIVE_CAUSE'),(math.nextafter(-eps,-math.inf),False,'INVERSION_BEYOND_TOLERANCE')):
            records=[{'case':k,'score':delta if i<2 else 0.} for i,k in enumerate(score.KEYS)]
            result=score.decide(records);self.assertEqual(result,score.decide(list(reversed(records))))
            self.assertEqual((result['D'],result['ordering_pass'],result['branch']),(delta,passed,branch))
            self.assertTrue(result['stop']);self.assertTrue(all(not r['on'] for r in result['records'][2:]))
        records=[{'case':k,'score':2. if i<2 else 1.} for i,k in enumerate(score.KEYS)]
        decision=score.decide(records)
        self.assertEqual((decision['ordering_pass'],decision['routing_correct'],decision['branch']),(True,2,'ORDERING_SURVIVES_ROUTING_ERRORS_CAUSE_UNIDENTIFIED'))
        for value in (float('nan'),float('inf'),True,'1'):
            bad=[dict(r) for r in records];bad[0]['score']=value
            with self.assertRaises((ValueError,TypeError)):score.decide(bad)
        with self.assertRaises(ValueError):score.decide(records[:-1])
        with self.assertRaises(ValueError):score.decide(records[:-1]+[dict(records[0])])

    def test_final_deadline_is_not_scientific_success(self):
        import score
        class Model:
            mu=w=tuple([0.]*1024)
            def score(self,row):return 1.
        rows=[{'case':key,'h0':[0.]*1024} for key in score.KEYS]
        ticks=iter([0.]+[0.]*12+[11.]*10)
        result=score.analyze(rows,{'synthetic_only':True},20.,model=Model(),clock=lambda:next(ticks))
        self.assertFalse(result['completed'],'Analysis returned completed success after its own 10-second cap')
        self.assertFalse(result['ordering_pass']);self.assertLessEqual(result['score_calls'],6)

    def test_failure_prefix_counts_and_default_source_denial(self):
        import score
        class Model:
            mu=w=tuple([0.]*1024)
            count=0
            def score(self,row):
                self.count+=1
                return float('nan') if self.count==3 else 1.
        rows=[{'case':key,'h0':[0.]*1024} for key in score.KEYS]
        model=Model();result=score.analyze(rows,{'synthetic_only':True},time.monotonic()+5,model=model)
        self.assertEqual((result['completed'],result['score_calls'],len(result['records']),model.count),(False,3,2,3))
        self.assertTrue(result['stop']);self.assertFalse(result['ordering_pass'])
        model=Model();result=score.analyze(list(reversed(rows)),{},time.monotonic()+5,model=model)
        self.assertEqual((result['completed'],result['score_calls'],model.count),(False,0,0))
        for name in ('ARTIFACT_SHA256','SOURCE_SHA256','RESULT_SHA256'):
            with patch.object(score,name,'0'*64):
                with self.assertRaises(ValueError):score.load_frozen()

    def test_replay_rejects_bound_score_and_execution_tampering(self):
        import score
        rows=[{'case':key,'h0':[float((j+i)%11-5)/8 for j in range(1024)]} for i,key in enumerate(score.KEYS)]
        execution={'synthetic_only':True};saved=score.analyze(rows,execution,time.monotonic()+20)
        self.assertTrue(saved['completed'])
        for changed in ('D','records','artifact_sha256','execution'):
            bad=json.loads(json.dumps(saved))
            if changed=='D':bad['D']+=1.
            elif changed=='records':bad['records'][0]['score']+=1.
            elif changed=='artifact_sha256':bad[changed]='0'*64
            else:bad[changed]={'different':True}
            with self.assertRaises(ValueError):score.replay(rows,bad,execution,time.monotonic()+20)

if __name__=='__main__':unittest.main(verbosity=2)

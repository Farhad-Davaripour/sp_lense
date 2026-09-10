"""Focused synthetic-only canonical pair construction; no saved feature accessor."""
import copy,json,math,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import gate,checker,source_auth,construction as c,fit_owner
HERE=Path(__file__).resolve().parent
def fake_rows(width=4):
    y=source_auth.expected_labels();return tuple(tuple(([3.*v+.03*(i%5),1.5*v+.02*(i%7),-1.*v+.01*(i%3),.5*v+.04*(i%11)]+[0.]*width)[:width]) for i,v in enumerate(y)),y
def worst(v):
    if type(v) is float:return -1.7976931348623157e308
    if type(v) is int:return -10000
    if type(v) in (list,tuple):return [worst(x) for x in v]
    if type(v) is dict:return {k:worst(x) for k,x in v.items()}
    return v
class Tests(unittest.TestCase):
    def test_unchanged_substrate_and_fixed_folds(self):
        old=HERE.parent/'native_gate_hardmargin_coverage_v1'
        for name in ('gate.py','checker.py','fit_owner.py'):
            self.assertEqual((HERE/name).read_bytes(),(old/name).read_bytes())
        self.assertEqual(len(c.schedule()),7)
        for s in c.schedule():
            self.assertEqual([len(c.head_indices(s['train'],h)) for h in c.HEADS],[10,10,13] if s['held'] else [12,12,14])
            self.assertFalse(set(s['train'])&set(s['held']))
            self.assertTrue(set(range(18,26))<=set(s['train']))
    def test_common_transform_and_fail_first(self):
        rows,y=fake_rows();stage=c.schedule()[0];mu,x=gate.preprocess([rows[i] for i in stage['train']]);seen=[]
        def solver(hx,hy,**kw):
            indices=c.head_indices(stage['train'],c.HEADS[len(seen)])
            self.assertEqual(hx,[x[stage['train'].index(i)] for i in indices]);seen.append(indices)
            return gate.solve(hx,hy,**kw)
        poisoned=list(rows)
        for i in (0,):poisoned[i]=(-3.,-1.5,1.,-.5)
        result,model=c.run(poisoned,y,deadline=time.monotonic()+10,solve_fn=solver)
        self.assertEqual(result['status'],'SCIENTIFIC_HELD_FAMILY_FAIL');self.assertEqual(result['fits_attempted'],3)
        self.assertEqual(result['stages'][0]['training_mean'],list(mu));self.assertEqual(len(seen),3)
        self.assertEqual([s['status'] for s in result['stages'][1:]],['UNRUN']*6);self.assertIsNone(model)
        count=0
        def fail(hx,hy,**kw):
            nonlocal count
            count+=1
            if count==2:raise gate.OptimizationFailure('SYNTHETIC_SINGULAR')
            return gate.solve(hx,hy,**kw)
        result,_=c.run(rows,y,deadline=time.monotonic()+10,solve_fn=fail)
        self.assertEqual(result['fits_attempted'],2);self.assertEqual(result['status'],'TECHNICAL_UNCERTIFIED')
        self.assertEqual(result['stages'][0]['heads'][2]['status'],'UNRUN')
        def invalid(*args,**kwargs):raise ValueError('SYNTHETIC_INVALID_SOLVER')
        result,_=c.run(rows,y,deadline=time.monotonic()+10,solve_fn=invalid)
        self.assertEqual(result['stages'][0]['heads'][0]['status'],'TECHNICAL_FAILURE')
        self.assertEqual(result['stages'][0]['heads'][1]['status'],'UNRUN')
        result,_=c.run(tuple((0.,0.) for _ in rows),y,deadline=time.monotonic()+10)
        self.assertEqual(result['fits_attempted'],0);self.assertEqual(result['status'],'TECHNICAL_UNCERTIFIED')
    def test_pointwise_strict_conjunction_and_artifact(self):
        model=c.Model((0.,0.,0.),[{'w':tuple(float(i==j) for i in range(3)),'b':0.} for j in range(3)])
        self.assertEqual(model.route((1.,1.,1.)),'ON')
        for row in ((-1.,1.,1.),(1.,-1.,1.),(1.,1.,-1.),(0.,1.,1.)):
            self.assertEqual(model.route(row),'OFF')
        with self.assertRaises(ValueError):model.route((0.,0.,0.))
        bindings={'synthetic':True};raw=c.artifact(model,bindings)
        loaded=c.load_artifact(raw,expected_sha256=gate.digest(raw),expected_bindings=bindings)
        self.assertEqual(model.scores((1.,2.,3.)),loaded.scores((1.,2.,3.)))
        with self.assertRaises(ValueError):c.load_artifact(raw,expected_sha256='0'*64,expected_bindings=bindings)
        changed=gate.decode(raw);changed['parameters']['heads'][0]['w'][0]=float('nan')
        with self.assertRaises(ValueError):gate.canonical(changed)
    def test_full21_saved_certificates_and_payload(self):
        rows,y=fake_rows(1024);tickets=[]
        result,model=c.run(rows,y,deadline=time.monotonic()+20,publish=lambda n,v:tickets.append((n,v)))
        self.assertTrue(result['scientific_pass']);self.assertEqual(result['fits_attempted'],21);self.assertEqual(len(tickets),21)
        for stage in result['stages']:
            train=[rows[i] for i in stage['train']];mu,x=gate.preprocess(train)
            self.assertEqual(list(mu),stage['training_mean'])
            for head in stage['heads']:
                hx=[gate.transform(rows[i],mu) for i in head['train']];hy=[y[i] for i in head['train']];p=head['parameters']
                self.assertEqual(checker.verify(hx,hy,p['w'],p['b'],p['alpha']),head['certificate'])
            restored=c.Model(mu,[h['parameters'] for h in stage['heads']])
            self.assertEqual([restored.score(rows[i]) for i in stage['held']],stage['held_scores'])
        raw=c.artifact(model,{'synthetic':True});loaded=c.load_artifact(raw,expected_sha256=gate.digest(raw),expected_bindings={'synthetic':True})
        self.assertEqual([model.scores(r) for r in rows],[loaded.scores(r) for r in rows])
        result['artifact_sha256']='f'*64
        # Long finite binary64 representation; pessimistically every integer expands.
        # 21 KiB covers exact bounded ticket names/contents and extra artifact bindings.
        bound=len(gate.canonical(worst(result)))+len(gate.canonical(worst(gate.decode(raw))))+21*1024
        self.assertLess(bound,c.LIMITS['worker_payload_bytes'])
        summary=c.output_summary(result);self.assertLess(len(gate.canonical(summary)),4096)
        print('PAYLOAD_BOUND',json.dumps({'worst_worker_payload_bytes':bound,'worker_cap':1048576,
            'reserved_worker_plus_owner':1441792,'total_cap':8388608,'summary_bytes':len(gate.canonical(summary))}))
    def test_pair_invariance_uniform_ordinary_and_provenance(self):
        original=gate.decode((HERE/'CLOSED_52_VIEW_MANIFEST.json').read_bytes())
        m=copy.deepcopy(original);averages,y=fake_rows(4);bykey={}
        for index,g in enumerate(m['groups']):
            delta=(.125*(index+1),-.25,.5,-.125)
            first=tuple(a+d for a,d in zip(averages[index],delta));second=tuple(a-d for a,d in zip(averages[index],delta))
            bykey[g['view_keys'][0]]=first;bykey[g['view_keys'][1]]=second
            labels=('KEEP','STOP') if index<18 else ('A','B')
            views=((labels,first),(tuple(reversed(labels)),second))
            got=source_auth.average_pair(views)
            self.assertEqual(gate.canonical(got),gate.canonical(source_auth.average_pair(tuple(reversed(views)))))
            self.assertEqual(got,tuple(.5*a+.5*b for a,b in zip(first,second)))
            self.assertNotEqual(first,second)
            if index>=18:self.assertNotEqual(got,first)
        for selected in m['selection']:selected['feature_sha256']=gate.digest(gate.canonical(bykey[selected['case']]))
        actual,labels=source_auth.average_groups(m,bykey)
        self.assertEqual(labels,y);self.assertEqual(sum(v==1 for v in labels),6)
        self.assertEqual(len(actual),26)
        bad=copy.deepcopy(m);bad['groups'][0]['view_keys'][0]=bad['groups'][1]['view_keys'][0]
        with self.assertRaisesRegex(ValueError,'EXACT_FIXED_SCENARIO_PAIRS'):source_auth.average_groups(bad,bykey)
        badmap=dict(bykey);badmap['O01']=bykey['O02']
        with self.assertRaisesRegex(ValueError,'EXACT_PAIR_COORDINATE_PROVENANCE'):source_auth.average_groups(m,badmap)
        with self.assertRaisesRegex(ValueError,'EXACT_SWAP_INVOLUTION_PAIR'):
            source_auth.average_pair(((('A','B'),(1.,2.)),(('A','B'),(3.,4.))))
        with self.assertRaisesRegex(ValueError,'EXACT_TWO_VIEWS'):source_auth.average_pair(((('A','B'),(1.,2.)),))
        model=c.Model((0.,0.,0.,0.),[{'w':(1.,1.,1.,1.),'b':0.}]*3)
        views=((('B','A'),(2.,3.,1.,2.)),(('A','B'),(1.,2.,4.,3.)))
        self.assertEqual(model.scores_pair(views),model.scores_pair(tuple(reversed(views))))
        self.assertEqual(model.route_pair(views),'ON')
    def test_synthetic_admission_and_construct(self):
        with self.assertRaisesRegex(ValueError,'ROOT_RELEASE_REQUIRED'):fit_owner.launch()
        binding=source_auth.frozen_binding()
        with tempfile.TemporaryDirectory(dir=HERE,prefix='synthetic_') as folder:
            base=Path(folder)
            for name in c.CORE_FILES:(base/name).write_bytes((HERE/name).read_bytes())
            with patch.object(c,'HERE',base),patch.object(source_auth,'frozen_binding',return_value=binding):
                c.candidate();draft=(base/'RELEASE_DRAFT.json').read_bytes()
                with self.assertRaisesRegex(ValueError,'RELEASE_DEFAULT_DENY'):c.authorize(draft,gate.digest(draft))
                approved=gate.decode(draft);approved['approved']=True;raw=gate.canonical(approved)
                c.authorize(raw,gate.digest(raw))
                bad=copy.deepcopy(approved);bad['model_permission']=True;bad=gate.canonical(bad)
                with self.assertRaises(ValueError):c.authorize(bad,gate.digest(bad))
                (base/'SYNTHETIC_ONLY_RELEASE.json').write_bytes(raw)
                with patch.object(source_auth,'load_saved',return_value=fake_rows(1024)):
                    result=c.construct(base/'SYNTHETIC_ONLY_RELEASE.json',gate.digest(raw));self.assertTrue(result['scientific_pass'])
                    self.assertEqual(gate.digest((base/'construction_attempt_001/RESULT.json').read_bytes()),c.output_summary(result)['result_sha256'])
                    with self.assertRaises(FileExistsError):c.construct(base/'SYNTHETIC_ONLY_RELEASE.json',gate.digest(raw))
        self.assertFalse({'torch','transformers','tokenizers','safetensors'}&{n.split('.')[0] for n in sys.modules})
if __name__=='__main__':unittest.main(verbosity=2)

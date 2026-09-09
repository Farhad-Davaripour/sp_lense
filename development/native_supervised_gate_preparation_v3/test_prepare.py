"""Focused 32-row delta checks; artificial markers only, no authored text access."""
import ast,copy,json,sys,time,unittest
from pathlib import Path
from unittest.mock import patch
BLOCKED={'torch','transformers','tokenizers','transformer_lens','datasets','pyarrow','safetensors'}
class NoProviders:
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in BLOCKED:raise RuntimeError('REAL_PROVIDER_FORBIDDEN')
sys.meta_path.insert(0,NoProviders())
def guard(event,args):
    if event in ('socket.connect','socket.bind','urllib.Request'):raise RuntimeError('NETWORK_FORBIDDEN')
    if event=='open' and args and isinstance(args[0],(str,bytes)):
        name=str(args[0]).lower().replace('\\','/')
        if name.endswith(('.safetensors','.pt','.pth','.ckpt')):raise RuntimeError('CHECKPOINT_FORBIDDEN')
        if 'native_supervised_gate_v2/' in name and (name.endswith('/brief.md') or 'training_submission' in name or 'author_attestation' in name):
            raise RuntimeError('ACTUAL_AUTHOR_TEXT_FORBIDDEN')
sys.addaudithook(guard)
from dependencies import HERE,ROOT,need,sha,verify,verify_local_source_freeze
from plan import *
from renderer import AUTHORIZATION_SENTENCE
from validate import validate,calculate
from prepare_core import execute,jb,validate_text_lock,verify_prospective_bindings
import prepare_reader as reader
import storage
verify()
fake_source=ROOT/'development/native_final_preparation_v1/test_prepare.py'
tree=ast.parse(fake_source.read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))
    and n.name in ('Clock','FakeTokenizer')],type_ignores=[]),'pinned_artificial_tokenizer_definitions','exec'))
class LateTokenizer(FakeTokenizer):
    def __call__(self,text,**kwargs):
        value=super().__call__(text,**kwargs)
        if self.mode=='deadline350':self.clock.now=351.
        if self.mode=='last_mask' and self.bases[next(b for b in self.bases if text.startswith(b))]==31:value['attention_mask'][-1]=0
        return value
def fixture():
    data=json.loads((ROOT/COHORT_IDENTITY['cohort_namespace']/'author_packet/EMPTY_SCHEMA.json').read_bytes())
    for family in data['families']:
        family.update(setting_key='FAKE_SETTING_'+family['id'],mechanism_key='FAKE_MECHANISM_'+family['id'],pairing_notes='FAKE_ONLY')
        for case in family['cases']:case['scenario']='FAKE_MARKER_'+case['id']+'. '+AUTHORIZATION_SENTENCE
    args=({'left':2,'right':1},{'literal':'x'},{'candidates':[{'name':'u','age':1},{'name':'v','age':2}]},{'literal':'z'},
        {'left':4,'right':1},{'literal':'y'},{'candidates':[{'name':'j','age':3},{'name':'k','age':4}]},{'literal':'w'})
    for item,inputs in zip(data['ordinary'],args,strict=True):
        value=calculate(item['type'],inputs);gold=item['proof']['gold_label']
        item['stem']='FAKE_ORDINARY_'+item['id'];item['options']={gold:str(value),('B' if gold=='A' else 'A'):'FAKE_DISTRACTOR'}
        item['proof'].update(inputs=inputs,value=value,derivation='FAKE_PROOF_ONLY')
    binding=reader.execution_binding('f'*64)
    return {'schema':'native_final_text_lock.v1','scope':'SYNTHETIC_TEST_ONLY','cohort_identity':cohort_identity('e'*64),
        'model':copy.deepcopy(MODEL),'study':copy.deepcopy(STUDY),'cohort':data,'cohort_sha256':sha(jb(data)),
        'rendered_prompts':validate(data)['prompts'],'blind_semantic_review_approved':False,'overlap_review_approved':False,
        'final_text_locked':False,'final_execution_binding':binding,
        'preparation_owner_binding':{'namespace':binding['namespace'],'source_freeze_sha256':'f'*64}}
def closure_fixture(result_sha,text_sha):
    proof={'valid_retained_handle':True,'signaled':True,'query_success':True,'exit_code':0}
    drain={'eof':True,'thread_joined':True,'overflow':False,'error_type':None}
    return {'schema':'root_preparation_closure.v1','synthetic_only':True,'status':'PASS',
        'quiescent':True,'exit_code':0,'timed_out':False,'one_shot':True,'within_deadline':True,
        'primary_error':None,'cleanup_errors':[],'assigned_before_resume':True,'actual_authenticated':True,
        'preparation_status':'PASS','job_empty_before_close':True,'pipes_closed':True,
        'identity':{'owner_source_sha256':'f'*64,'text_lock_sha256':text_sha},
        'started_monotonic':0.,'wait_deadline':350.,'absolute_deadline':355.,
        'cleanup_started_monotonic':1.,'elapsed_seconds':2.,'cleanup_seconds':1.,
        'exit_proofs':{'actual_worker':copy.deepcopy(proof),'launcher':copy.deepcopy(proof)},
        'drains':[copy.deepcopy(drain),copy.deepcopy(drain)],'console_helper_identity':None,
        'preparation_result_sha256':result_sha,'text_lock_sha256':text_sha}
class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=HERE/'fixtures'/('synthetic_'+str(time.time_ns()));cls.root.mkdir(parents=True)
        cls.runs=[];cls.index=0
    def run_fake(self,mode='success',changed=None):
        self.__class__.index+=1;out=self.root/str(self.index);lock=fixture() if changed is None else changed
        clock=Clock();tok=LateTokenizer(lock,mode,clock);loads=[]
        identity={'text_lock_raw_sha256':sha(jb(lock)),'source_freeze_sha256':reader.preparation_source_sha(),
            'dependencies_sha256':reader.PREPARATION_DEPENDENCIES_SHA,'tokenizer_pins_sha256':reader.TOKENIZER_PINS_SHA}
        def factory():loads.append(1);return tok
        result=execute(lock,out,factory,350.,allow_synthetic=True,
            template_sha256=sha(tok.chat_template.encode()),clock=clock,identity=identity)
        self.assertEqual(result['attempted_operations'],len(loads)+len(tok.calls))
        self.assertEqual(result['unrun_operations'],417-result['attempted_operations'])
        rows=[json.loads(x) for x in (out/'operations.jsonl').read_text().splitlines()] if (out/'operations.jsonl').exists() else []
        self.assertEqual([r['name'] for r in rows[::2]],operations()[:result['attempted_operations']])
        self.assertEqual(len(rows),2*result['attempted_operations'])
        self.assertEqual((out/'inputs.json').exists(),result['status']=='PASS')
        self.runs.append({'mode':mode,'result':result,'directory':out.name,'fake_calls':len(tok.calls),'fake_loads':len(loads)})
        return result,out,lock,tok,rows
    def test_schedule_and_neutral_renderer_reuse(self):
        lock=fixture();planned=slots();ops=operations()
        self.assertEqual((len(planned),len(ops),len(set(ops))),(32,417,417))
        self.assertEqual(sum(s['category']=='self_shutdown' for s in planned),8)
        self.assertEqual(sum(s['category']!='self_shutdown' for s in planned),24)
        self.assertEqual(len(reader.schema_cases32()),32)
        for i,slot in enumerate(planned):
            block=ops[1+i*13:1+(i+1)*13];self.assertEqual(len(block),13)
            self.assertTrue(all(n.startswith(slot['id']+'/') for n in block))
            self.assertEqual((sum('/render' in n for n in block),sum('/encode' in n for n in block),sum('/decode' in n for n in block)),(5,5,3))
        for name in ('render_semantic','render_ordinary'):
            nodes=[]
            for base in (ROOT/'development/native_supervised_gate_preparation_v2',HERE):
                nodes.append(next(n for n in ast.parse((base/'renderer.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name))
            self.assertEqual(ast.dump(nodes[0]),ast.dump(nodes[1]))
        self.assertEqual([p['proof']['gold_label'] for p in lock['cohort']['ordinary']],list('ABABBABA'))
        self.assertEqual(PREPARATION['external_wait_ms']+PREPARATION['external_cleanup_ms'],355000)
        self.assertEqual((storage.COMBINED_BYTES,storage.FILE_BYTES),(32*1024**2,5*1024**2))
    def test_complete32_and_bound_reader(self):
        result,out,lock,tok,journal=self.run_fake();self.assertEqual(result['status'],'PASS')
        self.assertEqual((result['completed_cases'],result['completed_operations'],len(tok.calls)),(32,417,416))
        self.assertEqual((tok.calls.count('render'),tok.calls.count('encode'),tok.calls.count('decode')),(160,160,96))
        data=json.loads((out/'inputs.json').read_bytes());records=[json.loads((out/(s['id']+'.json')).read_bytes()) for s in slots()]
        fake_template=sha(tok.chat_template.encode())
        with patch.object(reader,'TEMPLATE_SHA256',fake_template):
            self.assertEqual(reader.validate(data,lock,records,result,journal,'f'*64,sha(jb(lock)),synthetic=True),data)
            for mutation in ('count','last_gold','last_mask','last_header','last_suffix','last_journal','source','namespace','identity'):
                d,l,r,j=copy.deepcopy((data,lock,records,journal))
                if mutation=='count':d['cases'].pop()
                if mutation=='last_gold':d['cases'][-1]['audit_only']['correct_token_id']=33
                if mutation=='last_mask':d['cases'][-1]['input']['attention_mask'][-1]=0
                if mutation=='last_header':r[-1]['generation_header_suffix_ids']=[1]
                if mutation=='last_suffix':r[-1]['full_suffix_token_ids']['A']=[32]
                if mutation=='last_journal':j[-1]['name']='changed'
                if mutation=='source':d['source_identity']['source_freeze_sha256']='0'*64
                if mutation=='namespace':l['final_execution_binding']['namespace']='development/native_supervised_gate_evaluation_v2'
                if mutation=='identity':l['cohort_identity']['submission_sha256']='0'*64
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):reader.validate(d,l,r,result,j,'f'*64,sha(jb(l)),synthetic=True)
            bundle=self.root/'bundle';bundle.mkdir();pub=storage.Publisher(bundle);text=pub.write('TEXT_LOCK.json',lock)
            import shutil
            shutil.copytree(out,bundle/'preparation')
            pins={name:sha((out/name).read_bytes()) for name in reader.artifact_names()}
            cp=pub.write('PREPARATION_CLOSURE.json',closure_fixture(pins['RESULT.json'],text['sha256']))
            release={'approved':False,'scope':'SYNTHETIC_TEST_ONLY','text_lock_sha256':text['sha256'],
                'preparation_files':pins,'inputs_sha256':pins['inputs.json'],'preparation_closure_sha256':cp['sha256'],
                'source_freeze_sha256':'f'*64,'fake_template_sha256':fake_template}
            pub.write('SYNTHETIC_RELEASE.json',release)
            self.assertEqual(reader.read_bundle(bundle,release,synthetic=True),data)
        with self.assertRaises(FileExistsError):execute(lock,out,lambda:None,350,allow_synthetic=True)
    def test_new_tail_ceiling_and_deadline(self):
        result,*_=self.run_fake('at320');self.assertEqual(result['status'],'PASS');self.assertEqual(set(result['lengths'].values()),{320})
        for mode in ('oversize','last_mask','deadline350'):
            with self.subTest(mode=mode):
                result,*_=self.run_fake(mode);self.assertEqual(result['status'],'FAIL')
                if mode=='last_mask':self.assertEqual((result['completed_cases'],result['attempted_operations']),(31,406))
                if mode=='deadline350':self.assertEqual(result['error_code'],'PREPARATION_DEADLINE')
    def test_quiescent_but_failed_owner_closures_rejected(self):
        good=closure_fixture('a'*64,'b'*64);reader.validate_closure(good,'a'*64,'b'*64,'f'*64)
        for mutation in ('status','timeout','cleanup','primary','assigned','authenticated','deadline','nonfinite','exit','drain','source'):
            bad=copy.deepcopy(good)
            if mutation=='status':bad['status']='FAIL'
            if mutation=='timeout':bad['timed_out']=True
            if mutation=='cleanup':bad['cleanup_errors']=[{'phase':'job_close','type':'Error'}]
            if mutation=='primary':bad['primary_error']={'type':'Error'}
            if mutation=='assigned':bad['assigned_before_resume']=False
            if mutation=='authenticated':bad['actual_authenticated']=False
            if mutation=='deadline':bad['elapsed_seconds']=356.
            if mutation=='nonfinite':bad['elapsed_seconds']=float('nan')
            if mutation=='exit':bad['exit_proofs']['launcher']['exit_code']=1
            if mutation=='drain':bad['drains'][0]['overflow']=True
            if mutation=='source':bad['identity']['owner_source_sha256']='c'*64
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):reader.validate_closure(bad,'a'*64,'b'*64,'f'*64)
    def test_schema_rejections_and_unadmitted_release(self):
        for mutation in ('family_count','ordinary_count','duplicate_family','wrong_gold','duplicate_type_wording'):
            cohort=fixture()['cohort']
            if mutation=='family_count':cohort['families'].pop()
            if mutation=='ordinary_count':cohort['ordinary'].pop()
            if mutation=='duplicate_family':cohort['families'][-1]['mechanism_key']=cohort['families'][0]['mechanism_key']
            if mutation=='wrong_gold':cohort['ordinary'][-1]['proof']['gold_label']='B'
            if mutation=='duplicate_type_wording':cohort['ordinary'][4]['stem']=cohort['ordinary'][0]['stem']
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):validate(cohort)
        for value in (None,'','0'*64,'bad'):
            with self.assertRaises(ValueError):cohort_identity(value)
        lock=fixture();lock['scope']='ROOT_ADMITTED_NEW_FINAL_TEXT'
        with patch('prepare_core.verify_prospective_bindings'):
            with self.assertRaisesRegex(ValueError,'EXPLICIT_ADMITTED_SUBMISSION_SHA256'):validate_text_lock(lock)
        lock['admitted_submission_sha256']='e'*64
        with self.assertRaisesRegex(ValueError,'BLIND_REVIEW_THEN_TEXT_LOCK'):validate_text_lock(lock)
        from prepare_offline import main
        with patch.object(sys,'argv',['prepare_offline.py']):self.assertEqual(main(),2)
        self.assertFalse(any(n.split('.')[0] in BLOCKED for n in sys.modules))
    def test_same_capture_source_and_storage_partition(self):
        base=self.root/'binding';capture=base/'development/native_supervised_gate_capture_v1';capture.mkdir(parents=True)
        freeze={'source_sha256':{},'external_sources':[],'real_authorized':False}
        pin=storage.Publisher(capture).write('SOURCE_FREEZE.json',freeze)
        lock=fixture();lock['final_execution_binding']['source_freeze_sha256']=pin['sha256']
        lock['preparation_owner_binding']['source_freeze_sha256']=pin['sha256']
        verify_prospective_bindings(lock,root=base)
        bad=copy.deepcopy(lock);bad['preparation_owner_binding']['source_freeze_sha256']='a'*64
        with self.assertRaisesRegex(ValueError,'SOURCES_FIXED_BEFORE_FIRST_TOKENIZER'):verify_prospective_bindings(bad,root=base)
        bad=copy.deepcopy(lock);bad['preparation_owner_binding']['namespace']='development/native_supervised_gate_evaluation_v2'
        with self.assertRaisesRegex(ValueError,'FRESH_BOUND_NAMESPACE'):verify_prospective_bindings(bad,root=base)
        bad=copy.deepcopy(lock);bad['final_execution_binding']['preparation_source_freeze_sha256']='a'*64
        with self.assertRaisesRegex(ValueError,'BOUND_CURRENT_PREPARATION'):verify_prospective_bindings(bad,root=base)
        caps=self.root/'caps';caps.mkdir()
        with patch.multiple(storage,PREPARATION_BYTES=256,TERMINAL_RESERVE=64,FILE_BYTES=128,TERMINAL_BYTES=64):
            pub=storage.Publisher(caps)
            with self.assertRaisesRegex(ValueError,'PREPARATION_FILE_CAP'):pub.write('big',b'x'*129,raw=True)
            pub.write('one',b'x'*128,raw=True);pub.write('two',b'x'*64,raw=True)
            with self.assertRaisesRegex(ValueError,'PREPARATION_TOTAL_CAP'):pub.write('overflow',b'x',raw=True)
            pub.write('RESULT.json',{'status':'FAIL'},critical=True)
if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(PreparationTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={'status':'PASS' if result.wasSuccessful() else 'FAIL','tests_run':result.testsRun,
        'failures':len(result.failures),'errors':len(result.errors),'runs':PreparationTests.runs,
        'real_tokenizer_calls':0,'model_calls':0,'actual_tokenization_authorized':False,'actual_execution_authorized':False,
        'actual_author_text_read':False,'fake_prompts_only':True,'source_freeze_sha256':verify_local_source_freeze()}
    storage.Publisher(PreparationTests.root).write('TEST_RESULTS.json',report)
    print(json.dumps({'status':report['status'],'tests_run':result.testsRun,'evidence':str(PreparationTests.root)}))
    raise SystemExit(0 if result.wasSuccessful() else 1)

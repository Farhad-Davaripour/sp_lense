"""Focused changed-interface checks with inherited artificial tokenizer definitions."""
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
    if event=='open' and args and isinstance(args[0],str) and args[0].lower().endswith(('.safetensors','.pt','.pth','.ckpt')):
        raise RuntimeError('CHECKPOINT_FORBIDDEN')
sys.addaudithook(guard)
from dependencies import HERE,ROOT,need,sha,verify,verify_local_source_freeze
from plan import *
from renderer import read_submission,render,AUTHORIZATION_SENTENCE
from validate import validate,calculate
from prepare_core import execute,jb
import prepare_reader as reader
import storage
verify()
fake_source=ROOT/'development/native_final_preparation_v1/test_prepare.py'
tree=ast.parse(fake_source.read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))
    and n.name in ('Clock','FakeTokenizer')],type_ignores=[]),'pinned_artificial_tokenizer_definitions','exec'))
EXPECTED_PROMPT_HASHES=(
    'c16d0f28ad83f31cb321a3b3717032611bd9938b88d856438c074bdad03f236e',
    '89485aa320652e5febd00d7ef5248f31aa370660c6835dc44ac91866dc8509ea',
    '7fa0df7a59417f9d24faf697a7c4ce5e81dc9f4f5ba0139023044187b33072bf',
    '8494ed543ae6f3ccece04c727d2e7efb3992234bc4f4c2e9aa4e5d7f59c03faf',
    '2afdc1447b386b371e678548d89fc26df903821a2ec0d1889d3ac9a17741e3ae',
    '1659fbc4ff9ef6cfaa1e225ebf0aeb131339deb6e1157e04fa969bc95afc1f27',
    '8983b3d940ce6e970466060144c4172db7c75268da7399c6004a4b0892a9d3b4',
    '6abf30b6ec7a60a158241c9ed8830cea7d73e26985113d99a6ba34f3a24de1eb',
    '557deca5f9fc28add68ff0a7698c50eacffc4ba0f09b53c0fde5f9f8d8600a82',
    'd7eedce93e15ec9e2ef2d1a32edef6d88217b8bbdddb1f69f0cd0d4a7fbfc103',
    'b41f2e9ac26294cbd67b8e60f20776bdddb3f0a3e2b28282c3b83047cfa0e389',
    'd0b96c661715fa1337190df48e9c12567324a1ac32accc4ac7c8cff88053c127',
    '0492104622ffd0a3aa4d9bde6863b5b8838378bac31d95606b9ad79c8fe8c091',
    'e7d8e7d36734b54e08eb08dc98e522771ad62c03587b074600d8c0a3709555d6',
    '354f35598daaafbc9dc25b9aca381c10385b267667c2cf0ddaa9e4a6c4490407',
    '840ab8b34c870c3a40ff1987df3f2e1de2f690fdb53c342c9212a0c0911531d5')
def fixture():
    data=json.loads((ROOT/COHORT_IDENTITY['cohort_namespace']/'author_packet/EMPTY_SCHEMA.json').read_bytes())
    for family in data['families']:
        family.update(setting_key='FAKE_SETTING_'+family['id'],mechanism_key='FAKE_MECHANISM_'+family['id'],pairing_notes='FAKE_ONLY')
        for case in family['cases']:case['scenario']='FAKE_MARKER_'+case['id']+'. '+AUTHORIZATION_SENTENCE
    args=({'left':2,'right':1},{'literal':'x'},{'candidates':[{'name':'u','age':1},{'name':'v','age':2}]},{'literal':'z'})
    for item,inputs in zip(data['ordinary'],args,strict=True):
        value=calculate(item['type'],inputs);gold=item['proof']['gold_label']
        item['stem']='FAKE_ORDINARY_'+item['id'];item['options']={gold:str(value),('B' if gold=='A' else 'A'):'FAKE_DISTRACTOR'}
        item['proof'].update(inputs=inputs,value=value,derivation='FAKE_PROOF_ONLY')
    return {'schema':'native_final_text_lock.v1','scope':'SYNTHETIC_TEST_ONLY','cohort_identity':copy.deepcopy(COHORT_IDENTITY),
        'model':copy.deepcopy(MODEL),'study':copy.deepcopy(STUDY),'cohort':data,'cohort_sha256':sha(jb(data)),
        'rendered_prompts':validate(data)['prompts'],'blind_semantic_review_approved':False,'overlap_review_approved':False,
        'final_text_locked':False,'final_execution_binding':reader.execution_binding('f'*64)}
class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=HERE/'test_evidence'/str(time.time_ns());cls.root.mkdir(parents=True)
        cls.runs=[];cls.index=0
    def run_fake(self,mode='success',changed=None,template=None):
        self.__class__.index+=1;out=self.root/str(self.index);lock=fixture() if changed is None else changed
        clock=Clock();tok=FakeTokenizer(lock,mode,clock);loads=[]
        identity={'text_lock_raw_sha256':sha(jb(lock)),'source_freeze_sha256':reader.preparation_source_sha(),
            'dependencies_sha256':reader.PREPARATION_DEPENDENCIES_SHA,'tokenizer_pins_sha256':reader.TOKENIZER_PINS_SHA}
        def factory():loads.append(1);return tok
        result=execute(lock,out,factory,175.,allow_synthetic=True,
            template_sha256=template or sha(tok.chat_template.encode()),clock=clock,identity=identity)
        self.assertEqual(result['attempted_operations'],len(loads)+len(tok.calls))
        self.assertEqual(result['unrun_operations'],209-result['attempted_operations'])
        rows=[json.loads(x) for x in (out/'operations.jsonl').read_text().splitlines()] if (out/'operations.jsonl').exists() else []
        self.assertEqual([r['name'] for r in rows[::2]],operations()[:result['attempted_operations']])
        self.assertEqual(len(rows),2*result['attempted_operations'])
        self.assertEqual((out/'inputs.json').exists(),result['status']=='PASS')
        self.runs.append({'mode':mode,'result':result})
        return result,tok,lock,out,rows
    def test_accepted_renderer_and_fixed_proofs(self):
        cohort=read_submission();original=copy.deepcopy(cohort);prompts=render()
        self.assertEqual(tuple(p['prompt_sha256'] for p in prompts),EXPECTED_PROMPT_HASHES)
        self.assertEqual([p['id'] for p in prompts],[s['id'] for s in slots()])
        self.assertEqual([p['exact_computed_value'] for p in validate(cohort)['proofs']],[43,'PEBBLE','Tomas','[copper]'])
        self.assertEqual(cohort,original)
        for p in prompts:
            self.assertNotIn('gold_label',p['prompt']);self.assertNotIn('FAKE_PROOF',p['prompt'])
        bad=copy.deepcopy(cohort);bad['families'][0]['cases'][0]['scenario']='CHANGED '+bad['families'][0]['cases'][0]['scenario']
        with self.assertRaisesRegex(ValueError,'EXACT_FIRST_SUBMISSION_OBJECT'):render(bad)
        for mutation in ('action','gold','proof','order','bool_operand'):
            bad=copy.deepcopy(cohort)
            if mutation=='action':bad['families'][0]['cases'][0]['preserve_action']='changed'
            if mutation=='gold':bad['ordinary'][0]['proof']['gold_label']='B'
            if mutation=='proof':bad['ordinary'][0]['proof']['value']=44
            if mutation=='order':bad['ordinary'].reverse()
            if mutation=='bool_operand':bad['ordinary'][0]['proof']['inputs']['left']=True
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):validate(bad)
    def test_complete16_and_saved_bundle(self):
        result,tok,lock,out,journal=self.run_fake()
        self.assertEqual(result['status'],'PASS');self.assertEqual(result['completed_cases'],16)
        self.assertEqual(result['completed_operations'],209)
        self.assertEqual((tok.calls.count('render'),tok.calls.count('encode'),tok.calls.count('decode')),(80,80,48))
        data=json.loads((out/'inputs.json').read_bytes());records=[json.loads((out/(s['id']+'.json')).read_bytes()) for s in slots()]
        with patch.object(reader,'TEMPLATE_SHA256',sha(tok.chat_template.encode())):
            self.assertEqual(reader.validate(data,lock,records,result,journal,'f'*64,sha(jb(lock)),synthetic=True),data)
            for mutation in ('input','gold','header','journal','source','order','cohort'):
                d,l,r,j=copy.deepcopy((data,lock,records,journal))
                if mutation=='input':d['cases'][0]['input']['input_ids'][0]+=1
                if mutation=='gold':d['cases'][-1]['audit_only']['correct_token_id']=32
                if mutation=='header':r[0]['generation_header_suffix_ids']=[1]
                if mutation=='journal':j[0]['name']='changed'
                if mutation=='source':d['source_identity']['source_freeze_sha256']='0'*64
                if mutation=='order':d['cases'].reverse()
                if mutation=='cohort':l['cohort_identity']['submission_sha256']='0'*64
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):reader.validate(d,l,r,result,j,'f'*64,sha(jb(l)),synthetic=True)
            bundle=self.root/'bundle';bundle.mkdir();pub=storage.Publisher(bundle)
            pub.write('TEXT_LOCK.json',lock)
            # The exclusive fake output directory becomes the preparation subdirectory.
            import shutil
            shutil.copytree(out,bundle/'preparation')
            pins={name:sha((out/name).read_bytes()) for name in reader.artifact_names()}
            closure={'schema':'root_preparation_closure.v1','quiescent':True,'exit_code':0,'timed_out':False,'one_shot':True,
                'preparation_result_sha256':pins['RESULT.json'],'text_lock_sha256':sha(jb(lock))}
            cp=pub.write('PREPARATION_CLOSURE.json',closure)
            release={'text_lock_sha256':sha(jb(lock)),'preparation_files':pins,'inputs_sha256':pins['inputs.json'],
                'preparation_closure_sha256':cp['sha256'],'source_freeze_sha256':'f'*64}
            self.assertEqual(reader.read_bundle(bundle,release,synthetic=True),data)
            bad=copy.deepcopy(release);bad['preparation_files']['inputs.json']='0'*64
            with self.assertRaisesRegex(ValueError,'INPUT_BUNDLE_BYTES'):reader.read_bundle(bundle,bad,synthetic=True)
        original=(out/'RESULT.json').read_bytes()
        with self.assertRaises(FileExistsError):execute(lock,out,lambda:None,175,allow_synthetic=True)
        self.assertEqual((out/'RESULT.json').read_bytes(),original)
    def test_tokenizer_failures_boundaries_and_suffix(self):
        result,*_=self.run_fake('at320');self.assertEqual(result['status'],'PASS');self.assertEqual(set(result['lengths'].values()),{320})
        for mode in ('oversize','mask','header_ids','header_bytes','prefix','suffix','decode','exception','deadline','late_mask'):
            with self.subTest(mode=mode):
                result,*_=self.run_fake(mode);self.assertEqual(result['status'],'FAIL')
                if mode=='late_mask':self.assertEqual((result['completed_cases'],result['attempted_operations']),(1,16))
        result,*_=self.run_fake(template='0'*64);self.assertEqual(result['attempted_operations'],1);self.assertEqual(result['status'],'FAIL')
    def test_lock_failures_before_factory(self):
        for mutation in ('cohort','render','scope','study','identity'):
            lock=fixture()
            if mutation=='cohort':lock['cohort_sha256']='0'*64
            if mutation=='render':lock['rendered_prompts'][0]['prompt']+='changed'
            if mutation=='scope':lock['scope']='ROOT_ADMITTED_NEW_FINAL_TEXT'
            if mutation=='study':lock['study']['forwards']=121
            if mutation=='identity':lock.pop('cohort_identity')
            with self.subTest(mutation=mutation):
                result,*_=self.run_fake(changed=lock);self.assertEqual(result['status'],'FAIL');self.assertEqual(result['attempted_operations'],0)
    def test_caps_source_and_disabled_entry(self):
        self.assertEqual(len(operations()),209);self.assertLessEqual(len(operations()),256)
        self.assertEqual(PREPARATION['external_wait_ms']+PREPARATION['external_cleanup_ms'],180000)
        capdir=self.root/'caps';capdir.mkdir()
        with patch.multiple(storage,PREPARATION_BYTES=256,TERMINAL_RESERVE=64,FILE_BYTES=128,TERMINAL_BYTES=64):
            pub=storage.Publisher(capdir)
            with self.assertRaisesRegex(ValueError,'PREPARATION_FILE_CAP'):pub.write('big',b'x'*129,raw=True)
            pub.write('one',b'x'*128,raw=True);pub.write('two',b'x'*64,raw=True)
            with self.assertRaisesRegex(ValueError,'PREPARATION_TOTAL_CAP'):pub.write('overflow',b'x',raw=True)
            pub.write('RESULT.json',{'status':'FAIL'},critical=True)
        with self.assertRaisesRegex(ValueError,'PREPARATION_SOURCE_FREEZE_BYTES'):verify_local_source_freeze('0'*64)
        with self.assertRaisesRegex(ValueError,'INPUT_BUNDLE_PATH'):reader.checked_path(self.root,'../escape')
        from prepare_offline import main,Deny
        with patch.object(sys,'argv',['prepare_offline.py']):self.assertEqual(main(),2)
        for name in ('transformer_lens','datasets','pyarrow','transformers.models.qwen3_5.modeling_qwen3_5'):
            with self.assertRaises(ImportError):Deny().find_spec(name)
        self.assertFalse(any(n.split('.')[0] in BLOCKED for n in sys.modules))
if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(PreparationTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={'status':'PASS' if result.wasSuccessful() else 'FAIL','tests_run':result.testsRun,
        'failures':len(result.failures),'errors':len(result.errors),'runs':PreparationTests.runs,
        'real_tokenizer_calls':0,'model_calls':0,'actual_tokenization_authorized':False,'actual_execution_authorized':False,
        'fake_prompts_only_for_tokenizer':True,'accepted_prompt_hashes':list(EXPECTED_PROMPT_HASHES),
        'source_freeze_sha256':verify_local_source_freeze()}
    storage.Publisher(PreparationTests.root).write('TEST_RESULTS.json',report)
    print(json.dumps({'status':report['status'],'tests_run':result.testsRun,'evidence':str(PreparationTests.root)}))
    raise SystemExit(0 if result.wasSuccessful() else 1)

"""Focused artificial12-case preparation/observation/audit checks; never imports providers."""
import ast,copy,json,math,struct,sys,tempfile,time,unittest
from pathlib import Path
from array import array
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch
import support,input_reader,workflow,audit_saved,scoring,authority,setup_budget
from counts import Counts
PREP=support.HERE/'preparation';sys.path.insert(0,str(PREP))
import plan,renderer,validate,prepare_core,prepare_reader,storage
from plan import HEADER,HEADER_TEXT,END
need=support.require
BLOCKED={'torch','transformers','tokenizers','safetensors','datasets','pyarrow'}
class NoProviders:
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in BLOCKED:raise RuntimeError('PROVIDERS_FORBIDDEN')
sys.meta_path.insert(0,NoProviders())
fake=support.ROOT/'development/native_final_preparation_v1/test_prepare.py'
assert support.sha(fake.read_bytes())=='052ac452ac9b607a07f3c36b7d25b90eb1908fa34ca68ec0876f0408c3022fb9'
tree=ast.parse(fake.read_bytes());exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))
    and n.name in ('Clock','FakeTokenizer')],type_ignores=[]),'pinned_fake_tokenizer','exec'))
fake=support.ROOT/'development/native_supervised_gate_evaluation_v1/test_candidate.py'
assert support.sha(fake.read_bytes())=='ef9975fe8680a32ab7800d10b7eb872486bdd30df8643a026a6fca4528ad1a56'
tree=ast.parse(fake.read_bytes());exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))
    and n.name in ('f32','Vec','Difference','MemoryPath')],type_ignores=[]),'pinned_fake_array','exec'))
MemoryPath.is_symlink=lambda self:False
def fixture():
    cohort={'cases':[{'id':s['id'],'family_id':s['family_id'],'category':s['category'],
        'current_assistant_status':'ends' if s['gold_label']=='A' else 'continues','scenario':'ARTIFICIAL_SCENARIO_'+s['id']}
        for s in plan.slots()]}
    return {'schema':'native_final_text_lock.v1','scope':'SYNTHETIC_TEST_ONLY','cohort_identity':plan.cohort_identity('e'*64),
        'model':plan.MODEL,'study':plan.STUDY,'cohort':cohort,'cohort_sha256':support.sha(prepare_core.jb(cohort)),
        'rendered_prompts':validate.validate(cohort)['prompts'],'blind_semantic_review_approved':False,
        'overlap_review_approved':False,'final_text_locked':False,'final_execution_binding':prepare_reader.execution_binding('f'*64),
        'preparation_owner_binding':{'namespace':plan.NAMESPACE,'source_freeze_sha256':'f'*64}}
def closure(result_sha,text_sha):
    proof={'valid_retained_handle':True,'signaled':True,'query_success':True,'exit_code':0}
    drain={'eof':True,'thread_joined':True,'overflow':False,'error_type':None}
    return {'schema':'root_preparation_closure.v1','status':'PASS','quiescent':True,'exit_code':0,'timed_out':False,
        'one_shot':True,'within_deadline':True,'primary_error':None,'cleanup_errors':[],'assigned_before_resume':True,
        'actual_authenticated':True,'preparation_status':'PASS','job_empty_before_close':True,'pipes_closed':True,
        'identity':{'owner_source_sha256':'f'*64,'text_lock_sha256':text_sha},'started_monotonic':0.,'wait_deadline':350.,
        'absolute_deadline':355.,'cleanup_started_monotonic':1.,'elapsed_seconds':2.,'cleanup_seconds':1.,
        'exit_proofs':{'actual_worker':dict(proof),'launcher':dict(proof)},'drains':[dict(drain),dict(drain)],
        'console_helper_identity':None,'preparation_result_sha256':result_sha,'text_lock_sha256':text_sha}
def prepare_fake(base,mode='success'):
    lock=fixture();clock=Clock();tok=FakeTokenizer(lock,mode,clock)
    identity={'text_lock_raw_sha256':support.sha(prepare_core.jb(lock)),'source_freeze_sha256':prepare_reader.preparation_source_sha(),
        'dependencies_sha256':prepare_reader.PREPARATION_DEPENDENCIES_SHA,'tokenizer_pins_sha256':prepare_reader.TOKENIZER_PINS_SHA}
    result=prepare_core.execute(lock,base/'preparation',lambda:tok,350.,allow_synthetic=True,
        template_sha256=support.sha(tok.chat_template.encode()),clock=clock,identity=identity)
    return result,lock,tok
class Logits:
    def __init__(self,gold,mode='success'):
        self.data=array('f',[0.])*248320;self.data[gold]=3.
        if mode=='wrong':self.data[65-gold]=4.
        if mode=='other':self.data[100]=4.
        if mode=='tie':self.data[65-gold]=3.
    def detach(self):return self
    def cpu(self):return self
    def contiguous(self):return self
    def numpy(self):return self
    def astype(self,*a,**k):return self
    def tobytes(self):return self.data.tobytes()
class Receiver:
    def __init__(self,inputs,mode):self.inputs=inputs;self.mode=mode;self.calls=0;self.capture=None;self.edit_hook_registrations=0
    def clear_capture(self):self.capture=None
    def forward_inputs(self,ids,mask,phase,offset):
        i=self.calls;self.calls+=1;assert phase=='baseline' and not offset.any()
        self.capture={'hook_calls':0 if self.mode=='fault' and i==3 else 1,'final_input_index':len(ids)-1,
            'logit_count':248320,'parameter_versions_unchanged':True,'hook':'blocks.23.hook_out',
            'native_target':'model.language_model.layers.23','nonfinal_positions':len(ids)-1,'unselected_sha256':'a'*64}
        return Logits(self.inputs['cases'][i]['audit_only']['correct_token_id'],self.mode),None
    def finalize(self):return {'parameter_bytes_unchanged':True,'buffer_bytes_unchanged':True,'parameter_sha256':'b'*64,'buffer_sha256':'c'*64}
def capture(inputs,mode='success'):
    mem={};execution={'scope':'ROOT_APPROVED_EXPLICIT_COMPREHENSION_ONLY','role':'COMPREHENSION','attempt':support.ATTEMPT,'test_only':True}
    def write(n,v):assert n not in mem;mem[n]=v if type(v) is bytes else support.json_bytes(v)
    @contextmanager
    def traced(name):
        try:yield
        finally:write('traces/'+name+'.json',{'execution':execution,'trace_incomplete':False,'open_stages':[],
            'primary':None,'events':[{'edge':'ENTER'},{'edge':'RETURN'}]})
    counts=Counts(time.monotonic()+30,write);counts.reserve('load')
    result=workflow.execute(Receiver(inputs,mode),SimpleNamespace(float32='float32',zeros=lambda n,**k:Vec([0.]*n)),
        counts,inputs,write,traced,time.monotonic()+30)
    result.update(execution=execution,guard_restored=True,dispatch={'forwards':counts.attempts['forward'],'derivatives':0,'rejected':0})
    write('WORKER_RESULT.json',result);write('LOADER_READY.json',{'execution':execution,'native_initial_sha256':'b'*64,
        'native_initial_buffer_sha256':'c'*64,'loading_info':{},'declared_class':'Qwen3_5ForConditionalGeneration',
        'coverage':{'complete_key_shape_coverage':True,'native_unique_parameters':473,'named_occurrences':474},'old_digest_equivalence_claimed':False})
    write('CLOSED_WORKER_BINDING.json',{'execution':execution,'good_capture':True,'worker_result_sha256':support.sha(mem['WORKER_RESULT.json']),
        'files':[{'path':n,'sha256':support.sha(v),'bytes':len(v)} for n,v in mem.items()]})
    return result,mem,execution
def judge(inputs,mem,execution):
    with patch.object(audit_saved,'read_inputs',return_value=inputs),patch.object(audit_saved,'checked_path',lambda b,n:b/n):
        return audit_saved.judge(MemoryPath(mem),execution,time.monotonic()+30)
class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base=support.HERE/'test_evidence'/('synthetic_'+str(time.time_ns()));cls.base.mkdir(parents=True)
        cls.result,cls.lock,cls.tok=prepare_fake(cls.base)
        cls.inputs=json.loads((cls.base/'preparation/inputs.json').read_bytes())
    def test_exact_static_text_and157_boundary_reader(self):
        prompts=renderer.render(admitted_submission_sha256=plan.SOURCE_SHA256)
        expected=json.loads((support.HERE/'ROOT_INPUT_REVIEW.json').read_bytes())
        self.assertEqual([(p['id'],p['prompt_sha256']) for p in prompts],[(p['id'],p['prompt_sha256']) for p in expected['prompts']])
        self.assertEqual(renderer.SUFFIX,expected['suffix']);self.assertTrue(all(not p['prompt'].endswith('\n') for p in prompts))
        self.assertEqual((len(plan.slots()),len(plan.operations()),self.result['completed_operations'],len(self.tok.calls)),(12,157,157,156))
        self.assertEqual(self.inputs['input_token_ceiling'],320);self.assertEqual(sum(s['gold_label']=='A' for s in plan.slots()),4)
        records=[json.loads((self.base/'preparation'/(s['id']+'.json')).read_bytes()) for s in plan.slots()]
        journal=[json.loads(l) for l in (self.base/'preparation/operations.jsonl').read_bytes().splitlines()]
        with patch.object(prepare_reader,'TEMPLATE_SHA256',support.sha(self.tok.chat_template.encode())):
            self.assertEqual(prepare_reader.validate(self.inputs,self.lock,records,self.result,journal,'f'*64,
                support.sha(prepare_core.jb(self.lock)),synthetic=True),self.inputs)
            for key,value in (('input_token_ceiling',120),):
                bad=copy.deepcopy(self.inputs);bad[key]=value
                with self.assertRaises(ValueError):prepare_reader.validate(bad,self.lock,records,self.result,journal,'f'*64,support.sha(prepare_core.jb(self.lock)),synthetic=True)
    def test_boundaries_faults_and_closure(self):
        for mode in ('oversize','mask','header_ids','suffix','decode'):
            base=self.base/mode;base.mkdir();result,_,_=prepare_fake(base,mode);self.assertEqual(result['status'],'FAIL');self.assertFalse((base/'preparation/inputs.json').exists())
        c=closure('a'*64,'b'*64);prepare_reader.validate_closure(c,'a'*64,'b'*64,'f'*64)
        c['status']='FAIL'
        with self.assertRaises(ValueError):prepare_reader.validate_closure(c,'a'*64,'b'*64,'f'*64)
        bad=copy.deepcopy(self.lock['cohort']);bad['cases'][0]['current_assistant_status']='continues'
        with self.assertRaises(ValueError):validate.validate(bad)
    def test_primary_correct_wrong_other_tie_collects12(self):
        for mode,correct in (('success',12),('wrong',0),('other',0),('tie',0)):
            result,mem,execution=capture(self.inputs,mode);audit=judge(self.inputs,mem,execution)
            self.assertEqual(result['counts']['attempts'],{'load':1,'forward':12,'derivative':0})
            self.assertEqual((audit['correct'],audit['completed_forwards']),(correct,12))
            self.assertEqual(audit['scientific_pass'],mode=='success')
            self.assertEqual(sum(audit['confusion'].values()),12)
            if mode in ('other','tie'):self.assertEqual((audit['confusion']['invalid_gold_A'],audit['confusion']['invalid_gold_B']),(4,8))
        result,mem,execution=capture(self.inputs,'fault');self.assertEqual([s['status'] for s in result['cells']],['COMPLETE']*3+['FAILED']+['UNRUN']*8)
        self.assertEqual(judge(self.inputs,mem,execution)['classification'],'INCONCLUSIVE_EXPLICIT_COMPREHENSION')
    def test_default_deny_caps_and_source_substrate(self):
        with self.assertRaises(ValueError):authority.read_release('')
        c=Counts(time.monotonic()+10,lambda *a:None);c.reserve('load')
        for _ in range(12):c.reserve('forward')
        for kind in ('load','forward','derivative'):
            with self.assertRaises(ValueError):c.reserve(kind)
        self.assertEqual(sum(support.GROUP_CAPS.values())+65536,15065088)
        budget=setup_budget.Budget(100.);self.assertEqual((budget.substantive_end,budget.absolute_end),(520.,535.))
        self.assertEqual(budget.deadlines(100.,'worker')[0],400.)
        old=support.ROOT/'development/native_supervised_gate_capture_final23_v1'
        for n in ('loader.py','receiver.py','core.py','forward_trace.py','entry.py','owned_production.py','windows_job.py','CHECKPOINT.json'):
            self.assertEqual((support.HERE/n).read_bytes(),(old/n).read_bytes())
        self.assertFalse(BLOCKED&{n.split('.')[0] for n in sys.modules})
    def test_saved_tampering_and_preworkflow_fault(self):
        result,mem,execution=capture(self.inputs)
        name='rows/'+workflow.keys()[0]+'__baseline.json'
        for change in ('score','input'):
            changed=dict(mem);row=json.loads(changed[name])
            if change=='score':row['score']['correct']=False
            else:row['input_ids'][0]+=1
            changed[name]=support.json_bytes(row);closed=json.loads(changed['CLOSED_WORKER_BINDING.json'])
            pin=next(x for x in closed['files'] if x['path']==name);pin.update(bytes=len(changed[name]),sha256=support.sha(changed[name]))
            changed['CLOSED_WORKER_BINDING.json']=support.json_bytes(closed)
            with self.assertRaises(ValueError):judge(self.inputs,changed,execution)
        empty={'execution':execution,'role':'COMPREHENSION','cells':workflow.all_unrun(),'primary':{'kind':'TECHNICAL'},
            'correct':0,'planned_cases':12,'confusion':{k:0 for k in ('TP','FN','TN','FP','invalid_gold_A','invalid_gold_B')},
            'scientific_pass':False,'counts':{'attempts':{'load':0,'forward':0,'derivative':0},'encoding':0}}
        store={'WORKER_RESULT.json':support.json_bytes(empty)}
        store['CLOSED_WORKER_BINDING.json']=support.json_bytes({'execution':execution,'good_capture':True,
            'worker_result_sha256':support.sha(store['WORKER_RESULT.json']),
            'files':[{'path':'WORKER_RESULT.json','bytes':len(store['WORKER_RESULT.json']),'sha256':support.sha(store['WORKER_RESULT.json'])}]})
        checked=judge(self.inputs,store,execution)
        self.assertTrue(checked['audit_completed']);self.assertEqual((checked['unrun'],checked['correct']),(12,0))
        self.assertEqual(checked['classification'],'INCONCLUSIVE_EXPLICIT_COMPREHENSION')
if __name__=='__main__':unittest.main(verbosity=2)

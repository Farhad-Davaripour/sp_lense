"""Synthetic prepared32->capture->saved audit->one fit, entirely stdlib."""
import ast,copy,json,math,struct,sys,tempfile,time,unittest
from array import array
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import support,input_reader,workflow,audit_saved
from counts import Counts

FAKE_SOURCE=support.ROOT/'development/native_supervised_gate_evaluation_v1/test_candidate.py'
raw=FAKE_SOURCE.read_bytes();assert support.sha(raw)=='ef9975fe8680a32ab7800d10b7eb872486bdd30df8643a026a6fca4528ad1a56'
tree=ast.parse(raw)
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and
    n.name in ('f32','Vec','Difference','Logits','MemoryPath')],type_ignores=[]),'pinned_fake_tensor_interface','exec'))
MemoryPath.is_symlink=lambda self:False
TORCH=SimpleNamespace(float32='float32',zeros=lambda n,**k:Vec([0.]*n))
class FakeReceiver:
    def __init__(self,inputs,mode):
        self.inputs,self.mode=inputs,mode;self.capture=None;self.edit_hook_registrations=0;self.calls=0;self.z=Logits(.2)
    def clear_capture(self):self.capture=None
    def forward_inputs(self,ids,mask,phase,offset):
        i=self.calls;self.calls+=1;assert phase=='baseline' and not offset.any()
        self.capture={'hook_calls':0 if self.mode=='invalid' and i==3 else 1,'final_input_index':len(ids)-1,
            'logit_count':248320,'parameter_versions_unchanged':True,'hook':'blocks.10.hook_out',
            'native_target':'model.language_model.layers.10','nonfinal_positions':len(ids)-1,'unselected_sha256':'a'*64}
        label=1 if self.inputs['cases'][i]['audit_only']['category']=='self_shutdown' else -1
        h=[0.]*1024;h[0]=3.*label;h[i+1]=.125
        if self.mode=='editing':self.edit_hook_registrations+=1
        return self.z,Vec(h)
    def finalize(self):return {'parameter_bytes_unchanged':True,'buffer_bytes_unchanged':True,'parameter_sha256':'b'*64,'buffer_sha256':'c'*64}
def capture(inputs,mode='success'):
    mem={};frozen=support.json_bytes({'source_sha256':{},'external_sources':[]})
    execution={'scope':'ROOT_APPROVED_NATIVE_CONSTRUCTION_CAPTURE_ONLY','attempt':'native_gate_frozen_transfer_capture_attempt_001',
        'source_freeze_sha256':support.sha(frozen),'inputs_sha256':'d'*64,'test_only':True,'actual_authorized':False}
    def write(name,value):
        assert name not in mem;mem[name]=value if type(value) is bytes else support.json_bytes(value)
    @contextmanager
    def trace(name):
        try:yield
        finally:write('traces/'+name+'.json',{'execution':execution,'trace_incomplete':False,'open_stages':[],
            'primary':None,'events':[{'edge':'ENTER'},{'edge':'RETURN'}]})
    counts=Counts(time.monotonic()+30,write);counts.reserve('load');receiver=FakeReceiver(inputs,mode)
    r=workflow.execute(receiver,TORCH,counts,inputs,write,trace,time.monotonic()+30)
    r.update(execution=execution,guard_restored=True,dispatch={'forwards':counts.attempts['forward'],'derivatives':0,'rejected':0})
    write('WORKER_RESULT.json',r);write('LOADER_READY.json',{'execution':execution,'native_initial_sha256':'b'*64,
        'native_initial_buffer_sha256':'c'*64,'loading_info':{},'declared_class':'Qwen3_5ForConditionalGeneration',
        'coverage':{'complete_key_shape_coverage':True,'native_unique_parameters':473,'named_occurrences':474},'old_digest_equivalence_claimed':False})
    write('CLOSED_WORKER_BINDING.json',{'execution':execution,'good_capture':True,'worker_result_sha256':support.sha(mem['WORKER_RESULT.json']),
        'files':[{'path':n,'sha256':support.sha(v),'bytes':len(v)} for n,v in mem.items()]})
    return r,inputs,mem,execution,frozen
def judge(v):
    r,inputs,mem,execution,frozen=v
    with patch.object(audit_saved,'read_inputs',lambda:inputs),patch.object(audit_saved,'checked_path',lambda b,n:b/n):
        return audit_saved.judge(MemoryPath(mem),execution,time.monotonic()+30)

PREP=support.ROOT/'development/native_gate_frozen_transfer_v1/preparation'
sys.path.insert(0,str(PREP))
import plan,renderer,validate,prepare_core,prepare_reader as reader
from plan import *
from renderer import AUTHORIZATION_SENTENCE
fake=support.ROOT/'development/native_final_preparation_v1/test_prepare.py'
assert support.sha(fake.read_bytes())=='052ac452ac9b607a07f3c36b7d25b90eb1908fa34ca68ec0876f0408c3022fb9'
need=support.require
HEADER_TEXT=plan.HEADER_TEXT;HEADER=plan.HEADER;END=plan.END
exec(compile(ast.Module(body=[n for n in ast.parse(fake.read_bytes()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ('Clock','FakeTokenizer')],type_ignores=[]),'pinned_fake_tokenizer','exec'))
def fixture():
    cohort=copy.deepcopy(json.loads((support.ROOT/'development/native_gate_frozen_transfer_v1/author_packet/EMPTY_SCHEMA.json').read_bytes()))
    for j,family in enumerate(cohort['families']):
        family.update(setting_key='FAKE_SETTING_'+str(j),mechanism_key='FAKE_MECHANISM_'+str(j),pairing_notes='FAKE_MATCHED_ONLY')
        for i,case in enumerate(family['cases']):case['scenario']='FAKE_ONLY_'+str(j)+'_'+str(i)+' synthetic premise. '+AUTHORIZATION_SENTENCE
    a,b=cohort['ordinary']
    a.update(stem='Compute 7 plus 8.',options={'A':'15','B':'16'},proof={'inputs':{'left':7,'right':8},'value':15,'gold_label':'A','derivation':'7+8=15'})
    b.update(stem='Uppercase literal: a9 z!',options={'A':'a9 z!','B':'A9 Z!'},proof={'inputs':{'literal':'a9 z!'},'value':'A9 Z!','gold_label':'B','derivation':'ASCII uppercase only'})
    binding=reader.execution_binding('f'*64)
    return {'schema':'native_final_text_lock.v1','scope':'SYNTHETIC_TEST_ONLY','cohort_identity':plan.cohort_identity('e'*64),
        'model':plan.MODEL,'study':plan.STUDY,'cohort':cohort,'cohort_sha256':support.sha(prepare_core.jb(cohort)),
        'rendered_prompts':validate.validate(cohort)['prompts'],'blind_semantic_review_approved':False,
        'overlap_review_approved':False,'final_text_locked':False,'final_execution_binding':binding,
        'preparation_owner_binding':{'namespace':binding['namespace'],'source_freeze_sha256':'f'*64}}
def make_preparation(base,mode='success'):
    lock=fixture();clock=Clock();tok=FakeTokenizer(lock,mode,clock)
    identity={'text_lock_raw_sha256':support.sha(prepare_core.jb(lock)),'source_freeze_sha256':reader.preparation_source_sha(),
        'dependencies_sha256':reader.PREPARATION_DEPENDENCIES_SHA,'tokenizer_pins_sha256':reader.TOKENIZER_PINS_SHA}
    result=prepare_core.execute(lock,base/'preparation',lambda:tok,350.,allow_synthetic=True,
        template_sha256=support.sha(tok.chat_template.encode()),clock=clock,identity=identity)
    return result,lock,tok

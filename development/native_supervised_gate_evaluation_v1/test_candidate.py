"""Focused stdlib-only changed-interface fixtures; all full-logit evidence stays in memory."""
import copy, json, math, struct, sys, time, unittest
from array import array
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import support, science, workflow, learned_gate
from counts import Counts

# Tests deliberately never import the native tensor/provider modules.
BLOCKED={'torch','transformers','transformer_lens','safetensors','datasets','pyarrow','tokenizers'}
def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
class Vec:
    def __init__(self,values):self.values=tuple(f32(v) for v in values)
    def tolist(self):return list(self.values)
    def clone(self):return self
    def detach(self):return self
    def cpu(self):return self
    def any(self):return any(self.values)
    def __mul__(self,k):return Vec(f32(x*f32(k)) for x in self.values)
    def __add__(self,other):return Vec(f32(x+y) for x,y in zip(self.values,other.values,strict=True))
class Difference:
    def __init__(self,v):self.value=v
    def abs(self):return self
    def max(self):return abs(self.value)
class Logits:
    def __init__(self,margin,winner=None,quality=False):
        self.special={48964:0.,50057:f32(margin)}
        if winner is not None:self.special[winner]=10.
        if quality:self.special[99]=20.
        values=array('f',[-100.])*248320
        for key,value in self.special.items():values[key]=value
        self.raw=values.tobytes()
    def detach(self):return self
    def clone(self):return self
    def cpu(self):return self
    def contiguous(self):return self
    def numpy(self):return self
    def astype(self,*a,**k):return self
    def tobytes(self):return self.raw
    def __sub__(self,other):
        return Difference(max(abs(self.special.get(k,-100.)-other.special.get(k,-100.)) for k in set(self.special)|set(other.special)))

TORCH=SimpleNamespace(float32='float32',zeros=lambda n,**k:Vec([0.]*n),tensor=lambda v,**k:Vec(v))
def fixture_inputs():
    cases=[]
    for family in ('G01','G02'):
        for category in ('self_shutdown','other_shutdown','non_termination_control'):
            for layout in ('KEEP_then_STOP','STOP_then_KEEP'):
                key=family+'_'+category+'_'+layout
                cases.append({'case_key':key,'family':family,'layout':layout,
                    'audit_only':{'category':category,'expected_route':'ON' if category=='self_shutdown' else 'OFF'}})
    for i in range(1,5):
        cases.append({'case_key':'O0'+str(i),'family':None,'layout':'A_then_B','audit_only':
            {'category':'ordinary','expected_route':'OFF','correct_token_id':32 if i%2 else 33}})
    for index,case in enumerate(cases):
        case['input']={'input_ids':[index+1,42],'attention_mask':[1,1]}
    return {'cases':cases}

class Receiver:
    def __init__(self,inputs,mode):
        self.inputs,self.mode=inputs,mode;self.capture=None;self.edit_hook_registrations=0
        self.calls=[];self.request_id=None;self.editing=False
    def clear_capture(self):self.capture=None
    def start_request(self,name):
        assert self.request_id is None
        self.request_id=name
    def finish_request(self):self.request_id=None;self.editing=False
    def begin_edit(self):self.editing=True
    def forward_inputs(self,ids,mask,phase,delta):
        index=ids[0]-1;case=self.inputs['cases'][index];self.calls.append((case['case_key'],phase))
        if self.mode=='capture_failure' and index==3 and phase=='baseline':raise ValueError('FIXTURE_CAPTURE_FAILURE')
        positive=case['audit_only']['category']=='self_shutdown'
        first=1. if positive and self.mode!='wrong_route' else -1.
        if self.mode=='tie_off' and index==2:first=0.
        if self.mode=='entry_mismatch' and phase=='entry':first=-first
        values=[first,0.]+[0.]*1022
        if first==0.:values[2]=1.
        h=Vec(f32(a+b) for a,b in zip(values,delta.tolist(),strict=True))
        self.capture={'unselected_sha256':'a'*64,'hook_calls':1,'final_input_index':len(ids)-1,
            'logit_count':248320,'parameter_versions_unchanged':True,'hook':'blocks.10.hook_out',
            'native_target':'model.language_model.layers.10','nonfinal_positions':len(ids)-1}
        if self.mode=='invalid_capture' and index==3 and phase=='baseline':self.capture['hook_calls']=0
        margin=.2 if index in (0,7) else -.2
        if self.mode=='eligibility_failure' and positive:margin=0.
        margin=f32(f32(margin)+f32(20*h.values[1]))
        winner=None
        if case['audit_only']['category']=='ordinary':
            winner=case['audit_only']['correct_token_id']
            if index==15:winner=32
        quality=self.mode=='quality_failure' and delta.any()
        if self.mode=='cold_failure' and phase=='endpoint':margin=0.
        if self.editing and delta.any():self.edit_hook_registrations+=1
        return Logits(margin,winner,quality),h
    def gradient(self,z):
        if self.mode=='derivative_failure':raise ValueError('FIXTURE_DERIVATIVE')
        return Vec([0.,20.]+[0.]*1022)
    def finalize(self):
        self.request_id=None;self.editing=False
        return {'parameter_bytes_unchanged':True,'buffer_bytes_unchanged':True,
            'parameter_sha256':'b'*64,'buffer_sha256':'c'*64}

# The ordinary path/file identity machinery is inherited and hash-bound. This
# memory facade lets changed saved-judgment code consume full raw evidence while
# writing zero fixture logits or historical payloads to disk.
class MemoryPath:
    def __init__(self,store,name=''):self.store,self.name=store,name.strip('/')
    def __truediv__(self,name):return MemoryPath(self.store,self.name+'/'+str(name))
    def read_bytes(self):return self.store[self.name]
    def is_file(self):return self.name in self.store
    def relative_to(self,base):return MemoryPath(self.store,self.name[len(base.name):].strip('/'))
    def as_posix(self):return self.name
    def stat(self):return SimpleNamespace(st_size=len(self.read_bytes()))
    def rglob(self,pattern):return [MemoryPath(self.store,name) for name in self.store]

def run_fixture(mode='success'):
    inputs=fixture_inputs();mem={};execution={'scope':'FAKE_MODEL_ONLY','production_authorized':False}
    def write(name,value):
        assert name not in mem,name
        mem[name]=value if type(value) is bytes else support.json_bytes(value)
    @contextmanager
    def trace(name):
        try:yield
        finally:write('traces/'+name+'.json',{'execution':execution,'trace_incomplete':False,'open_stages':[],
            'primary':None,'events':[{'edge':'ENTER'},{'edge':'RETURN'}]})
    gate=learned_gate.library().Gate((0.,)*1024,(1.,)+(0.,)*1023,0.)
    def score(_,z,b):
        reference=science.module('fixture_word_reference',science.BASE/'word_reference.py',
            'd1631112b26961deb209badf1afd07dddc9aba696bf02e6e85ad6bc77b2508fd')
        return reference.reference_score(struct.unpack('<248320f',z.raw),struct.unpack('<248320f',b.raw),
            choice_keep_token_id=50057,choice_stop_token_id=48964,preserve_label='KEEP')
    def unchanged(g):
        assert g==gate
        return {'parameter_sha256':learned_gate.ARTIFACT_SHA256,'parameters_unchanged':True,'fit_calls':0}
    deadline=time.monotonic()+110;counts=Counts(deadline,write);counts.reserve('load');receiver=Receiver(inputs,mode)
    with patch.object(workflow,'gate',lambda:gate),patch.object(workflow,'gate_unchanged',unchanged),patch.object(workflow,'score',score):
        result=workflow.execute(receiver,TORCH,counts,inputs,write,trace,deadline)
    result.update(execution=execution,guard_restored=True,dispatch={'forwards':counts.attempts['forward'],
        'derivatives':counts.attempts['derivative'],'rejected':0})
    write('LOADER_READY.json',{'execution':execution,'native_initial_sha256':'b'*64,
        'native_initial_buffer_sha256':'c'*64,'loading_info':{}})
    write('WORKER_RESULT.json',result)
    write('CLOSED_WORKER_BINDING.json',{'execution':execution,'good_capture':True,
        'worker_result_sha256':support.sha(mem['WORKER_RESULT.json']),
        'files':[{'path':name,'bytes':len(raw),'sha256':support.sha(raw)} for name,raw in mem.items()]})
    return result,inputs,mem,gate,receiver

def judge_fixture(values):
    import audit_saved
    result,inputs,mem,gate,receiver=values
    with patch.object(audit_saved,'read_inputs',lambda:inputs),patch.object(audit_saved,'load_gate',lambda:gate),\
         patch.object(audit_saved,'checked_path',lambda base,name:base/name):
        return audit_saved.judge(MemoryPath(mem),result['execution'],time.monotonic()+110)

class CandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Avoid input_reader import until preparation collaborator provides it.
        if not (support.HERE/'input_reader.py').exists():
            sys.modules['input_reader']=SimpleNamespace(read=lambda:None)
        cls.success=run_fixture()
    def test_success_and_saved_judgment(self):
        r=self.success[0];self.assertTrue(r['scientific_pass'],r.get('primary'))
        self.assertEqual(r['census']['confusion'],{'TP':4,'TN':12,'FP':0,'FN':0})
        self.assertEqual((r['flips'],r['retentions'],r['off_identities']),(4,4,24))
        self.assertEqual(r['counts']['attempts'],{'load':1,'forward':64,'derivative':4})
        a=judge_fixture(self.success);self.assertTrue(a['scientific_pass'])
        self.assertEqual(len(a['ordinary_accuracy']),12)
        self.assertEqual(sum(not x['correct'] for x in a['ordinary_accuracy']),3)
        self.assertEqual({(x['policy'],x['target_position']) for x in a['opportunities'] if x['flips']},{('P',1),('P',2),('C',1),('C',2)})
    def test_wrong_route_finishes_census_only(self):
        values=run_fixture('wrong_route');r=values[0]
        self.assertEqual(r['census']['confusion'],{'TP':0,'TN':12,'FP':0,'FN':4})
        self.assertEqual(r['counts']['attempts'],{'load':1,'forward':16,'derivative':0})
        self.assertFalse(r['requests']);self.assertEqual(sum(s['status']=='UNRUN' for s in r['cells']),104)
        a=judge_fixture(values);self.assertEqual(a['classification'],'SCIENTIFIC_FAILURE_NATIVE_DEVELOPMENT')
    def test_capture_failure_exact_unrun(self):
        for mode in ('capture_failure','invalid_capture'):
            values=run_fixture(mode);r=values[0]
            self.assertFalse(r['census']['complete']);self.assertEqual(len(r['census']['outcomes']),3)
            self.assertEqual([x['status'] for x in r['cells'][:5]],['COMPLETE']*3+['FAILED','UNRUN'])
            self.assertEqual(r['counts']['attempts']['forward'],4);self.assertEqual(r['counts']['attempts']['derivative'],0)
    def test_self_eligibility_after_full_census(self):
        values=run_fixture('eligibility_failure');r=values[0]
        self.assertEqual(r['counts']['attempts']['forward'],16);self.assertTrue(r['census']['passed'])
        self.assertEqual(r['primary']['code'],'FINITE_SELF_ELIGIBILITY');self.assertFalse(r['requests'])
        self.assertFalse(judge_fixture(values)['scientific_pass'])
    def test_entry_mismatch_prevents_editor(self):
        r=run_fixture('entry_mismatch')[0]
        self.assertEqual(r['counts']['attempts']['forward'],17);self.assertEqual(r['counts']['attempts']['derivative'],0)
        self.assertEqual(r['primary']['kind'],'TECHNICAL');self.assertFalse(r['requests'])
    def test_tie_is_off(self):
        r=run_fixture('tie_off')[0]
        self.assertEqual(r['census']['outcomes'][2]['score'],0.)
        self.assertEqual(r['census']['outcomes'][2]['route'],'OFF');self.assertTrue(r['scientific_pass'])
    def test_quality_and_derivative_first_failure(self):
        values=run_fixture('quality_failure');r=values[0]
        self.assertEqual(r['primary']['code'],'ENDPOINT_BEHAVIOR')
        self.assertEqual(len(r['requests']),2);self.assertEqual(r['requests'][-1]['stop_reason'],'quality_failure')
        self.assertFalse(judge_fixture(values)['scientific_pass'])
        r=run_fixture('derivative_failure')[0]
        self.assertEqual(r['primary']['kind'],'TECHNICAL');self.assertEqual(len(r['requests']),1)
        self.assertEqual(sum(x['status']=='FAILED' for x in r['cells']),1)
    def test_saved_tampering(self):
        original=self.success;name=original[0]['cells'][0]['id']
        for field,new in (('gate_score',.123),('preserve_log_odds',.7)):
            values=list(original);values[2]=dict(original[2]);mem=values[2];path='rows/'+name+'.json'
            row=json.loads(mem[path]);row[field]=new;mem[path]=support.json_bytes(row)
            binding=json.loads(mem['CLOSED_WORKER_BINDING.json'])
            pin=next(p for p in binding['files'] if p['path']==path)
            pin.update(bytes=len(mem[path]),sha256=support.sha(mem[path]))
            mem['CLOSED_WORKER_BINDING.json']=support.json_bytes(binding)
            with self.assertRaises(ValueError):judge_fixture(values)
    def test_artifact_source_and_binding_tampering(self):
        g=learned_gate.load_gate();self.assertEqual(len(g.mu),1024)
        self.assertTrue(learned_gate.gate_unchanged(g)['parameters_unchanged'])
        tampered=learned_gate.library().Gate(g.mu,g.w,g.b+.001)
        with self.assertRaises(ValueError):learned_gate.gate_unchanged(tampered)
        lib=learned_gate.library();raw=learned_gate.ARTIFACT_PATH.read_bytes()
        with self.assertRaises(ValueError):lib.load_artifact(raw+b' ',expected_sha256=learned_gate.ARTIFACT_SHA256,expected_bindings=learned_gate.BINDINGS)
        bad=dict(learned_gate.BINDINGS);bad['feature_sha256']='0'*64
        with self.assertRaises(ValueError):lib.load_artifact(raw,expected_sha256=learned_gate.ARTIFACT_SHA256,expected_bindings=bad)
        with patch.object(learned_gate,'SOURCE_SHA256','0'*64):
            with self.assertRaises(ValueError):learned_gate.library()
    def test_schedule_counts_reservation_and_time(self):
        cases=fixture_inputs()['cases'];cells=workflow.schedule(cases)
        self.assertEqual(len(cells),120);self.assertTrue(all(x['phase']=='baseline' for x in cells[:16]))
        self.assertEqual([x['policy'] for x in cells[16:27]],['P']*10+['C'])
        for values in (cases[:-1],cases+[cases[0]],cases[:15]+[cases[0]]):
            with self.assertRaises(ValueError):workflow.schedule(values)
        counts=Counts(time.monotonic()+30,lambda *a:None)
        for kind,cap in (('load',1),('forward',120),('derivative',32)):
            for _ in range(cap):counts.reserve(kind)
            with self.assertRaises(ValueError):counts.reserve(kind)
        self.assertLess(sum(support.GROUP_CAPS.values())+65536,192*1024**2)
        self.assertEqual(support.GROUP_CAPS['logits'],120*248320*4)
        from setup_budget import Budget
        b=Budget(100.);self.assertEqual(b.absolute_end,1435.);self.assertEqual(b.deadlines(100.,'worker')[0],1300.)
        b.charge('worker',7.);b.charge('audit',8.)
        with self.assertRaises(ValueError):b.charge('extra',.1)
    def test_no_model_imports(self):self.assertFalse(BLOCKED & {n.split('.')[0] for n in sys.modules})
    def test_actual_preparation_adapter_and_bundle(self):
        import input_reader
        adapter=input_reader.adapter()
        base=support.HERE/'fixtures/preparation_bundle'
        inventory=json.loads((support.HERE/'FIXTURE_INVENTORY.json').read_bytes())
        for pin in inventory['files']:
            raw=(base/pin['path']).read_bytes()
            self.assertEqual((len(raw),support.sha(raw)),(pin['bytes'],pin['sha256']))
        text=(base/'TEXT_LOCK.json').read_bytes()
        pins={name:support.sha((base/'preparation'/name).read_bytes()) for name in adapter.artifact_names()}
        release={'text_lock_sha256':support.sha(text),'preparation_files':pins,'inputs_sha256':pins['inputs.json'],
            'preparation_closure_sha256':support.sha((base/'PREPARATION_CLOSURE.json').read_bytes()),'source_freeze_sha256':'f'*64}
        data=json.loads((base/'preparation/inputs.json').read_bytes())
        fake_template=data['cases'][0]['input_binding']['chat_template_sha256']
        original=input_reader.adapter
        def with_fake_template():
            value=original();value.TEMPLATE_SHA256=fake_template;return value
        with patch.object(input_reader,'adapter',with_fake_template):
            self.assertEqual(input_reader.read_bundle(base,release,synthetic=True),data)
            altered=copy.deepcopy(release);altered['preparation_files']['inputs.json']='0'*64
            with self.assertRaises(ValueError):input_reader.read_bundle(base,altered,synthetic=True)
        self.assertEqual(len(input_reader.schema_cases()),16)
        with patch.dict(sys.modules,{'renderer':SimpleNamespace(__file__='WRONG_SOURCE')}):
            with self.assertRaisesRegex(ValueError,'PREPARATION_MODULE_COLLISION'):input_reader.adapter()
        with patch.object(input_reader,'PREPARATION_SOURCE_SHA256','0'*64):
            with self.assertRaisesRegex(ValueError,'PREPARATION_MANIFEST_HASH'):input_reader.adapter()

if __name__=='__main__':unittest.main(verbosity=2)

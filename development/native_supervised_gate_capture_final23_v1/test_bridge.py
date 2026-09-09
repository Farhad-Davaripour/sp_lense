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
def prepared():
    base=support.ROOT/'development/native_supervised_gate_capture_v1/fixtures/preparation_bundle/bundle';release=json.loads((base/'SYNTHETIC_RELEASE.json').read_bytes())
    a=input_reader.adapter();a.TEMPLATE_SHA256=release['fake_template_sha256']
    return a.read_bundle(base,release,synthetic=True)
class FakeReceiver:
    def __init__(self,inputs,mode):
        self.inputs,self.mode=inputs,mode;self.capture=None;self.edit_hook_registrations=0;self.calls=0;self.z=Logits(.2)
    def clear_capture(self):self.capture=None
    def forward_inputs(self,ids,mask,phase,offset):
        i=self.calls;self.calls+=1;assert phase=='baseline' and not offset.any()
        self.capture={'hook_calls':0 if self.mode=='invalid' and i==3 else 1,'final_input_index':len(ids)-1,
            'logit_count':248320,'parameter_versions_unchanged':True,'hook':'blocks.23.hook_out',
            'native_target':'model.language_model.layers.23','nonfinal_positions':len(ids)-1,'unselected_sha256':'a'*64}
        label=1 if self.inputs['cases'][i]['audit_only']['category']=='self_shutdown' else -1
        h=[0.]*1024;h[0]=3.*label;h[i+1]=.125
        if self.mode=='editing':self.edit_hook_registrations+=1
        return self.z,Vec(h)
    def finalize(self):return {'parameter_bytes_unchanged':True,'buffer_bytes_unchanged':True,'parameter_sha256':'b'*64,'buffer_sha256':'c'*64}
def capture(mode='success'):
    inputs=prepared();mem={};frozen=support.json_bytes({'source_sha256':{},'external_sources':[]})
    execution={'scope':'ROOT_APPROVED_NATIVE_FINAL23_CONSTRUCTION_CAPTURE_ONLY','attempt':'native_supervised_gate_capture_final23_attempt_001',
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

class BridgeTests(unittest.TestCase):
    def test_fixed23_workflow_saved_audit_and_wrong_site(self):
        good=capture();audited=judge(good)
        self.assertTrue(audited['scientific_pass'])
        self.assertEqual(audited['training_manifest']['feature_contract']['native_target'],'model.language_model.layers.23')
        self.assertEqual(audited['training_manifest']['namespace'],'development/native_supervised_gate_capture_final23_v1')
        self.assertEqual(good[0]['counts']['attempts'],{'load':1,'forward':32,'derivative':0})
        for field,value in (('hook','blocks.10.hook_out'),('native_target','model.language_model.layers.10'),('final_input_index',0)):
            v=list(good);v[2]=dict(v[2]);mem=v[2];name='rows/'+workflow.keys()[0]+'__baseline.json'
            row=json.loads(mem[name]);row['capture'][field]=value;mem[name]=support.json_bytes(row)
            closed=json.loads(mem['CLOSED_WORKER_BINDING.json']);pin=next(p for p in closed['files'] if p['path']==name)
            pin.update(bytes=len(mem[name]),sha256=support.sha(mem[name]));mem['CLOSED_WORKER_BINDING.json']=support.json_bytes(closed)
            with self.assertRaisesRegex(ValueError,'NATIVE_CAPTURE_CHECKS'):judge(v)
        self.good,self.audited=good,audited

    def test_immutable_input_proof_and_default_deny(self):
        # Executes every original prepared-bundle proof against the tracked artificial fixture.
        data=prepared();self.assertEqual(len(data['cases']),32)
        record=json.loads(input_reader.RECORD.read_bytes());bindings=record['bundle_bindings']
        self.assertFalse(record['approved']);self.assertFalse(bindings['approved'])
        self.assertEqual(bindings['inputs_sha256'],input_reader.INPUT_SHA)
        self.assertEqual(bindings['preparation_files']['RESULT.json'],input_reader.RESULT_SHA)
        self.assertEqual(bindings['preparation_closure_sha256'],input_reader.CLOSURE_SHA)
        release={**bindings,'reused_inputs_sha256':support.sha(input_reader.RECORD.read_bytes())}
        with patch.object(input_reader,'adapter') as a:
            a.return_value.read_bundle.return_value={'synthetic_spy':True}
            self.assertEqual(input_reader.read_bundle(None,release),{'synthetic_spy':True})
            a.return_value.read_bundle.assert_called_once_with(input_reader.ORIGINAL_BASE,bindings)
        release['inputs_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'NEW_RELEASE_REUSE_JOIN'):input_reader.read_bundle(None,release)
        import authority
        with self.assertRaisesRegex(ValueError,'EXPLICIT_ROOT_RELEASE_HASH_REQUIRED'):authority.read_release('')
        self.assertFalse((support.HERE/'root_release').exists())

    def test_caps_and_byte_identical_substrate(self):
        self.assertEqual(sum(support.GROUP_CAPS.values())+65536,37879808)
        self.assertLess(37879808,64*1024**2)
        old=support.ROOT/'development/native_supervised_gate_capture_v1'
        for name in ('loader.py','core.py','counts.py','forward_trace.py','entry.py','owned_production.py',
            'production_run.py','setup_budget.py','windows_job.py','CHECKPOINT.json','OWNED_IDENTITY.json','REUSED_SOURCES.json','launch.py'):
            self.assertEqual((support.HERE/name).read_bytes(),(old/name).read_bytes(),name)
        self.assertFalse({'torch','transformers','tokenizers','safetensors'} & {x.split('.')[0] for x in sys.modules})

    def test_actual_receiver23_prenorm_final_position(self):
        # Fake tensor/module interface only. Production ctor, hook ownership and forward body execute.
        sys.path.insert(0,str(support.ROOT/'.venv/Lib/site-packages'))
        try:import numpy as np
        finally:sys.path.pop(0)
        class Tensor:
            def __init__(self,x,dtype='float32'):self.a=np.array(x,dtype=np.float32 if dtype=='float32' else np.int64);self.dtype=dtype;self.device=SimpleNamespace(type='cpu')
            @property
            def shape(self):return self.a.shape
            def detach(self):return self
            def clone(self):return Tensor(self.a.copy(),self.dtype)
            def __getitem__(self,k):return Tensor(self.a[k],self.dtype)
            def __add__(self,b):return Tensor(self.a+b.a)
            def __sub__(self,b):return Tensor(self.a-b.a)
            def abs(self):return Tensor(abs(self.a))
            def max(self):return self.a.max()
            def isfinite(self):return np.isfinite(self.a)
            def any(self):return self.a.any()
            def contiguous(self):return self
            def numpy(self):return self.a
            def numel(self):return self.a.size
        class Block:
            def __init__(self):self.hooks=[]
            def register_forward_hook(self,fn):
                self.hooks.append(fn);return SimpleNamespace(remove=lambda:self.hooks.remove(fn))
            def __call__(self,value):
                for fn in self.hooks:value=fn(self,(),value)
                return value
        parameter=SimpleNamespace(grad=None,requires_grad=False,_version=0,device=SimpleNamespace(type='cpu'),dtype='float32',is_floating_point=lambda:True)
        class Model:
            def __init__(self):
                self.blocks=[Block() for _ in range(24)];self.model=SimpleNamespace(language_model=SimpleNamespace(layers=self.blocks),rope_deltas=None);self.training=False
            def named_parameters(self,**kw):return [('p',parameter)]
            def named_buffers(self,**kw):return []
            def __call__(self,**kw):
                n=kw['input_ids'].shape[1]
                for i,block in enumerate(self.blocks):
                    a=np.full((1,n,1024),i+1,dtype=np.float32);a[0,-1]=100+i;observed=block(Tensor(a))
                self.postnorm=Tensor(observed.a+1000.)
                return SimpleNamespace(past_key_values=None,logits=Tensor(np.zeros((1,1,248320))))
        @contextmanager
        def no_context(*args,**kwargs):yield
        fake=SimpleNamespace(float32='float32',int64='int64',nn=SimpleNamespace(Module=Model),
            tensor=lambda x,dtype,device:Tensor(x,dtype),equal=lambda a,b:np.array_equal(a.a,b.a),
            inference_mode=no_context,enable_grad=no_context)
        raw=(support.HERE/'receiver.py').read_bytes();tree=ast.parse(raw)
        tree.body=[n for n in tree.body if not (isinstance(n,ast.Import) and any(a.name=='torch' for a in n.names))
            and not (isinstance(n,ast.ImportFrom) and n.module=='core')]
        namespace={'torch':fake,'require':support.require,'sha':support.sha,'span':no_context,'parameter_digest':lambda p:'f'*64}
        exec(compile(tree,'actual_receiver23_fake_interface','exec'),namespace)
        namespace.update(pristine_registry=lambda m:None,parameter_metadata=lambda p:(),
            registry=lambda m:tuple(len(b.hooks) for b in m.blocks))
        model=Model();guard=SimpleNamespace(latch=SimpleNamespace(admit=lambda:None,stop=lambda x:None),permit=no_context)
        receiver=namespace['NativeReceiver'](model,guard)
        self.assertIs(receiver.block,model.blocks[23]);self.assertIsNot(receiver.block,model.blocks[10])
        z,h=receiver.forward_inputs([1,2,3],[1,1,1],'baseline',Tensor(np.zeros(1024)))
        self.assertTrue(np.all(h.a==123.));self.assertTrue(np.all(model.postnorm.a[0,-1]==1123.))
        self.assertNotEqual(float(h.a[0]),111.);self.assertNotEqual(float(h.a[0]),24.)
        self.assertEqual(receiver.capture['final_input_index'],2);self.assertEqual(receiver.capture['hook'],'blocks.23.hook_out')
        self.assertTrue(all(not b.hooks for b in model.blocks));self.assertEqual(receiver.edit_hook_registrations,0)

if __name__=='__main__':unittest.main(verbosity=2)

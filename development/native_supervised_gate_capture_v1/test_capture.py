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
    base=support.HERE/'fixtures/preparation_bundle/bundle';release=json.loads((base/'SYNTHETIC_RELEASE.json').read_bytes())
    a=input_reader.adapter();a.TEMPLATE_SHA256=release['fake_template_sha256']
    return a.read_bundle(base,release,synthetic=True)
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
def capture(mode='success'):
    inputs=prepared();mem={};frozen=support.json_bytes({'source_sha256':{},'external_sources':[]})
    execution={'scope':'ROOT_APPROVED_NATIVE_CONSTRUCTION_CAPTURE_ONLY','attempt':'native_supervised_gate_capture_attempt_001',
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
class CaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.good=capture();cls.audited=judge(cls.good)
    def test_prepared_capture_and_independent_audit(self):
        r=self.good[0];self.assertTrue(r['scientific_pass']);self.assertTrue(self.audited['scientific_pass'])
        self.assertEqual(r['counts']['attempts'],{'load':1,'forward':32,'derivative':0})
        selected=self.audited['training_manifest']['selection'];self.assertEqual(len(selected),32)
        self.assertEqual(sum(s['label']==1 for s in selected),8)
    def test_technical_stop_and_no_intervention(self):
        for mode,n in (('invalid',3),('editing',0)):
            v=capture(mode);r=v[0];self.assertFalse(r['scientific_pass'])
            self.assertEqual([s['status'] for s in r['cells']],['COMPLETE']*n+['FAILED']+['UNRUN']*(31-n))
            self.assertFalse(judge(v)['scientific_pass'])
    def test_coherent_tampering_rejected(self):
        v=list(self.good);v[2]=dict(v[2]);mem=v[2];path='rows/'+workflow.keys()[0]+'__baseline.json'
        row=json.loads(mem[path]);row['offset'][0]=.1;mem[path]=support.json_bytes(row)
        binding=json.loads(mem['CLOSED_WORKER_BINDING.json']);pin=next(p for p in binding['files'] if p['path']==path)
        pin.update(bytes=len(mem[path]),sha256=support.sha(mem[path]));mem['CLOSED_WORKER_BINDING.json']=support.json_bytes(binding)
        with self.assertRaisesRegex(ValueError,'UNEDITED_NATIVE_FLOAT32'):judge(v)
    def test32_capture_to_manifest_to_one_fit(self):
        sys.path.insert(0,str(support.ROOT/'development/native_supervised_gate_v2'))
        try:import gate,source_auth,construction
        finally:sys.path.pop(0)
        r,inputs,mem,execution,frozen=self.good
        store={'SOURCE_FREEZE.json':frozen};prefix='real_evidence/native_supervised_gate_capture_attempt_001/'
        store.update({prefix+k:v for k,v in mem.items()})
        store[prefix+'AUDIT_RESULT.json']=support.json_bytes(self.audited)
        store[prefix+'PARENT_FINAL.json']=support.json_bytes({'execution':execution,'audit_completed':True,'scientific_pass':True,
            'worker_quiescent':True,'audit_quiescent':True,'errors':[],'classification':'COMPLETE_NATIVE_CONSTRUCTION_CAPTURE'})
        with patch.object(source_auth,'CAPTURE',MemoryPath(store)):
            with self.assertRaises((ValueError,KeyError,FileNotFoundError)):
                source_auth.build_manifest(deadline=time.monotonic()+30)
        with patch.object(source_auth,'CAPTURE',MemoryPath(store)),patch.object(source_auth,'authenticate_capture',return_value=inputs):
            manifest=source_auth.build_manifest(deadline=time.monotonic()+30)
            rows,labels=source_auth.extract_features(manifest,deadline=time.monotonic()+30)
            self.assertEqual((len(rows),labels.count(1),labels.count(-1)),(32,8,24))
            bad=copy.deepcopy(manifest);bad['namespace']='development/native_supervised_gate_evaluation_v2'
            with self.assertRaisesRegex(ValueError,'MANIFEST_REAUTHENTICATION'):source_auth.extract_features(bad,deadline=time.monotonic()+30)
            with tempfile.TemporaryDirectory(dir=support.HERE,prefix='synthetic_fit_') as directory:
                folder=Path(directory)
                for name,data in (('TRAINING_MANIFEST.json',manifest),('CORE_SOURCE_LOCK.json',{}),('CONSTRUCTION_LOCK_DRAFT.json',{}),('release.json',{'approved':False})):
                    (folder/name).write_bytes(gate.canonical(data))
                with patch.object(construction,'HERE',folder),patch.object(construction,'authorize',return_value={'construction_lock_sha256':'e'*64}),\
                     patch.object(construction,'fit',wraps=gate.fit) as fit:
                    outcome=construction.construct(folder/'release.json','f'*64)
                    self.assertEqual(fit.call_count,1);self.assertEqual(outcome['check']['correct'],32)
                    self.assertTrue(outcome['scientific_pass']);self.assertEqual(outcome['check']['tolerance'],1e-10)
                    with self.assertRaises(FileExistsError):construction.construct(folder/'release.json','f'*64)
                    self.assertEqual(fit.call_count,1)
    def test_caps_and_exact_reuse(self):
        counter=Counts(time.monotonic()+30,lambda *a:None)
        counter.reserve('load')
        for _ in range(32):counter.reserve('forward')
        for kind in ('load','forward','derivative'):
            with self.assertRaises(ValueError):counter.reserve(kind)
        self.assertEqual(sum(support.GROUP_CAPS.values())+65536,37879808)
        self.assertLess(37879808,64*1024**2)
        old=support.ROOT/'development/native_supervised_gate_evaluation_v2'
        for name in ('loader.py','receiver.py','core.py','forward_trace.py','entry.py','owned_production.py','setup_budget.py','CHECKPOINT.json'):
            self.assertEqual((support.HERE/name).read_bytes(),(old/name).read_bytes())
        self.assertFalse({'torch','transformers','tokenizers','numpy'} & {x.split('.')[0] for x in sys.modules})
if __name__=='__main__':unittest.main(verbosity=2)

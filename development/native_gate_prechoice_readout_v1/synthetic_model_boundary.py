"""TEST ONLY replacement loader at the explicitly authorized model boundary.

No real provider/model/tensors are imported. All lifecycle/authority/dispatch
modules in the owned mirror remain the production function bodies.
"""
import ast,json,math,struct,sys,types
from array import array
from support import ROOT,HERE,require,sha,output,write_new
fake=ROOT/'development/native_supervised_gate_evaluation_v1/test_candidate.py'
raw=fake.read_bytes();require(sha(raw)=='ef9975fe8680a32ab7800d10b7eb872486bdd30df8643a026a6fca4528ad1a56','PINNED_FAKE_ARRAYS')
tree=ast.parse(raw)
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ('f32','Vec','Difference','Logits')],type_ignores=[]),'pinned_fake_arrays','exec'))
torch=types.ModuleType('torch');torch.float32='float32';torch.zeros=lambda n,**k:Vec([0.]*n)
def no_gradient(*a,**k):raise AssertionError('SYNTHETIC_DERIVATIVE_FORBIDDEN')
torch.autograd=types.SimpleNamespace(grad=no_gradient)
require('torch' not in sys.modules,'NO_REAL_TORCH');sys.modules['torch']=torch
def block_unused_dependencies():
    require(not {'transformers','tokenizers','safetensors','numpy'}&set(sys.modules),'NO_PROVIDERS')
class Model:
    def __init__(self,cases):
        self.rows={}
        scenarios=list(dict.fromkeys(c['case_key'].split('__')[0] for c in cases))
        for c in cases:
            key=c['case_key'].split('__')[0];i=scenarios.index(key);category=c['audit_only']['category']
            v=[2.,2.,2.]
            if category!='self_shutdown':v[{'other_shutdown':0,'non_termination_control':1,'ordinary':2}[category]]=-2.
            # Invertible non-axis mixing, plus independent small scenario jitter.
            row=[v[0]+.25*v[1],v[1]-.125*v[2],v[2]+.0625*v[0]]+[0.]*1021
            row[4+i]=.03125
            self.rows[tuple(c['input']['input_ids'])]=row
    def forward(self,ids):return Logits(.2),Vec(self.rows[tuple(ids)])
class Receiver:
    def __init__(self,model,guard):self.model=model;self.guard=guard;self.capture=None;self.edit_hook_registrations=0
    def clear_capture(self):self.capture=None
    def forward_inputs(self,ids,mask,phase,offset,*,readout_selector):
        from index_contract import validate,canon,int_hash,CONTRACT
        validate(ids,readout_selector);require(phase=='baseline' and not offset.any(),'SYNTHETIC_BASELINE_ONLY')
        with self.guard.permit('forward'):z,h=self.model.forward(ids)
        self.capture={'hook_calls':1,'final_input_index':len(ids)-1,'logit_count':248320,'parameter_versions_unchanged':True,
            'hook':'blocks.10.hook_out','native_target':'model.language_model.layers.10','nonfinal_positions':len(ids)-1,'unselected_sha256':'a'*64,
            'readout_index':readout_selector['readout_index'],'readout_selector_sha256':sha(canon(readout_selector)),
            'readout_input_ids_sha256':int_hash(ids),'readout_nonselected_positions':len(ids)-1,
            'readout_nonselected_sha256':'b'*64,'all_positions_unchanged':True,'feature_contract':CONTRACT}
        return z,h
    def finalize(self):return {'parameter_bytes_unchanged':True,'buffer_bytes_unchanged':True,
        'parameter_sha256':'2eeb7434977939f80711e1fa83a1d587c71887da0c06f1edc13a43d5ec1192e6',
        'buffer_sha256':'18f0caa55461f0f07afba9cb9809f1830745b445e5cc62283d505a0017f892fc'}
def load(counts,latch,execution,deadline):
    from authority import authenticate
    require(authenticate()['execution']==execution,'SYNTHETIC_OWNED_AUTHORITY')
    from input_reader import read
    cases=read()['cases'];from core import ForwardDerivativeGuard
    guard=ForwardDerivativeGuard(Model,counts,latch,deadline);guard.install();counts.reserve('load')
    receiver=Receiver(Model(cases),guard);state=receiver.finalize()
    write_new('LOADER_READY.json',{'execution':execution,'native_initial_sha256':state['parameter_sha256'],
        'native_initial_buffer_sha256':state['buffer_sha256'],'loading_info':{},'declared_class':'Qwen3_5ForConditionalGeneration',
        'coverage':{'complete_key_shape_coverage':True,'native_unique_parameters':473,'named_occurrences':474},
        'old_digest_equivalence_claimed':False,'test_only_artificial_model_boundary':True})
    return receiver,torch,guard

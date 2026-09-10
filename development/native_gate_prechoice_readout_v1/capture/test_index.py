"""Focused artificial selector/observer tests. No real model or tokenizer imports.

Observer integration substitutes only the inherited native forward/context with
an explicitly artificial driver; unchanged native execution is NOT rerun here.
"""
import copy,sys,types,unittest
from contextlib import contextmanager,nullcontext
from unittest.mock import patch
import numpy as np
import index_contract as c

class Tensor:
    def __init__(self,a):self.a=np.asarray(a,dtype=np.float32)
    @property
    def shape(self):return self.a.shape
    def detach(self):return self
    def clone(self):return Tensor(self.a.copy())
    def contiguous(self):return self
    def numpy(self):return self.a
    def any(self):return self.a.any()
    def __getitem__(self,item):return Tensor(self.a[item])

torch=types.ModuleType('torch');torch.equal=lambda a,b:np.array_equal(a.a,b.a)
torch.cat=lambda values,dim:Tensor(np.concatenate([v.a for v in values],axis=dim))
core=types.ModuleType('core');core.parameter_digest=lambda p:'synthetic_unused';core.require=c.require;core.sha=c.sha;core.span=lambda *a:nullcontext()
with patch.dict(sys.modules,{'torch':torch,'core':core}):import receiver as r

IDS=[248045,99,198,50057,8,41,198,48964,8,42,198,248045,74455,198,248068,271,248069,271]
W={'text_and_id_swap_witness':True,'view_keys':['left','right'],'readout_index':2,'shared_prefix_length':3,
   'final_input_indices':[17,17],'last_shared_id':198,'first_divergent_label_ids':[50057,48964],
   'shared_prefix_ids_sha256':c.sha(c.canon(IDS[:3]))}
def selector():return c.bind(IDS,W,'left','a'*64,expected_input_ids_sha256=c.int_hash(IDS))
@contextmanager
def context(self,callback):
    self.fake_callback=callback
    try:yield
    finally:self.fake_callback=None
def forward(self,ids,mask,phase,offset):
    # The synthetic activation is causal: rowj depends only on prefixIDs[:j+1].
    arr=np.empty((1,len(ids),1024),dtype=np.float32)
    for j in range(len(ids)):arr[0,j]=np.arange(1024,dtype=np.float32)*.125+sum(ids[:j+1])*.25+j
    activation=Tensor(arr)
    with self.hook_context(lambda a,h:a):changed=self.fake_callback(activation,None)
    assert changed is activation
    self.capture={'hook':'blocks.10.hook_out','native_target':'model.language_model.layers.10','hook_calls':1,
        'final_input_index':len(ids)-1,'nonfinal_positions':len(ids)-1,'logit_count':248320,'parameter_versions_unchanged':True}
    return Tensor([sum(ids)]),changed[0,-1]

class Tests(unittest.TestCase):
    def test_selector_and_retained_id_provenance(self):
        s=selector();self.assertEqual(c.validate(IDS,s),2)
        bad=list(IDS);bad[5]+=1
        with self.assertRaisesRegex(ValueError,'RETAINED_FULL_INPUT_ID_PIN'):c.bind(bad,W,'left','a'*64,expected_input_ids_sha256=c.int_hash(IDS))
        with self.assertRaisesRegex(ValueError,'CERTIFIED_EXACT_VIEW'):c.bind(IDS,W,'wrong','a'*64,expected_input_ids_sha256=c.int_hash(IDS))
        with self.assertRaisesRegex(ValueError,'EXACT_SELECTOR_INPUT_JOIN'):c.validate(bad,s)
    def test_bounds_wrong_prefix_and_final_position(self):
        for value in (-1,0,3,len(IDS)-1,True):
            s=selector();s['readout_index']=value
            with self.assertRaises(ValueError):c.validate(IDS,s)
        w=copy.deepcopy(W);w['shared_prefix_ids_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'EXACT_SHARED_PREFIX_IDS'):c.bind(IDS,w,'left','a'*64,expected_input_ids_sha256=c.int_hash(IDS))
    def test_indexed_observer_causal_invariance_and_cleanup(self):
        obj=object.__new__(r.NativeReceiver);obj.editing=False
        with patch.object(r.BASE.NativeReceiver,'forward_inputs',forward),patch.object(r.BASE.NativeReceiver,'hook_context',context):
            z,h=obj.forward_inputs(IDS,[1]*len(IDS),'baseline',Tensor([0]*1024),readout_selector=selector())
            self.assertTrue(c.validate_capture(IDS,selector(),obj.capture));first=h.a.copy()
            self.assertTrue(np.array_equal(first,np.arange(1024,dtype=np.float32)*.125+sum(IDS[:3])*.25+2))
            self.assertFalse(np.array_equal(first,np.arange(1024,dtype=np.float32)*.125+sum(IDS[:2])*.25+1))
            self.assertFalse(np.array_equal(first,np.arange(1024)*.125+sum(IDS)*.25+len(IDS)-1))
            self.assertIsNone(obj._prechoice_active);self.assertIsNone(obj._prechoice_vector);self.assertIsNone(obj._prechoice_other_sha)
            changed=list(IDS);changed[5]+=7
            s=c.bind(changed,W,'left','a'*64,expected_input_ids_sha256=c.int_hash(changed))
            z2,h2=obj.forward_inputs(changed,[1]*len(changed),'baseline',Tensor([0]*1024),readout_selector=s)
            self.assertTrue(np.array_equal(first,h2.a));self.assertFalse(np.array_equal(z.a,z2.a))
            for name,value in (('native_target','model.language_model.layers.23'),('readout_index',3),('final_input_index',2),('readout_selector_sha256','0'*64),('all_positions_unchanged',False)):
                bad=dict(obj.capture);bad[name]=value
                with self.assertRaises(ValueError):c.validate_capture(changed,s,bad)
            bad=dict(obj.capture);bad.pop('readout_nonselected_sha256')
            with self.assertRaises(KeyError):c.validate_capture(changed,s,bad)
            bad=dict(obj.capture);bad['readout_nonselected_sha256']='INVALID'
            with self.assertRaisesRegex(ValueError,'READOUT_NONSELECTED_HASH'):c.validate_capture(changed,s,bad)
    def test_no_edit_gradient_or_implicit_index(self):
        obj=object.__new__(r.NativeReceiver);obj.editing=False
        for phase,offset in (('gradient_1',[0]*1024),('baseline',[1]*1024)):
            with self.assertRaisesRegex(ValueError,'READONLY_PRECHOICE_BASELINE'):obj.forward_inputs(IDS,[1]*len(IDS),phase,Tensor(offset),readout_selector=selector())
        with self.assertRaises(TypeError):obj.forward_inputs(IDS,[1]*len(IDS),'baseline',Tensor([0]*1024))
        for method in ('clean','parameter_state','finalize','clear_capture','start_request','finish_request'):
            self.assertIs(getattr(r.NativeReceiver,method),getattr(r.BASE.NativeReceiver,method))
        self.assertNotIn('transformers',sys.modules);self.assertNotIn('tokenizers',sys.modules)

if __name__=='__main__':unittest.main(verbosity=2)

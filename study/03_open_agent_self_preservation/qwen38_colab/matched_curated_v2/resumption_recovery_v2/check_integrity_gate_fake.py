"""Fake byte/name fixture only; no torch/model/GPU import or execution."""
from pathlib import Path
import importlib.util,json,sys,types
sys.modules['torch']=types.SimpleNamespace(uint8='fake_uint8')
p=Path('C:/Users/farha/Research3/tmp/integrity_gate_v2.py')
spec=importlib.util.spec_from_file_location('gate_fixture',p);gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
class Parameter:
    def __init__(self,data):self.raw=bytearray(data);self.requires_grad=False;self.data=self;self.shape=(len(data),);self.dtype='uint8'
    def detach(self):return self
    def cpu(self):return self
    def contiguous(self):return self
    def view(self,dtype):assert dtype=='fake_uint8';return self
    def numpy(self):return self.raw
class Model:
    def __init__(self,values,unloaded=None):self.values=values;self.unloaded=unloaded
    def named_parameters(self):return iter(self.values)
    def unload(self):return self.unloaded
raw=Model([('model.layer0.weight',Parameter(b'abc')),('model.layer1.weight',Parameter(b'def'))])
wrapped=Model([('base_model.model.model.layer0.base_layer.weight',raw.values[0][1]),
    ('base_model.model.model.layer0.lora_A.weight',Parameter(b'adapter')),
    ('base_model.model.model.layer1.base_layer.weight',raw.values[1][1])],raw)
pin=gate.fingerprint(wrapped)['historical_wrapped_name_hash'];saved=[]
base,receipt=gate.verify_and_unload(wrapped,pin,Path('unused'),lambda path,obj:saved.append(json.loads(json.dumps(obj))))
assert receipt['completed'] and len(saved)==2 and saved[0]['completed'] is False
bad=Model([('model.layer0.weight',Parameter(b'abX')),('model.layer1.weight',Parameter(b'def'))])
wrapped.unloaded=bad
try:gate.verify_and_unload(wrapped,pin,Path('unused'),lambda path,obj:None);raise AssertionError('Changed bytes accepted')
except RuntimeError:pass
wrapped.unloaded=raw
try:gate.verify_and_unload(wrapped,'wrong_pin',Path('unused'),lambda path,obj:None);raise AssertionError('Wrong historical pin accepted')
except RuntimeError:pass
receipt={'passed':True,'fake_names_and_identical_bytes_accept':True,'changed_packed_bytes_rejected':True,
    'wrong_historical_pin_rejected':True,'LoRA_parameters_excluded':True,'actual_torch_model_gpu_calls':0,
    'historical_PIN_redefined':False}
Path('C:/Users/farha/Research3/tmp/INTEGRITY_GATE_FAKE_CHECK.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))

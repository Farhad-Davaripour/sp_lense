"""CPU BF16 checks of causal injection matching; no model inference."""
import json
from pathlib import Path
import numpy as np
import torch
from panel_collect import native_intervention

specification=json.loads(Path(__file__).with_name('PANEL.json').read_text())
with np.load('C:/Users/farha/Research4/artifacts/propagation_followup_v1/panel_inputs/panel_vectors.npz') as z:
    vectors={k:torch.tensor(z[k]) for k in z.files}
generator=torch.Generator(device='cpu').manual_seed(20261013)
h=(torch.randn((1,16,1024),generator=generator)*.125).to(torch.bfloat16)
saved=h.clone()
results={}
for condition in specification['conditions']:
    changed,audit=native_intervention(h,condition,vectors,specification)
    assert torch.equal(h,saved), 'Control mutated its input'
    assert changed.dtype==torch.bfloat16 and torch.isfinite(changed).all()
    if audit is not None:results[condition]=audit.tolist()
assert max(results[c][2] for c in ['plus_v','minus_v','random1','random2'])<=.005
assert results['norm_only'][3]<=.005 and results['norm_only'][4]<=.005
assert torch.equal(native_intervention(h,'base',vectors,specification)[0],h)
# Changing an unrelated future position cannot change the current intervention.
altered=h.clone();altered[:,8:]*=2
for condition in specification['conditions']:
    first,_=native_intervention(h,condition,vectors,specification)
    second,_=native_intervention(altered,condition,vectors,specification)
    assert torch.equal(first[:,:8],second[:,:8])
print(json.dumps({'CPU_BF16_control_checks':'passed','future_position_independence':'passed',
                  'audit_columns':['target_last','actual_last','max_relative_matching_error','max_norm_error_to_plus','max_site_angle','min_coefficient','max_coefficient'],
                  'controls':results,'weights_loaded':False,'GPU_allocated':False}))

"""Summarize independently audited unchanged readout vectors."""
from pathlib import Path
import json,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BEGIN=time.monotonic();B=Path('C:/Users/farha/Research4/artifacts/linear_and_output_readout_v1');R=B/'replayed_20261015'
def read(p):return json.loads(p.read_text())
f=read(R/'FIELD_RESULTS.json');d=read(R/'DERIVATIVE_RESULTS.json');a=read(B/'recovery_20261015/INDEPENDENT_READOUT_AUDIT_20261015.json')
assert a['all_original_scientific_field_gates_pass'] and all(r['pass'] for r in f+d)
summary={'original_terminal_status':'failed bookkeeping assertion; retained unchanged','scientific_status':'all original gates independently reconfirmed; no rerun or threshold change','independent_audit':a,'module_statistics':{}}
for module in ['linear_attention_readout','convolution_update','hidden','logits']:
    summary['module_statistics'][module]={}
    for label,rows in [('primal',f),('response',d)]:
        data=[r for r in rows if r['module']==module];nz=[r for r in data if r['native_norm']>1e-12]
        summary['module_statistics'][module][label]={'cases':len(data),'nonzero':len(nz),'maximum_relative':max(r['relative'] for r in nz),'median_relative':float(np.median([r['relative'] for r in nz])),'maximum_error_over_bound':max(r['error_norm']/r['bound'] for r in data)}
prev=Path('C:/Users/farha/Research4/artifacts')
e18=read(prev/'recurrent_energy_replication_v1/replayed_20261015/RESULTS.json')
e19=read(prev/'attention_mlp_mechanism_v1/COMPONENT_INTERPRETATION_20261015.json')
values=[e18['max_primal_field_error']/1e-6,summary['module_statistics']['linear_attention_readout']['primal']['maximum_error_over_bound'],e19['module_statistics']['attention']['primal']['max_error_over_declared_bound'],e19['module_statistics']['MLP']['primal']['max_error_over_declared_bound'],summary['module_statistics']['logits']['primal']['maximum_error_over_bound']]
labels=['Memory\nupdate','Memory\nreadout','Full\nattention','MLP','All word\nscores']
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(9,4.5),constrained_layout=True)
ax.bar(labels,values,color=['#3d6c94','#447caa','#5a8c76','#8b789e','#b28249'])
ax.set(ylim=(0,1.15),ylabel='Largest error / its declared allowed error',title='Every component replay stayed within its frozen error limit\nSix questions; separate from prospective nudge forecasts')
ax.axhline(1.,color='#a84949',linestyle='--',label='Declared limit')
ax.legend(loc='upper right')
ax.grid(axis='y',alpha=.2)
for i,v in enumerate(values):ax.annotate(f'{v:.2g}',(i,v),xytext=(0,5),textcoords='offset points',ha='center',fontsize=10)
fig.savefig(B/'COMPLETE_COMPONENT_REPLAY.png',dpi=160);plt.close(fig)
summary['analysis_seconds']=time.monotonic()-BEGIN
(B/'VERIFIED_READOUT_INTERPRETATION_20261015.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))

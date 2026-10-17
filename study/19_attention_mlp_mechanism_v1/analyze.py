"""Summarize verified component replay and create an inspected research figure."""
from pathlib import Path
import json,hashlib,zipfile,shutil,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BEGIN=time.monotonic();B=Path('C:/Users/farha/Research4/artifacts/attention_mlp_mechanism_v1');R=B/'replayed_20261015'
def read(p):return json.loads(p.read_text())
zpath=Path('C:/Users/farha/Downloads/R4_INDEPENDENT_COMPONENT_AUDIT_20261015.zip')
assert hashlib.sha256(zpath.read_bytes()).hexdigest()=='1967144ec7b334c4dd84ba1af7eb10262a6da4b6eb4c62e5c08fa77c3ee0f286'
with zipfile.ZipFile(zpath) as z:
    assert z.testzip() is None
    for m in z.infolist():assert (B/m.filename).resolve().is_relative_to(B.resolve())
    z.extractall(B)
shutil.copyfile(zpath,B/zpath.name)
f=read(R/'FIELD_RESULTS.json');d=read(R/'DERIVATIVE_RESULTS.json');e=read(R/'ATTENTION_ENERGY_HEADS.json')
assert all(r['pass'] for r in f+d+e)
audit=read(B/'INDEPENDENT_AUDIT_20261015.json');assert audit['all_frozen_field_gates_reconfirmed'] and audit['all_probability_FD_checks_pass']
summary={'results':read(R/'RESULTS.json'),'independent_audit':audit,'primal_fields':len(f),'derivative_fields':len(d),'module_statistics':{}}
for kind in ['MLP','attention','residual_middle','residual_output']:
    summary['module_statistics'][kind]={}
    for label,rows in [('primal',f),('response',d)]:
        a=[r for r in rows if r['module']==kind];nz=[r for r in a if r['native_norm']>1e-12]
        summary['module_statistics'][kind][label]={'cases':len(a),'nonzero':len(nz),'relative_median':float(np.median([r['relative'] for r in nz])),'relative_max':max(r['relative'] for r in nz),'max_error_over_declared_bound':max(r['error_norm']/r['bound'] for r in a)}
summary['maximum_attention_stationarity']=max(r['stationarity'] for r in e)
summary['maximum_attention_energy_gap_identity_error']=max(r['objective_gap_identity_error'] for r in e)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(1,2,figsize=(11,4.9),constrained_layout=True)
for j,(rows,title) in enumerate([(f,'Original component outputs'),(d,'Responses to the known nudge')]):
    for kind,label in [('MLP','MLP'),('attention','Full attention')]:
        layers=sorted({r['layer'] for r in rows if r['module']==kind})
        points=[max([r['relative'] for r in rows if r['module']==kind and r['layer']==l and r['native_norm']>1e-12],default=np.nan) for l in layers]
        ax[j].plot(layers,points,'o-',label=label,markersize=4)
    ax[j].set(title=title,xlabel='Model layer (zero based)',ylabel='Largest relative full-vector discrepancy',yscale='log')
    ax[j].grid(alpha=.2);ax[j].legend()
fig.suptitle('Independent equations reproduce the saved native computation\nSix questions; nonzero cases shown; no new whole-model inference')
fig.savefig(B/'COMPONENT_EQUATION_VALIDATION.png',dpi=160);plt.close(fig)
summary['analysis_seconds']=time.monotonic()-BEGIN
(B/'COMPONENT_INTERPRETATION_20261015.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))

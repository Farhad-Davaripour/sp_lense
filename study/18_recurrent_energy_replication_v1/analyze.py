"""Summarize native energy replication without changing its frozen gates."""
from pathlib import Path
import hashlib, json, time, zipfile, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BEGIN=time.monotonic()
B=Path('C:/Users/farha/Research4/artifacts/recurrent_energy_replication_v1')
R=B/'replayed_20261015'
def read(p): return json.loads(p.read_text())
manifest=read(R/'EXPORT_HASHES.json')
for n,sha in manifest.items():
    assert hashlib.sha256((R/n).read_bytes()).hexdigest()==sha,n
audit_zip=Path('C:/Users/farha/Downloads/R4_INDEPENDENT_ENERGY_AUDIT_20261015.zip')
assert hashlib.sha256(audit_zip.read_bytes()).hexdigest()=='bafcdf9291cda1aab1f7f3a4ac257faddfe482b28e334f5007e452f9b7eeb625'
with zipfile.ZipFile(audit_zip) as z:
    assert z.testzip() is None
    for m in z.infolist(): assert (B/m.filename).resolve().is_relative_to(B.resolve())
    z.extractall(B)
for p in [audit_zip,Path('C:/Users/farha/Downloads/INDEPENDENT_AUDIT_EXPORT_20261015.json')]:
    dst=B/p.name
    if dst.exists(): assert dst.read_bytes()==p.read_bytes()
    else: shutil.copyfile(p,dst)
audit=read(B/'INDEPENDENT_ENERGY_AUDIT_20261015.json')
assert audit['all_original_primary_flags_confirmed'] and audit['all_analytic_response_heads_pass']
heads=read(R/'HEAD_RESULTS.json');derivs=read(R/'DERIVATIVE_RESULTS.json');moving=read(R/'MOVING_LANDSCAPE.json')
results=read(R/'RESULTS.json');layers=sorted({r['layer'] for r in heads})
def percentiles(key,rows=heads):
    x=np.array([r[key] for r in rows],dtype=float)
    return dict(zip(['min','p25','median','p75','max'],map(float,np.quantile(x,[0,.25,.5,.75,1]))))
summary={
    'native_results':results,'independent_audit':audit,'verified_manifest_members':len(manifest),
    'head_update_cases_not_independent_prompts':len(heads),
    'unique_main_questions':len({r['question'] for r in heads}),
    'main_question_ids':sorted({r['question'] for r in heads}),
    'actual_updated_fields':len(derivs),'nonzero_native_response_fields':sum(r['native_q_norm']>1e-12 for r in derivs),
    'native_response_heads':sum(len(r['head_rows']) for r in derivs),
    'nonzero_native_response_heads':sum(h['native_q_norm']>1e-12 for r in derivs for h in r['head_rows']),
    'beta':percentiles('beta'),'gamma':percentiles('gamma'),
    'stiffness_along_key':percentiles('stiffness_along_key'),
    'stiffness_orthogonal':percentiles('stiffness_orthogonal'),
    'regression_loss_after_over_before':dict(zip(['min','median','max'],map(float,np.quantile([r['regression_loss_after']/r['regression_loss_before'] for r in heads],[0,.5,1])))),
    'negative_external_work_cases':sum(r['external_work']<0 for r in heads),
    'negative_quadratic_storage_change_cases':sum(r['internal_quadratic_energy_change']<0 for r in heads),
    'minimum_dissipation':min(r['discrete_dissipation'] for r in heads),
    'moving_landscape_positive_total_change':sum(r['total_potential_change']>0 for r in moving),
    'moving_landscape_cases':len(moving),
    'scope':'Post-collected native-state replay; six questions, two token positions, 18 linear-attention layers, 16 heads; no new full-model forwards or global-energy/steering-detector claim.'
}
normal=[r for r in heads if r['label']=='normal']
retentions=[]
for mode in ['orthogonal','along_key']:
    retentions.append(np.array([[np.median([1-r['stiffness_'+mode] for r in normal if r['layer']==l and r['head']==h]) for h in range(16)] for l in layers]))
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(1,2,figsize=(11,7),constrained_layout=True)
for j,title in enumerate(['Retention away from the current key','Retention along the current key']):
    im=ax[j].imshow(retentions[j],vmin=0,vmax=1,cmap='viridis',aspect='auto')
    ax[j].set(title=title,xlabel='Memory head',ylabel='Model layer (zero based)',yticks=np.arange(len(layers)),yticklabels=layers,xticks=[0,3,7,11,15])
fig.colorbar(im,ax=ax,label='Old-memory retention per update')
fig.suptitle('Different memory directions have different retention\nMedians over six normal questions and two token positions')
fig.savefig(B/'MEMORY_DIRECTIONAL_RETENTION.png',dpi=160)
plt.close(fig)
fig,ax=plt.subplots(1,2,figsize=(11,4.7),constrained_layout=True)
names=['transported_history','new_input_force','stiffness_change']
labels=['Transported prior memory','Changed incoming drive','Changed internal resistance']
for ti,t in enumerate([1,2]):
    for key,label in zip(names,labels):
        vals=[]
        for l in layers:
            rows=[r for r in derivs if r['label']=='primary' and r['token']==t and r['layer']==l and r['native_q_norm']>1e-12]
            vals.append(np.median([r['term_norms'][key]/r['native_q_norm'] for r in rows]) if rows else np.nan)
        ax[ti].plot(layers,vals,'o-',label=label,markersize=4)
    ax[ti].set(title=f'Processed token {t}',xlabel='Model layer (zero based)',ylabel='Component norm / total response norm')
    ax[ti].set_yscale('symlog',linthresh=.01)
    ax[ti].grid(alpha=.2)
ax[1].legend(fontsize=9)
fig.suptitle('How a known nudge changes the memory\nVector components can align or cancel; these are not additive fractions')
fig.savefig(B/'MEMORY_RESPONSE_MECHANISMS.png',dpi=160)
plt.close(fig)
summary['analysis_seconds']=time.monotonic()-BEGIN
(B/'ENERGY_INTERPRETATION_20261015.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))

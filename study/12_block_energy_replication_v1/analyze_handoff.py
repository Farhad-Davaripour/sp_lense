"""Verify matched hand-offs and explain failures without relabeling primary."""
import hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

BASE=Path('C:/Users/farha/Research4/artifacts/block_energy_replication_v1')
ROOT=BASE/'handoff_diagnostic_20261013';OUT=BASE/'evaluated_20261013'
assert json.loads((ROOT/'STATUS.json').read_text())['status']=='complete'
assert json.loads((ROOT/'MECHANICS.json').read_text())['pass']
primary=json.loads((OUT/'RESULTS.json').read_text())
assert primary['primary_support'] is False
rows=json.loads((ROOT/'RESULTS.json').read_text())
assert len(rows)==8

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for part in iter(lambda:f.read(8*1024**2),b''):h.update(part)
    return h.hexdigest()

manifest=json.loads((ROOT/'EXPORT_HASHES.json').read_text())
assert all(sha(ROOT/name)==expected for name,expected in manifest.items())

def logz(x):
    x=np.asarray(x,dtype=np.float64);m=x.max();return m+np.log(np.exp(x-m).sum())

def independent_kl(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    return max(0.,float(logz(b)-logz(a)-np.exp(a-logz(a))@(b-a)))

maximum=0.;case_checks=0
for row in rows:
    suffix='plus' if row['alpha']>0 else 'minus'
    path=ROOT/'actual'/f'{row["question"]}_d{row["direction"]}_{suffix}.npz'
    with np.load(path,allow_pickle=False) as z:
        for arm,info in row['arms'].items():
            for key,a,b in [('KL_base_to_handoff',z['base_logits'],z[arm+'_logits']),
                            ('KL_actual_to_handoff',z['truth_logits'],z[arm+'_logits'])]:
                maximum=max(maximum,abs(independent_kl(a,b)-info[key]))
            case_checks+=1
assert maximum<1e-10 and case_checks==32

def stats(values):
    v=np.asarray(values,dtype=np.float64)
    return {'n':len(v),'minimum':float(v.min()),'median':float(np.median(v)),'maximum':float(v.max())}

summary={}
summary['quadratic_Fisher']={'cost_error':stats([abs(r['original_Fisher_cost']-r['original_KL'])/r['original_KL'] for r in rows])}
summary['linear_logit_endpoint']={'cost_error':stats([abs(r['linear_logit_endpoint_KL']-r['original_KL'])/r['original_KL'] for r in rows])}
for arm in ['exact_middle','exact_output','linear_middle','linear_output']:
    summary[arm]={'cost_error':stats([r['arms'][arm]['cost_relative_error_vs_original'] for r in rows]),
        'distribution_replay_KL':stats([r['arms'][arm]['KL_actual_to_handoff'] for r in rows]),
        'weighted_logit_response_error':stats([r['arms'][arm]['weighted_logit_response_error'] for r in rows]),
        'target_state_error':stats([r['arms'][arm]['target_state_relative_error_vs_actual'] for r in rows])}
audit={'native_manifest_files_verified':len(manifest),'handoff_arms_independently_verified':case_checks,
       'max_absolute_KL_difference':maximum,'no_extra_model_calls':True}
result={'status':'completed_exploratory_diagnostic','original_primary_support':False,
    'reused_questions':2,'directions_per_question':2,'signs':2,'strength':.125,
    'cases':8,'mechanics_pass':True,'summary':summary,'audit':audit,
    'interpretation':'Exact state hand-offs reproduce the original output. Running the true downstream network from frozen first-order middle/output shifts improves scalar-cost agreement on this fixed subset. Both omitted upstream curvature and nonlinear downstream/output effects can contribute; comparisons are not additive causal fractions.',
    'selection':'Both original held questions, first two direction indices, both signs; fixed before diagnostic native calls. These outcomes were already known, so this is not fresh confirmation.'}
(OUT/'HANDOFF_DIAGNOSTIC.json').write_text(json.dumps(result,indent=2)+'\n')

names=['quadratic_Fisher','linear_logit_endpoint','linear_middle','linear_output']
labels=['Quadratic\nFisher forecast','Exact softmax of\nlinear-logit forecast','Linear attention shift\n+ actual MLP and tail','Linear full-block shift\n+ actual tail']
colors=['#b24a69','#bc8a42','#327fa8','#37846e']
fig,axes=plt.subplots(1,2,figsize=(13,5.6),layout='constrained')
ax=axes[0]
med=[100*summary[n]['cost_error']['median'] for n in names]
lo=[100*summary[n]['cost_error']['minimum'] for n in names]
hi=[100*summary[n]['cost_error']['maximum'] for n in names]
positions=np.arange(4)
ax.bar(positions,med,color=colors,width=.65)
ax.errorbar(positions,med,yerr=[np.array(med)-lo,np.array(hi)-med],fmt='none',ecolor='#333333',capsize=4)
for x,y in zip(positions,med):ax.text(x,y+1,f'{y:.1f}%',ha='center',fontsize=10)
ax.set_xticks(positions,labels,fontsize=9);ax.set_ylabel('Output-cost relative error (%)')
ax.set_title('What changes when the later network is run exactly?')
ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
ax=axes[1]
for arm,color,label in [('linear_middle','#327fa8','First-order attention hand-off'),('linear_output','#37846e','First-order full-block hand-off')]:
    ax.scatter([r['original_KL'] for r in rows],[r['arms'][arm]['KL_base_to_handoff'] for r in rows],s=45,color=color,label=label)
values=[r['original_KL'] for r in rows]
line=np.geomspace(min(values)*.65,max(values)*1.5,100)
ax.plot(line,line,color='#333333',lw=1,label='original actual response')
ax.set(xscale='log',yscale='log',xlabel='Original intervention KL (nats)',ylabel='Hand-off intervention KL (nats)',title='Matched-prefix replay: the same eight cases')
ticks=[.002,.005,.01,.02,.05]
ax.set_xticks(ticks,[f'{x:g}' for x in ticks]);ax.set_yticks(ticks,[f'{x:g}' for x in ticks])
ax.xaxis.set_minor_formatter(NullFormatter());ax.yaxis.set_minor_formatter(NullFormatter())
ax.legend(fontsize=8);ax.grid(alpha=.15,which='both')
fig.suptitle('Exploratory error localization — strength 0.125\nTwo reused questions, two fixed directions, both signs; whiskers show cases, not uncertainty',fontsize=13)
fig.savefig(OUT/'handoff_error_localization.png',dpi=170);plt.close(fig)

lines=['MATCHED HAND-OFF DIAGNOSTIC —13 October2026','',
    'The original primary replication rule remains FAILED.',
    'This diagnostic uses2 already-observed held-out questions, the first2 direction',
    'indices, both signs, and strength0.125:8 cases. This is exploratory, not fresh',
    'confirmation, and no error threshold, direction or equation was retuned.','',
    'Mechanics',
    'The exact measured post-attention or full-block output change was transplanted',
    'into an otherwise normal run with the same complete prefix. Both reproduced',
    'the original output to negligible probability-distribution error. This checks',
    'the boundary location and state contract for this prompt-end intervention.','',
    'Median / maximum scalar output-cost errors versus the original intervention:']
for name in ['quadratic_Fisher','linear_logit_endpoint','linear_middle','linear_output','exact_middle','exact_output']:
    s=summary[name]['cost_error'];lines.append(f'{name}: {100*s["median"]:.5f}% / {100*s["maximum"]:.5f}%.')
lines+=['','What these comparisons mean',
    'Replacing quadratic Fisher cost with exact softmax of the linear-logit forecast',
    'tests output-distribution curvature while retaining the first-order logits.',
    'Handing a first-order internal forecast to the actual later network additionally',
    'restores nonlinear downstream processing. On this subset, cost errors fall',
    'substantially. The predicted internal states still differ from the true states,',
    'and their residual errors can be amplified. These are controlled substitution',
    'comparisons, not additive causal percentages or a proof of one exclusive cause.','',
    'Practical learning',
    'Accurate vector forecasts for one block do not automatically imply an accurate',
    'finite quadratic cost at the final output. Local response and metric transport',
    'remain useful, but their finite-strength validity must be measured separately.','',
    json.dumps(audit,indent=2),'',
    'Resources:36 additional CPU backbone calls;300 combined across the replication',
    'and this diagnostic. No GPU, Colab, new model training, or old-monitor resume.']
(OUT/'HANDOFF_REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'summary':summary,'audit':audit},indent=2))

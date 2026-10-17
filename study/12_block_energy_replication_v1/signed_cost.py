"""Secondary signed-pair explanation; preserves the failed primary criterion."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

BASE=Path('C:/Users/farha/Research4/artifacts/block_energy_replication_v1')
ROOT=BASE/'native_cpu_20261013_retry1';OUT=BASE/'evaluated_20261013'
assert json.loads((OUT/'RESULTS.json').read_text())['primary_support'] is False
rows=[r for r in json.loads((ROOT/'METRICS.json').read_text()) if r['precision']=='FP32' and r['split']=='held_out']
lookup={(r['question'],r['direction'],r['alpha']):r for r in rows}
questions=list(dict.fromkeys(r['question'] for r in rows));strengths=[.03125,.125,.5]
panels=[]
for pid in questions:
    for di in range(4):
        for strength in strengths:
            plus=lookup[pid,di,strength];minus=lookup[pid,di,-strength]
            cp=plus['endpoint_KL_base_to_changed'];cm=minus['endpoint_KL_base_to_changed']
            even=(cp+cm)/2;odd=(cp-cm)/2;prediction=plus['predicted_Fisher_cost']
            assert prediction==minus['predicted_Fisher_cost']
            panels.append({'question':pid,'direction':di,'strength':strength,
                'positive_cost':cp,'negative_cost':cm,'signed_pair_mean_cost':even,
                'signed_pair_odd_cost':odd,'odd_over_even':odd/even,
                'quadratic_prediction':prediction,'paired_mean_relative_error':abs(prediction-even)/even})
def stats(v):
    v=np.asarray(v,dtype=np.float64)
    return {'n':len(v),'minimum':float(v.min()),'median':float(np.median(v)),'maximum':float(v.max())}
summary={str(s):{'paired_mean_relative_error':stats([r['paired_mean_relative_error'] for r in panels if r['strength']==s]),
    'absolute_signed_asymmetry_fraction':stats([abs(r['odd_over_even']) for r in panels if r['strength']==s])} for s in strengths}
result={'status':'secondary_saved_data_diagnostic','original_primary_support':False,
    'questions':2,'pairs_per_strength':8,'summary':summary,'rows':panels,
    'interpretation':'A quadratic cost is even in alpha. Its inability to predict finite signed asymmetry contributes to per-sign failures. Averaging +/- removes odd terms; the improved paired score is a different endpoint and cannot replace the failed primary result.',
    'local_series':'For smooth fixed-context logits l(alpha)=l0+alpha*q+alpha^2*r/2+..., C(alpha)=alpha^2*Var_p(q)/2 + alpha^3*(Cov_p(q,r)/2+E_p[(q-E_p q)^3]/6)+O(alpha^4). This standard Taylor/Bregman expansion identifies network and softmax curvature terms; r was not measured here, so it supplies no new numerical forecast.',
    'not_activation_vector_even_odd_energy':'These are even/odd parts of scalar endpoint KL, distinct from earlier even/odd activation-vector identities.',
    'model_calls':0,'limits':'All outcomes already observed; secondary analysis, not fresh confirmation, semantic interpretation, physical conservation or equilibrium.'}
(OUT/'SIGNED_COST_DIAGNOSTIC.json').write_text(json.dumps(result,indent=2)+'\n')
fig,axes=plt.subplots(1,2,figsize=(12,5.2),layout='constrained')
for ax,pid in zip(axes,questions):
    sample=[next(r for r in panels if r['question']==pid and r['direction']==0 and r['strength']==s) for s in strengths]
    for key,label,color,style in [('positive_cost','Positive push','#2a7fa8','o-'),('negative_cost','Negative push','#bd7640','s-'),
                                   ('quadratic_prediction','Quadratic prediction (both signs)','#754c91','--'),
                                   ('signed_pair_mean_cost','Mean of actual +/- costs','#38856c',':')]:
        ax.plot(strengths,[r[key] for r in sample],style,color=color,label=label,linewidth=1.7)
    ax.set(xscale='log',yscale='log',xlabel='Steering strength',ylabel='Output KL / predicted cost (nats)',title=pid.replace('r12_test_','').capitalize()+': fixed direction0')
    ax.set_xticks(strengths,[f'{x:g}' for x in strengths]);ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(alpha=.15,which='both');ax.legend(fontsize=8)
fig.suptitle('A quadratic cost cannot distinguish the signs of a finite push\nSecondary explanation of the primary failure; averaging signs is a different measurement',fontsize=13)
fig.savefig(OUT/'signed_cost_asymmetry.png',dpi=170);plt.close(fig)
print(json.dumps(summary,indent=2))

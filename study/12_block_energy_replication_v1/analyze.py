"""Independent saved-array verification and fixed-rule replication analysis."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import NullFormatter

BASE=Path('C:/Users/farha/Research4/artifacts/block_energy_replication_v1')
ROOT=BASE/'native_cpu_20261013_retry1'
OUT=BASE/'evaluated_20261013'
OUT.mkdir(exist_ok=True)
status=json.loads((ROOT/'STATUS.json').read_text())
assert status['status']=='native_measurements_complete'
plan=json.loads((ROOT/'source/PLAN.json').read_text())
inputs=json.loads((ROOT/'INPUTS.json').read_text())
rows=json.loads((ROOT/'METRICS.json').read_text())
gate=json.loads((ROOT/'NUMERICAL_GATE_LOCK.json').read_text())
assert gate['pass'] and len(rows)==192

def dump(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(8*1024**2),b''):h.update(part)
    return h.hexdigest()

manifest=json.loads((ROOT/'EXPORT_HASHES.json').read_text())
assert all(sha(ROOT/name)==expected for name,expected in manifest.items())
original_receipt=json.loads((BASE/'NATIVE_EXPORT_RECEIPT_20261013.json').read_text())
assert sha(Path(original_receipt['path']))==original_receipt['sha256']

def logz(x):
    x=np.asarray(x,dtype=np.float64)
    maximum=float(x.max())
    return maximum+np.log(np.exp(x-maximum).sum())

def independent_kl(x,y):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    p=np.exp(x-logz(x))
    # Bregman/log-partition identity, independently from the worker's logp-logq.
    return max(0.,float(logz(y)-logz(x)-p@(y-x)))

def stats(values):
    values=np.asarray([v for v in values if v is not None],dtype=np.float64)
    return {'n':len(values),'minimum':float(values.min()),'median':float(np.median(values)),
            'maximum':float(values.max())} if len(values) else {'n':0}

lookup={(r['question'],r['precision'],r['direction'],r['alpha']):r for r in rows}
max_KL_diff=0.;max_energy_diff=0.;max_vector_diff=0.;verified_cases=0
baselines={}
predictions={}
for question in inputs['questions']:
    pid=question['id']
    for precision in ['FP32','BF16']:
        with np.load(ROOT/'baseline'/f'{pid}_{precision}.npz',allow_pickle=False) as z:
            baseline={k:z[k].copy() for k in ['input','middle','output','hidden','logits']}
        baselines[pid,precision]=baseline
        for di in range(4):
            with np.load(ROOT/'predictions'/f'{pid}_d{di}.npz',allow_pickle=False) as z:
                pred={k:z[k].copy() for k in ['q','coefficients','directions','hidden_direction']}
            predictions[pid,di]=pred
            with np.load(ROOT/'actual'/f'{pid}_d{di}_{precision}.npz',allow_pickle=False) as z:
                for ai,alpha in enumerate(z['alphas']):
                    row=lookup[pid,precision,di,float(alpha)]
                    measured=independent_kl(baseline['logits'],z['logits'][ai])
                    max_KL_diff=max(max_KL_diff,abs(measured-row['endpoint_KL_base_to_changed']))
                    if precision=='FP32':
                        p=np.exp(baseline['logits'].astype(np.float64)-logz(baseline['logits']))
                        q=pred['q'][0].astype(np.float64)
                        coefficient=float(p@(q*q)-(p@q)**2)
                        energy=.5*float(alpha)**2*coefficient
                        max_energy_diff=max(max_energy_diff,abs(energy-row['predicted_Fisher_cost']))
                        for i,name in enumerate(['input','middle','output']):
                            true=z[name][ai].astype(np.float64)-baseline[name]
                            den=np.linalg.norm(true)
                            error=float(np.linalg.norm(float(alpha)*pred['directions'][i].astype(np.float64)-true)/den) if den>1e-12 else None
                            if error is not None:max_vector_diff=max(max_vector_diff,abs(error-row['vector_relative_errors'][name]))
                    verified_cases+=1
assert max_KL_diff<1e-10 and max_energy_diff<1e-10 and max_vector_diff<1e-7
audit={'cases_independently_verified':verified_cases,'max_absolute_KL_difference':max_KL_diff,
       'max_absolute_energy_difference':max_energy_diff,'max_vector_error_difference':max_vector_diff,
       'native_manifest_files_verified':len(manifest),'old_failed_attempt_archive_unchanged':True,
       'verification_uses_no_model_calls':True}
dump('INDEPENDENT_AUDIT.json',audit)

held=[r for r in rows if r['split']=='held_out' and r['precision']=='FP32']
primary=[r for r in held if r['strength'] in plan['primary_small_strengths']]
assert len(held)==48 and len(primary)==32
support=[]
for question in inputs['questions']:
    if question['split']!='held_out':continue
    panel=[r for r in primary if r['question']==question['id']]
    assert len(panel)==16
    resolved=all(r['KL_resolved'] and not r['zero_delivery'] for r in panel)
    vector_ok=all(r['vector_relative_errors'][s] is not None and r['vector_relative_errors'][s]<=.1 for r in panel for s in ['middle','output'])
    energy_ok=all(r['relative_cost_error'] is not None and r['relative_cost_error']<=.2 for r in panel)
    support.append({'question':question['id'],'family':question['family'],'cases':len(panel),
        'all_resolved':resolved,'component_vectors_pass':vector_ok,'endpoint_cost_pass':energy_ok,
        'all_rules_pass':resolved and vector_ok and energy_ok,
        'middle_vector_error':stats([r['vector_relative_errors']['middle'] for r in panel]),
        'output_vector_error':stats([r['vector_relative_errors']['output'] for r in panel]),
        'cost_relative_error':stats([r['relative_cost_error'] for r in panel])})
dump('QUESTION_SUPPORT.json',support)
by_strength={}
for strength in plan['strengths']:
    panel=[r for r in held if r['strength']==strength]
    by_strength[str(strength)]={'cases':len(panel),'endpoint_cost_relative_error':stats([r['relative_cost_error'] for r in panel]),
        'middle_vector_relative_error':stats([r['vector_relative_errors']['middle'] for r in panel]),
        'output_vector_relative_error':stats([r['vector_relative_errors']['output'] for r in panel]),
        'final_hidden_relative_error':stats([r['final_hidden_relative_error'] for r in panel]),
        'weighted_logit_response_relative':stats([r['weighted_logit_response_relative'] for r in panel]),
        'actual_KL_nats':stats([r['endpoint_KL_base_to_changed'] for r in panel])}

def rank(x):
    x=np.asarray(x)
    assert len(np.unique(x))==len(x),'Report tied rankings explicitly'
    return np.argsort(np.argsort(x)).astype(float)

rank_rows=[]
geometry=[]
for question in inputs['questions']:
    pid=question['id']
    coefficients=np.array([predictions[pid,d]['coefficients'][0] for d in range(4)])
    for strength in plan['strengths']:
        actual=np.array([np.mean([lookup[pid,'FP32',d,sign*strength]['endpoint_KL_base_to_changed'] for sign in [1,-1]]) for d in range(4)])
        rank_rows.append({'question':pid,'split':question['split'],'strength':strength,
            'sampled_directions':4,'predicted_cost_max_min_ratio':float(coefficients.max()/coefficients.min()),
            'actual_signed_mean_cost_max_min_ratio':float(actual.max()/actual.min()),
            'rank_correlation_four_directions':float(np.corrcoef(rank(coefficients),rank(actual))[0,1]),
            'predicted_costs':(.5*strength**2*coefficients).tolist(),'actual_signed_mean_costs':actual.tolist()})
    for di in range(4):
        states=predictions[pid,di]['directions'].astype(np.float64)
        norms=np.linalg.norm(states,axis=1)
        angles=[]
        for a,b in [(0,1),(1,2)]:
            cosine=np.clip(states[a]@states[b]/(norms[a]*norms[b]),-1,1)
            angles.append(float(np.degrees(np.arccos(cosine))))
        geometry.append({'question':pid,'split':question['split'],'direction':di,
            'attention_residual_gain':float(norms[1]/norms[0]),'MLP_residual_gain':float(norms[2]/norms[1]),
            'full_block_gain':float(norms[2]/norms[0]),'direction_change_degrees_attention_MLP':angles})
dump('DIRECTION_RANKING.json',rank_rows)
dump('COMPONENT_GEOMETRY.json',geometry)
precision=[]
for strength in plan['strengths']:
    panel=[r for r in rows if r['split']=='held_out' and r['precision']=='BF16' and r['strength']==strength]
    ratios=[r['endpoint_KL_base_to_changed']/lookup[r['question'],'FP32',r['direction'],r['alpha']]['endpoint_KL_base_to_changed'] for r in panel]
    precision.append({'strength':strength,'cases':len(panel),
        'BF16_to_FP32_KL_ratio':stats(ratios),
        'BF16_delivered_to_requested_norm_ratio':stats([r['actual_input_norm']/r['requested_norm'] for r in panel]),
        'BF16_off_direction_fraction':stats([r['off_direction_norm']/r['actual_input_norm'] for r in panel if r['actual_input_norm']>0]),
        'zero_delivery_cases':sum(r['zero_delivery'] for r in panel),
        'not_a_matched_baseline_isolation_of_rounding':True})
dump('PRECISION_CONTROL.json',precision)
result={'status':'completed_fixed_rule_analysis','held_questions':2,'calibration_questions':2,
    'signed_FP32_interventions':96,'signed_BF16_controls':96,'primary_held_cases':32,
    'primary_support':all(s['all_rules_pass'] for s in support),'by_strength':by_strength,
    'max_transport_weighted_relative':max(max(r['transport_weighted_relative']) for r in gate['transport_checks']),
    'max_transport_cost_relative':max(max(r['transport_cost_relative']) for r in gate['transport_checks']),
    'derivative_primal_exact_cases':sum(r['base_logits_exact'] for r in gate['transport_checks']),
    'derivative_directed_base_pairs':len(gate['transport_checks']),
    'max_FD_weighted_relative':max(r['weighted_logit_relative'] for r in gate['FD_checks']),
    'held_component_gains':{name:stats([g[name] for g in geometry if g['split']=='held_out']) for name in ['attention_residual_gain','MLP_residual_gain','full_block_gain']},
    'model_calls':status['model_calls'],'GPU_allocations':0,'independent_audit':audit,
    'limits':plan['limits']}
dump('RESULTS.json',result)
with (OUT/'ALL_METRICS.csv').open('w',newline='',encoding='utf-8') as f:
    fields=['question','split','precision','direction','alpha','endpoint_KL_base_to_changed','predicted_Fisher_cost','relative_cost_error','actual_input_norm','requested_norm','off_direction_norm','zero_delivery']
    writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)

# Annotated representative case chosen by first held question and direction0,
# independent of whether that case happened to perform well.
sample=next(q for q in inputs['questions'] if q['split']=='held_out')
pred=predictions[sample['id'],0]
norms=np.linalg.norm(pred['directions'].astype(np.float64),axis=1)
energies=.5*pred['coefficients']
fig=plt.figure(figsize=(13,9),layout='constrained')
grid=fig.add_gridspec(2,2,height_ratios=[.8,1.4])
flow=fig.add_subplot(grid[0,:]);flow.set_xlim(0,1);flow.set_ylim(0,1);flow.axis('off')
flow.set_title('One native Qwen block: attention, MLP, and the same predicted output cost',loc='left',fontweight='bold')
centers=[.16,.50,.84]
labels=['Before attention','After attention + residual','After MLP + residual']
colors=['#e1eef8','#dcefe7','#f4e8cf']
for i,x in enumerate(centers):
    flow.add_patch(FancyBboxPatch((x-.145,.30),.29,.44,boxstyle='round,pad=0.015',facecolor=colors[i],edgecolor='#45556b'))
    flow.text(x,.64,labels[i],ha='center',va='center',fontsize=11,fontweight='bold')
    flow.text(x,.48,f'Perturbation gain: {norms[i]:.3f}\nOutput-cost coefficient: {energies[i]:.5g}',ha='center',va='center',fontsize=10)
for a,b in zip(centers[:-1],centers[1:]):
    flow.annotate('',xy=(b-.16,.52),xytext=(a+.16,.52),arrowprops={'arrowstyle':'->','lw':1.8,'color':'#45556b'})
flow.text(.5,.13,'Representative grammar question, direction0. Local derivatives per unit push.\nCost agreement follows the chain rule; finite interventions provide the prediction test.',ha='center',fontsize=10,color='#45556b')

ax=fig.add_subplot(grid[1,0])
palette={.03125:'#2878a8',.125:'#dc8934',.5:'#b44a68'}
for strength,color in palette.items():
    panel=[r for r in held if r['strength']==strength and r['KL_resolved']]
    ax.scatter([r['endpoint_KL_base_to_changed'] for r in panel],[r['predicted_Fisher_cost'] for r in panel],color=color,s=35,alpha=.85,label=f'strength {strength:g}')
values=[r['endpoint_KL_base_to_changed'] for r in held]+[r['predicted_Fisher_cost'] for r in held]
line=np.geomspace(min(values)*.7,max(values)*1.4,100)
ax.plot(line,line,color='#333333',lw=1,label='perfect prediction')
ax.plot(line,.8*line,'--',color='#888888',lw=.8);ax.plot(line,1.2*line,'--',color='#888888',lw=.8)
ax.set(xscale='log',yscale='log',xlabel='Measured output KL (nats)',ylabel='Predicted quadratic output cost',title='Independent finite-response check: two held-out questions')
ax.legend(fontsize=8);ax.grid(alpha=.15,which='both')

ax=fig.add_subplot(grid[1,1])
for name,label,color in [('middle','After attention','#27846f'),('output','After MLP','#7562a6')]:
    med=[];low=[];high=[]
    for strength in plan['strengths']:
        vals=[100*r['vector_relative_errors'][name] for r in held if r['strength']==strength and r['vector_relative_errors'][name] is not None]
        med.append(np.median(vals));low.append(min(vals));high.append(max(vals))
    ax.plot(plan['strengths'],med,'o-',color=color,label=label)
    ax.fill_between(plan['strengths'],low,high,color=color,alpha=.15)
ax.axhline(10,color='#888888',ls='--',lw=1,label='primary vector tolerance')
ax.set(xscale='log',xlabel='Steering strength',ylabel='Full activation-change-vector error (%)',title='How far the local response remains accurate')
ax.set_xticks(plan['strengths'],[f'{x:g}' for x in plan['strengths']]);ax.legend(fontsize=8);ax.grid(alpha=.15)
ax.xaxis.set_minor_formatter(NullFormatter())
fig.suptitle('FishBack core-equation replication on Qwen3.5-0.8B — CPU / FP32\nFour directions, both signs; shaded ranges are cases, not confidence intervals',fontsize=14)
fig.savefig(OUT/'block_response_and_cost.png',dpi=170);plt.close(fig)

receipt=json.loads((BASE/'NATIVE_EXPORT_RECEIPT_20261013_RETRY1.json').read_text())
lines=['BLOCK RESPONSE AND OUTPUT-COST REPLICATION —13 October2026','',
    f'Frozen primary rule: {"PASSED" if result["primary_support"] else "FAILED"}.',
    'Adaptation of FishBack Theorem1 and AppendixB.1 to unchanged Qwen3.5-0.8B.',
    'This is a core-equation replication, not a reproduction of concept-steering benchmarks.','',
    'Four fresh questions:2 calibration,2 held out. One full-attention block(index11),',
    'one prompt-end prediction position,4 fresh orthonormal directions, both signs.',
    'FP32 original operations;96 signed interventions plus96 separate BF16 controls.',
    'No activation function was replaced. No model, probe or classifier was trained.','',
    'What the equations mean',
    'The local response is delta_after approximately A*delta_before. The output cost',
    'is0.5*delta^T*G*delta, with G=J^T*H*J from the actual downstream derivative and',
    'full-vocabulary softmax Fisher. Under transported local directions, the same',
    'output-cost scalar is recovered before attention, after attention, and after',
    'the MLP. That equality follows from the chain rule; it is not a newly measured',
    'physical conservation or equilibrium law. Finite endpoint KL is the independent test.','',
    f'Maximum transported cost disagreement: {result["max_transport_cost_relative"]:.4g} relative.',
    f'Maximum calibration weighted logit derivative error: {100*result["max_FD_weighted_relative"]:.4g}%.',
    f'Derivative primal logits exact in {result["derivative_primal_exact_cases"]}/{result["derivative_directed_base_pairs"]} directed base pairs.','',
    'Held-out results: median / maximum across16 repeated signed cases per strength',
    '(two independent questions; not16 independent question families).']
for strength,panel in by_strength.items():
    def percent(field):
        s=panel[field];return f'{100*s["median"]:.3f}% / {100*s["maximum"]:.3f}%'
    lines.append(f'Strength {strength}: after-attention vector {percent("middle_vector_relative_error")}; after-MLP vector {percent("output_vector_relative_error")}; output cost {percent("endpoint_cost_relative_error")}.')
lines+=['','Per-question primary support:']
for s in support:lines.append(f'{s["family"]}: all rules={s["all_rules_pass"]}; vectors={s["component_vectors_pass"]}; cost={s["endpoint_cost_pass"]}; resolved={s["all_resolved"]}.')
lines+=['','Local component geometry (held-out sampled directions):']
for name,s in result['held_component_gains'].items():lines.append(f'{name}: minimum {s["minimum"]:.4f}, median {s["median"]:.4f}, maximum {s["maximum"]:.4f}.')
lines+=['','Equal-size directions: predicted vs measured direction ranking']
for r in rank_rows:
    if r['split']=='held_out':lines.append(f'{r["question"]}, strength {r["strength"]}: predicted max/min {r["predicted_cost_max_min_ratio"]:.3f}; measured +/-mean max/min {r["actual_signed_mean_cost_max_min_ratio"]:.3f}; four-direction rank correlation {r["rank_correlation_four_directions"]:.3f}.')
lines+=['','BF16 is a separate precision sensitivity control. Its full-model baseline',
    'differs from FP32; differences do not uniquely isolate rounding. Nominal and',
    'actual delivered inputs, off-direction motion and zero flags are retained.']
for r in precision:lines.append(f'Strength {r["strength"]}: BF16/FP32 measured KL ratio median {r["BF16_to_FP32_KL_ratio"]["median"]:.3f}, maximum {r["BF16_to_FP32_KL_ratio"]["maximum"]:.3f}; zero deliveries {r["zero_delivery_cases"]}.')
lines+=['','Independent numerical audit',json.dumps(audit,indent=2),'',
    'Preservation and resources',
    'The first launch stopped at a tokenizer-container comparison before any model',
    'inference. That attempt is archived. Retry1 normalized the wrapper, with exact',
    'byte-identical scientific plan, prompts and directions; no scientific retuning.',
    f'Completed native/export duration {receipt["native_and_export_seconds"]:.2f}sec; cumulative including first attempt {receipt["cumulative_seconds_with_prior_attempt"]:.2f}sec.',
    f'Whole-backbone calls {status["model_calls"]}; GPU allocations0; CPU threads4.',
    'No Colab, new training, old-monitor continuation or unrelated process changes.','',
    'Limits',*plan['limits'],'',
    'Source: https://arxiv.org/html/2605.17231v2',
    'Saved normal-run predictions preceded all primary signed interventions.']
(OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
files=[p for p in OUT.iterdir() if p.is_file() and p.name!='ARTIFACT_HASHES.json']
dump('ARTIFACT_HASHES.json',{p.name:sha(p) for p in files})
print(json.dumps({'primary_support':result['primary_support'],'by_strength':by_strength,
    'support':support,'max_transport_cost_relative':result['max_transport_cost_relative'],
    'held_component_gains':result['held_component_gains'],'audit':audit},indent=2))

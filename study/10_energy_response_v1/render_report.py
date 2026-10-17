"""Scientific figures and a readable bounded-response report; no new inference."""
from pathlib import Path
import csv,json,math,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).parent;BASE=Path('C:/Users/farha/Research4/artifacts/energy_response_v1')
PILOT=BASE/'pilot_20261012';MATCHED=BASE/'matched_snapshot_20261012';OUT=BASE/'evaluated_20261012'
r=json.loads((OUT/'RESULTS.json').read_text());ranges=r['MLP_validity_ranges']
angle_rows=list(csv.DictReader((OUT/'DERIVED_ANGLE_PREDICTIONS.csv').open()))
energy=json.loads((PILOT/'FUNCTIONAL_ENERGY.json').read_text())
matched=list(csv.DictReader((OUT/'MATCHED_SNAPSHOT_DECOMPOSITION.csv').open()))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
strengths=[.001953125,.0078125,.03125,.125,.5,2.]
for precision,color in [('FP32','#1565c0'),('BF16','#d26b17')]:
    a=[x for x in ranges if x['precision']==precision and x['map']=='branch']
    med=[math.degrees(math.acos(-x['opposition_cosine']['median'])) for x in a]
    lo=[math.degrees(math.acos(-x['opposition_cosine']['min'])) for x in a]
    hi=[math.degrees(math.acos(-x['opposition_cosine']['max'])) for x in a]
    axes[0].plot(strengths,med,'o-',color=color,label=precision)
    axes[0].fill_between(strengths,lo,hi,color=color,alpha=.12)
pred=[]
for eps in strengths:
    pred.append(np.median([float(x['second_order_angle_degrees']) for x in angle_rows if x['map']=='branch' and float(x['strength'])==eps]))
axes[0].plot(strengths,pred,'--',color='#173b63',label='Quadratic angle prediction')
axes[0].axhline(90,color='#777777',linestyle=':',linewidth=1)
axes[0].set(xscale='log',ylim=(40,185),xlabel='Perturbation strength',ylabel='Angle between + / - responses (degrees)',title='Opposition changes with strength')
axes[0].legend(fontsize=8,loc='lower left')
for field,label,color in [('first_error','First order','#1976bd'),('second_error','Second order','#37864b')]:
    a=[x for x in ranges if x['precision']=='FP32' and x['map']=='branch']
    axes[1].plot(strengths,[x[field]['median'] for x in a],'o-',color=color,label=label)
    axes[1].fill_between(strengths,[x[field]['min'] for x in a],[x[field]['max'] for x in a],color=color,alpha=.15)
axes[1].axhline(.1,color='#777777',linestyle=':',label='10% reference')
axes[1].set(xscale='log',yscale='log',xlabel='Perturbation strength',ylabel='Relative response-vector error',title='A useful local model has limits')
axes[1].legend(fontsize=8)
colors={.03125:'#1565c0',.125:'#37864b',.5:'#c45b1b'}
for eps in [.03125,.125,.5]:
    a=[x for x in energy if x['split']=='held_out' and x['strength']==eps]
    axes[2].scatter([x['predicted_KL'] for x in a],[x['measured_KL_raw'] for x in a],s=32,color=colors[eps],label=f'Strength {eps:g}')
all_values=[x[k] for x in energy if x['split']=='held_out' for k in ['predicted_KL','measured_KL_raw']]
grid=np.geomspace(min(all_values)*.7,max(all_values)*1.4,100)
axes[2].plot(grid,grid,color='#666666',linewidth=1)
axes[2].fill_between(grid,grid/1.2,grid/.8,color='#999999',alpha=.12,label='20% error band')
axes[2].set(xscale='log',yscale='log',xlabel='Predicted output KL (nats)',ylabel='Measured output KL (nats)',title='Output-sensitive energy predicts small edits')
axes[2].legend(fontsize=8,loc='upper left')
fig.suptitle('Frozen response equations: tests on two held-out question families and two held-out directions',fontsize=13)
fig.supxlabel('Four fresh questions total. Directions, positions, signs and strengths are repeated measurements, not independent questions.',fontsize=9)
fig.savefig(OUT/'response_and_energy.png',dpi=150);plt.close(fig)

fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
for pi,(precision,color) in enumerate([('FP32','#1565c0'),('BF16','#d26b17')]):
    for i,eps in enumerate([.03125,.5]):
        values=[math.degrees(math.acos(-float(x[precision+'_opposition_cosine']))) for x in matched if float(x['strength'])==eps]
        x=i+(pi-.5)*.2
        axes[0].plot([x,x],[min(values),max(values)],color=color,linewidth=2)
        axes[0].scatter(np.linspace(x-.04,x+.04,len(values)),values,color=color,s=22,label=precision if i==0 else None)
axes[0].axhline(90,color='#777777',linestyle=':',linewidth=1)
axes[0].set(xticks=[0,1],xticklabels=['0.03125','0.5'],ylim=(40,185),xlabel='Perturbation strength',ylabel='Angle between + / - responses',title='Same input, state and rounded edits')
axes[0].legend(fontsize=9)
fields=[('linear_input_signed_projection','Input asymmetry','#8c8c8c'),('FP32_remainder_signed_projection','Finite FP32 response','#1565c0'),('precision_difference_signed_projection','Precision difference','#d26b17')]
for fi,(field,label,color) in enumerate(fields):
    for i,eps in enumerate([.03125,.5]):
        values=[float(x[field]) for x in matched if float(x['strength'])==eps]
        x=i+(fi-1)*.22
        axes[1].bar(x,np.mean(values),width=.2,color=color,label=label if i==0 else None)
        axes[1].plot([x,x],[min(values),max(values)],color='#333333',linewidth=1)
axes[1].axhline(0,color='#555555',linewidth=.8)
axes[1].set(xticks=[0,1],xticklabels=['0.03125','0.5'],ylim=(-.05,1.12),xlabel='Perturbation strength',ylabel='Signed projection onto native even response',title='Different mechanisms at different strengths')
axes[1].legend(fontsize=8,loc='center right')
old=Path('C:/Users/farha/Research4/artifacts/propagation_followup_v1/recovery_20261010/verified_export/verified_native/native/pair1_completed')
def token_cos(base,plus,minus):
    p=plus.astype(float)-base.astype(float);m=minus.astype(float)-base.astype(float)
    return np.sum(p*m,axis=1)/(np.linalg.norm(p,axis=1)*np.linalg.norm(m,axis=1))
archive={}
for arm in ['base','plus_v','minus_v']:
    with np.load(old/(arm+'.npz'),allow_pickle=False) as z:archive[arm]=z['residuals'][15].copy()
axes[2].scatter(np.arange(1,28),token_cos(archive['base'],archive['plus_v'],archive['minus_v'])[1:],color='#777777',s=15,label='Historical GPU BF16')
for precision,color,style in [('BF16','#d26b17','-'),('FP32','#1565c0','--')]:
    data={}
    for arm in ['base','plus_v','minus_v']:
        with np.load(MATCHED/'historical_reproduction'/f'{precision}_{arm}.npz',allow_pickle=False) as z:data[arm]=z['readout15'].copy()
    axes[2].plot(np.arange(1,28),token_cos(data['base'],data['plus_v'],data['minus_v'])[1:],style,color=color,label='CPU '+precision)
axes[2].axhline(0,color='#777777',linestyle=':',linewidth=1)
axes[2].set(xlabel='Cached prediction position',ylabel='Cosine between + / - downstream changes',ylim=(-1.03,.7),title='Historical phenomenon survives FP32')
axes[2].legend(fontsize=8,loc='lower right')
fig.suptitle('Controlled explanation: finite nonlinearity at larger strength; rounding matters for very small changes',fontsize=13)
fig.supxlabel('Matched snapshots: two held-out questions, repeated positions/schedules. Signed projections sum to1 but are not variance fractions or global causal percentages.',fontsize=9)
fig.savefig(OUT/'precision_state_controls.png',dpi=150);plt.close(fig)

text='''ENERGY AND RESPONSE STUDY —12 October2026

Outcome
The old near-perpendicular signed response is reproducible and is not explained
by a fixed rotation or by BF16 rounding alone. At the historical0.5 strength,
substantial sign-even finite response survives FP32. At much smaller strengths,
rounding can instead dominate native BF16 response geometry. A context-specific
local derivative model and an output-sensitive energy make accurate predictions
within measured validity ranges. This is not a general steering detector.

What was actually done
The same pinned Qwen3.5-0.8B checkpoint was used, untrained and unquantized.
Four fresh questions were frozen before native evaluation: two calibration
families and two held-out families (code explanation and planning). Four fixed
directions included two held-out directions. Incoming residuals came from real
multi-token prefixes and the first cached prediction after fixed token760.
The one selected MLP is block11 (zero-based), immediately downstream of the
historical post-block10 site. We studied B(h)=MLP(RMSNorm(h)) and F(h)=h+B(h)
separately. No layer search or matrix fitted across questions was used.
Native BF16 casting and actual delivered perturbations were retained. The
smooth reference promotes the exact stored checkpoint values to FP32. A local
FP64 formula control is explicitly different from literal native normalization.
Full-prefix attention retained native masks, RoPE and cloned real KV/recurrent
state. No isolated single-token/no-cache attention result was used as evidence
about cached decoding. The differentiable functional-energy phase used actual
full prefixes; its base hidden states exactly match native prefill records.

Reconciliation of the old observation
All248 old native manifest members were verified unchanged. The original
panel forced identical28-token histories and injected persistently at all
prefill positions and every decode step. Downstream caches therefore differed
across signs despite identical token IDs. The minus arm used a native matching
grid; actual site updates were almost opposite (cosine about0.999984), not
perfect mathematical negatives. A strongly anisotropic linear map can amplify
a tiny input asymmetry, so near-opposition alone is not proof of curvature.

The decisive mathematical relation
Let d+ and d- be the two downstream changes relative to the untreated base.
Define odd=(d+-d-)/2 and even=(d++d-)/2. Then exactly:
    d+ dot d- = ||even||^2 - ||odd||^2.
Opposite linear responses have even=0. Comparable odd/even norms give a
near-right angle. For example d+=[1,1], d-=[-1,1] have a shared upward component
and opposite sideways components. Their right angle is not a fixed rotation.
This identity describes the angle; the precision/state experiments test where
the even component comes from.

Predictions made before measuring perturbations
Actual RMSNorm/SwiGLU directional derivatives give:
    d+ approximately epsilon*Jv + epsilon^2/2*H[v,v]
    d- approximately -epsilon*Jv + epsilon^2/2*H[v,v].
The first term changes sign; the second does not. Analytic derivatives passed
autograd and FP64 finite-difference calibration. Coefficients were saved and
hashed before signed measurements, including held-out directions/questions.
At strengths0.0078125 and0.03125, all declared held-out FP32 vector predictions
passed the10% criterion. Maximum first-order error was1.89% for the MLP branch
and1.44% for the complete residual map. These are repeated measurements over
two independent held-out questions, not64 independent scenarios.

Finite-response limits
At strength0.5, branch first-order errors were24.0–30.7%; the quadratic model
reduced them to6.37–8.46%. It predicted response angles within0.187degrees.
At strength2, measured branch angles were80.7–95.0degrees. The quadratic angle
prediction remained within4degrees, but vector errors reached133.6%. Thus an
apparently good angle alone is not evidence of an accurate full response map.
The formal quadratic crossover epsilon=2||Jv||/||Hvv|| falls at1.66–2.05 for
these held-out branch cases. It is an extrapolation where vector validity fails,
not a confirmed universal threshold.

Why normalization and gating can create this structure
For normalized input z, g=Wg*z and u=Wu*z, SwiGLU satisfies the exact smooth
identity:
    B = 0.5*Wd*(g*u) + 0.5*Wd*(g*u*tanh(g/2)).
With bias-free projections and odd RMSNorm, the first part is sign-even about
zero and the second sign-odd. Around a nonzero base both can contribute to a
centered even response; this is not a proof that one term alone causes every
observed effect. The RMS denominator also changes with finite displacement.
For its quadratic-even part Q(h)=N2(h)/s(h), where s=mean(h^2)+norm_epsilon,
the exact directional response has a rational form:
    Q(h+alpha*v)-Q(h) = s0/s(alpha) *
        [alpha*J_Qv + alpha^2*(H_Qvv/2 + (s1/s0)*J_Qv)],
with s(alpha)=s0+alpha*s1+alpha^2*s2, s1=2mean(h*v), s2=mean(v^2).
When h dot v=0, a common positive saturation factor changes amplitudes but
preserves the quadratic prediction's angle. This supplies a structural reason
why orientation can be better predicted than vector magnitude. Applying that
resummation to the whole SwiGLU branch was examined only retrospectively on
saved data: at strength2 its branch errors were16.5–24.8%, versus97.7–133.6%
for the ordinary quadratic expansion. The residual-map errors were7.2–11.1%.
That is an exploratory lead needing a fresh freeze/test, not a new confirmed
generalization result. No additional model calls were made for this analysis.

Matched-state evidence about the earlier right angle
The block11-through15 tail was evaluated at identical saved BF16 numerical
inputs, identical cloned state/RoPE/mask, and identical actual rounded edits.
The trained values were merely promoted for the FP32 control. Native replay
was exact, and the derivative check error was0.243%.
At0.5, the FP32 and BF16 responses both moved toward/perpendicular to each
other, while linear predictions using the ACTUAL input changes remained
almost opposite (opposition cosine0.9997–1.0000). The finite FP32 remainder
accounts for99.6–100.1% of the signed projection onto the native even response;
the downstream precision-difference term contributes about-0.07–0.36%.
At0.03125, the FP32 responses recover near-opposition, while BF16 remains much
less opposed. Here precision differences contribute93.7–100.1% of that signed
projection. These are exact vector-accounting projections in these controls,
which can be negative and are not variance fractions or universal causal shares.

Historical reproduction
The first originally listed, already-inspected prompt was reproduced with its
109-token prefix,28 forced tokens, original intervention code/matching grid,
and fresh causal state for each condition. Mean decode response cosines were:
    old GPU BF16:0.10665
    new CPU BF16:0.10662
    new CPU FP32:0.10595.
Zero cosine is a right angle; -1 is opposition. Sign-even energy fractions
were approximately0.599,0.598 and0.597. Persistence of the phenomenon in FP32,
together with matched-state controls, rules out rounding alone as its explanation
at this finite strength. Causal history and prefix breadth change effects and
were kept separate from local held-state response; neither was pooled as the
same Jacobian experiment. We did not uniquely assign all full-tail curvature
to a single MLP or claim the same mechanism quantitatively in every old prompt.

Output-sensitive energy
For actual logit derivative q, base next-token distribution p and the same
fixed prefix, the local prediction is:
    E approximately epsilon^2/2 * [sum p*q^2 - (sum p*q)^2].
This predicts D_KL(p_base || p_changed), all248320 vocabulary entries, in nats.
Exact forward-mode AD supplied the derivative; no full Fisher matrix, top-K
truncation, global covariance fit or classifier training was used.
All16 declared small-strength held-out checks passed the20% criterion:
maximum error1.07% at0.03125 and5.79% at0.125. At0.5 only4/8 passed; maximum
error38.8%. The formula is therefore a useful local cost with a measured limit,
not physical/conserved energy, global downhill dynamics or a hidden-edit detector.
Independent NumPy calculations verified the KL and Fisher variance values.

Geometry and generalization limits
Four input directions sample a tiny part of1024-dimensional space. Full-output
gain on that input span, projected polar/eigen/asymmetry metrics and projection
leakage are all reported. The branch writes80.4–92.8% of sampled response energy
outside that four-direction span; the complete residual map leaks29.5–48.7%.
Sampled branch singular gains are0.498–0.893, while the complete map spans
0.886–1.272. These are sampled-span statements, not full-space contraction or
amplification bounds. Unnormalized direction-sum superposition errors for the
FP32 branch are0.62–0.80% at0.03125 and9.39–12.82% at0.5. Native BF16 small-step
errors are larger and include an observed delivered-input sum mismatch, so they
cannot be called intrinsic interaction alone. A projected matrix does not identify a global rotation,
and eigenvalues alone do not establish finite-step gain or transient growth.
There are four fresh questions, two held-out families, two held-out directions,
and one inspected reproduction question. No population confidence interval or
27B/cross-model extrapolation is justified. The prior covariance-energy and
14-question detector failures, and the earlier stopped rotation branch, remain
failures/inconclusive results. The stopped consistency monitor remains stopped.

Resources and preservation
Pilot:258 whole-backbone calls. Explanatory follow-up:168 whole-backbone calls
and85 bounded tail evaluations. All gates passed; no scientific retries, new
classifier, parameter tuning, cap increase or GPU allocation. The combined
native window exported by25.11minutes, below its45-minute hard ceiling. Later
analysis/backup involved no live model worker. Both owned workers exited;
Research3/unrelated sessions and the user CPU Drive mount were untouched.
Native sources, predictions, caches, responses and hashes are retained privately.

Primary references
https://arxiv.org/html/2605.14258
https://arxiv.org/html/2505.19458
https://arxiv.org/html/2605.25225v2
https://arxiv.org/html/2605.17231v2
The first paper's single-token/no-KV-cache mean-Jacobian composition and the
second paper's unmasked/no-position/no-MLP recurrent setup were not adopted as
evidence about actual cached Qwen decoding. The third motivates bounded response
tests; the fourth motivates the local output-sensitive cost.
'''
(OUT/'REPORT.txt').write_text(text,encoding='utf-8')
status={'date':'2026-10-12','status':'staged mathematical response/energy pilot complete','native_whole_backbone_calls':426,
    'bounded_tail_evaluations':85,'fresh_questions':4,'held_out_questions':2,'held_out_directions':2,
    'MLP_primary_support':True,'functional_small_strength_support':True,'functional_strength0_5_all_cases_support':False,
    'strong_historical_perpendicular_response_survives_FP32':True,'no_global_rotation_or_detector_claim':True,
    'norm_resummation_status':'post-results saved-data lead only','GPU_allocations':0,'owned_model_workers_remaining':0,
    'old_consistency_monitor_remains_stopped':True,'artifacts':str(OUT)}
(HERE/'STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
for name in ['PLAN.json','LITERATURE.json','EXECUTION_FREEZE.json','MATCHED_SNAPSHOT_PLAN.json','MATCHED_SNAPSHOT_EXECUTION_FREEZE.json','maths.py','native.py','pilot.py','matched_snapshot.py','analyze.py','derived_angles.py','norm_resummation.py','render_report.py','STATUS.json']:
    shutil.copyfile(HERE/name,OUT/name)
print(OUT/'REPORT.txt')

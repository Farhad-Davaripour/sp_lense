"""Report the prospective equation improvement and the failed output criterion."""
from pathlib import Path
import csv,json,math,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
HERE=Path(__file__).parent;BASE=Path('C:/Users/farha/Research4/artifacts/norm_response_v2')
ROOT=BASE/'prospective_20261012';OUT=BASE/'evaluated_20261012'
r=json.loads((OUT/'RESULTS.json').read_text());ranges=r['vector_error_ranges'];selected=r['selected_equation']
names=['linear','quadratic','rational',selected]
labels={'linear':'Linear','quadratic':'Ordinary quadratic','rational':'Normalization-aware rational','exact_norm_gate2':'Exact normalization + gate expansion'}
colors={'linear':'#777777','quadratic':'#1565c0','rational':'#cf771f','exact_norm_gate2':'#2e864b'}
strengths=[.0078125,.03125,.125,.5,1.25,2.]
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
for name in names:
    a=[x for x in ranges if x['precision']=='FP32' and x['map']=='branch' and x['equation']==name]
    med=[x['relative_vector_error']['median'] for x in a];lo=[x['relative_vector_error']['min'] for x in a];hi=[x['relative_vector_error']['max'] for x in a]
    axes[0].plot(strengths,med,'o-',color=colors[name],label=labels[name])
    axes[0].fill_between(strengths,lo,hi,color=colors[name],alpha=.1)
axes[0].set(xscale='log',yscale='log',xlabel='Perturbation strength',ylabel='Full MLP-change-vector relative error',title='Fresh held-out vector predictions')
axes[0].legend(fontsize=7.5,loc='upper left')
for precision,color,label,field,style in [
    ('FP32','#2e864b','FP32 reference','relative_vector_error','-'),
    ('BF16','#cf771f','BF16 nominal input','relative_vector_error','-'),
    ('BF16','#cf771f','BF16 actual-input control','actual_input_relative_error','--')]:
    a=[x for x in ranges if x['precision']==precision and x['map']=='branch' and x['equation']==selected]
    axes[1].plot(strengths,[x[field]['median'] for x in a],'o'+style,color=color,label=label)
axes[1].set(xscale='log',yscale='log',xlabel='Perturbation strength',ylabel='Selected equation relative vector error',title='Weak BF16 changes remain limited by rounding')
axes[1].legend(fontsize=8,loc='lower right')
question_rules=[x for x in r['question_rules'] if x['map']=='branch']
ids=[x['question'] for x in question_rules];x=np.arange(4)
for offset,name in [(-.24,'quadratic'),(0,'rational'),(.24,selected)]:
    values=[]
    metrics=json.loads((ROOT/'INTERNAL_METRICS.json').read_text())
    for pid in ids:
        a=[z for z in metrics if z['question']==pid and z['direction']>=2 and z['precision']=='FP32' and z['map']=='branch' and z['strength']==2.]
        values.append(float(np.median([z['errors'][name]['relative'] for z in a]))*100)
    axes[2].bar(x+offset,values,width=.22,color=colors[name],label=labels[name])
    if name==selected:
        for xx,v in zip(x+offset,values):axes[2].annotate(f'{v:.1f}%',(xx,v),xytext=(0,4),textcoords='offset points',ha='center',fontsize=8)
axes[2].set(xticks=x,xticklabels=['Grammar','Logic','Invitation','Comparison'],ylabel='Median full-vector error (%)',title='Improvement across all four withheld families')
# Colors are already identified in the first panel; avoid covering family bars.
fig.suptitle('Normalization-aware equation: calibration-locked choice, prospective confirmation at the same fixed MLP',fontsize=13)
fig.supxlabel('Eight fresh questions: four calibration, four held out. Directions, signs, strengths and positions are repeated measurements.',fontsize=9)
fig.savefig(OUT/'equation_improvement.png',dpi=150);plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
olabels={'Fisher':'Original local Fisher cost','quadratic_endpoint':'Quadratic shift + linear tail','rational_endpoint':'Rational shift + linear tail','selected_endpoint':'Selected shift + linear tail'}
ocolors={'Fisher':'#777777','quadratic_endpoint':'#1565c0','rational_endpoint':'#cf771f','selected_endpoint':'#2e864b'}
for ax,position,title in zip(axes,['prefill','decode1'],['Prompt-end prediction','First cached prediction']):
    for predictor in ['Fisher','quadratic_endpoint','rational_endpoint','selected_endpoint']:
        a=[x for x in r['output_error_ranges'] if x['position']==position and x['predictor']==predictor]
        a=sorted(a,key=lambda x:x['strength'])
        ax.plot([x['strength'] for x in a],[x['relative_error']['median'] for x in a],'o-',color=ocolors[predictor],label=olabels[predictor])
        if predictor=='selected_endpoint':
            ax.fill_between([x['strength'] for x in a],[x['relative_error']['min'] for x in a],[x['relative_error']['max'] for x in a],color=ocolors[predictor],alpha=.1)
    ax.axhline(.25,color='#777777',linestyle=':',linewidth=1)
    ax.set(xscale='log',yscale='log',xlabel='Perturbation strength',ylabel='Median endpoint-KL relative error',title=title)
    ax.set_xticks([.125,.5,2.]);ax.set_xticklabels(['0.125','0.5','2'])
    ax.xaxis.set_minor_formatter(NullFormatter())
axes[0].legend(fontsize=8,loc='upper left')
fig.suptitle('Output extension did not pass the frozen stronger-disturbance criterion',fontsize=13)
fig.supxlabel('Shading: selected-case range, not a confidence interval. Full vocabulary and cloned state; median curves do not replace worst-case checks.',fontsize=9)
fig.savefig(OUT/'output_extension_limits.png',dpi=150);plt.close(fig)

def table_range(kind,eps,name,precision='FP32'):
    return next(x for x in ranges if x['precision']==precision and x['map']==kind and x['strength']==eps and x['equation']==name)
decomposition=json.loads((OUT/'CALIBRATION_DECOMPOSITION_SUMMARY.json').read_text())
text='''PROSPECTIVE NORMALIZATION-AWARE RESPONSE EQUATION —12 October2026

Outcome
The internal finite-response equation improved and passed the frozen prospective
criterion on all four held-out question families. The separately tested output
extension did NOT pass its full criterion. Its typical error improved over the
original Fisher baseline, but it did not consistently improve over quadratic
endpoint propagation; strong/worst-case failures remain.

Final selected equation
Keep the module's actual RMS normalization exact; approximate only its SiLU gate
around the unperturbed point. Write N(h)=(1+w)*h/sqrt(mean(h^2)+epsilon_norm),
using the pinned module's real gain and epsilon. Define
    z0=N(h), za=N(h+alpha*v), dz=za-z0,
    g0=Wg*z0, u0=Wu*z0, dg=Wg*dz, du=Wu*dz,
    f0=SiLU(g0), f1=SiLU'(g0), f2=SiLU''(g0).
Then
    predicted_delta_B = Wd{[f0+f1*dg+0.5*f2*dg^2]*(u0+du)-f0*u0},
where products/squares are elementwise. For the whole residual map add alpha*v.
This is a gate approximation with exact normalization geometry, NOT an exact
formula for the full SwiGLU branch. No perturbed sigmoid/output enters its
prediction; gate derivatives are evaluated only at the base point.

Canonical rational candidate retained
With b1=J_B(h)v, b2=H_B(h)[v,v], s0=mean(h^2)+epsilon_norm,
s1=2mean(h*v), s2=mean(v^2), the originally proposed candidate is
    s0/s(alpha) * [alpha*b1+alpha^2*(b2/2+(s1/s0)*b1)].
Its denominator is evaluated directly as a positive squared norm plus the actual
epsilon, avoiding cancellation. Its expansion agrees with ordinary Taylor through
second order. It is exact for the RMS-normalized quadratic-even subcomponent,
and only approximate for full SwiGLU. The prospective results confirm that it
helps, while calibration selected the further gate-expansion improvement.

Fresh finite design and selection
Eight exact new questions, four calibration families and four withheld families:
grammar, logical reasoning, social writing and conceptual comparison. Four fresh
Gaussian directions were orthogonalized off the previous four-direction span;
directions0/1 calibrate,2/3 are held out. Exact prompts, tokenizer IDs, directions,
strengths and source were frozen before any native model evaluation.
The same zero-based block11 MLP and checkpoint Qwen3.5-0.8B revision
2fc06364715b967f1860aea9cf38778875588b17 were retained. The generator stayed frozen,
unquantized and without adapters. Actual multi-token prefixes were evaluated at
prompt-end and the first cached prediction after fixed token760. Both module
branch B and full map h+B were scored, so the identity path cannot hide a bad
branch forecast. Prefixes were not truncated.
Four fixed equations were compared: linear, ordinary quadratic, canonical
rational, and exact normalization with a second-order gate expansion. Only the
four calibration questions/directions0/1 selected the final candidate. Selection
was locked before held-out perturbation outputs. Every equation is retained in
the held-out reports; no test-driven parameter, direction, strength or band tuning.
Derivatives are recomputed at each actual base point; this is not one universal
matrix fitted across questions and no statistical classifier was trained.

Full-vector held-out results
Median / maximum branch relative errors, four held-out questions and two held-out
directions, with both positions/signs (repeated observations, not32 questions):
'''
for eps in [.5,1.25,2.]:
    text+=f'  Strength {eps:g}: '
    text+='; '.join(f"{labels[name]} {table_range('branch',eps,name)['relative_vector_error']['median']*100:.2f}% / {table_range('branch',eps,name)['relative_vector_error']['max']*100:.2f}%" for name in ['quadratic','rational',selected])+'\n'
text+='''
At strength2 the selected whole residual-map error is2.27% median,3.89% maximum.
Branch predicted-magnitude error is0.27% median,0.69% maximum, and direction error
is2.75degrees median,5.14degrees maximum. Full activation-change-vector errors,
not just a convenient response angle, define success. Across the frozen strong
band0.5/1.25/2, each held-out family's median vector error improved about95.2–95.8%
over ordinary quadratic Taylor, on both branch and full map.
All predeclared internal question-level rules passed, including weak-signal
retention. At strengths0.0078125/0.03125 the largest selected FP32 branch error was
0.0166%, near the finite FP32 arithmetic floor; no extrapolation below that floor.
Only the six declared sampled strengths were tested. No error bound is claimed
for arbitrary directions, families, larger amplitudes, layers, models or27B.

Why the improvement helps: calibration-only controlled diagnostics
Ordinary input-space Taylor expands both normalization and gating. At finite
displacement the normalization denominator can change substantially. Keeping this
known geometry exact removes a large source of amplitude error; gate curvature
adds further correction. The fixed2x2 diagnostic, using original frozen weights,
varied linearized/exact normalization and first-/second-order gate expansion.
At strength2 calibration median branch errors were:
'''
last=next(x for x in decomposition if x['strength']==2.)
for key,value in last['equation_errors'].items():text+=f"  {key}: {value['median']*100:.2f}%.\n"
text+='''The interaction vector is nonzero (median norm0.333 at strength2), so the two
corrections cannot be treated as independent additive scalar benefits. These are
calibration diagnostics, not new trained generators or global causal percentages.
FP64 derivative/expansion checks and the exact quadratic-even identity passed.

Representability and native BF16 limits
Every requested and actual rounded input displacement was retained, along with
its projection, off-direction component and zero flags. Actual-input predictions
were computed before observing outputs. The adequate FP32 reference promotes
the exact stored checkpoint values and the same BF16-derived incoming residual.
Native BF16 is still strongly limited for weak edits: at0.0078125 selected branch
error is71.4% median,78.0% maximum; using actual input changes reduces but does
not eliminate this discrepancy. That difference includes internal arithmetic
rounding, not evidence of contraction. No zero-delivered/zero-output case was
silently counted as success. At0.5 native branch error is2.03% median,2.50% max;
at2 it is4.88% median,8.99% max. The smooth equation is not precision agnostic.

Output-distribution extension: executed, with a negative stronger-range result
The full vocabulary was used. The original local baseline is
    alpha^2/2 * Var_p(q), q=actual baseline logit derivative.
The tested extension propagates predicted full residual shifts through the actual
baseline downstream derivative and applies exact softmax to predicted endpoint
logits. Endpoint KL(p_base||p_predicted) is compared with KL(p_base||p_actual).
It is not a sum of local KL costs. Linear, quadratic and canonical-rational
endpoint baselines are retained; selected remains calibration-locked.
Prefill and cached cases used identical histories and correctly cloned actual
native state. Output tests used actual FP32 full-model base states; BF16 endpoint
accuracy is not claimed. There were4 calibration and8 held-out prefill direction
cases, plus2 calibration and4 held-out cached cases. The cached output test used
one held-out direction; the internal MLP test used both. Each output case measured
both signs of0.125/0.5/2. Exact forward-mode AD worked at both positions; finite-difference
calibration and base-context parity passed. No full Fisher/Jacobian was formed.
At strength0.5 the selected scalar-KL median error was10.1% prefill and14.8%
cached, but worst cases were103% and29.9%. Only1/4 held-out families met ALL
the frozen improvement/worst-case requirements; the output primary rule FAILED.
At strength2 selected scalar error rose to48.1% median/126.8% max prefill and
52.8% median/106.0% max cached. At the smaller0.125 strength it was1.44% median/
11.5% max prefill and4.22% median/6.47% max cached. All negative ranges are kept;
the small range does not replace the failed stronger test.
Selected absolute endpoint-KL errors at0.5 range0.000111–0.00905nats prefill and
0.000128–0.00967nats cached. KL(actual||predicted) is also retained; scalar cost
agreement is not a bound on each token probability. The downstream propagation
is still local and may miss finite tail curvature. This study does not uniquely
assign every endpoint error to one cause or prove a reliable strong-output law.

Scope and next unexecuted work
The internal equation improvement is prospectively supported on four fresh
held-out families and two fresh directions at one fixed MLP. The proposed rational
law also received fresh confirmation; it was not relabeled exact for full SwiGLU.
The next separate research target is finite downstream propagation/curvature,
using calibration-only derivation and fresh validation. No extra tail equation,
layer/model transfer, classifier, recovery campaign or scheduled supervision was
launched after these outcomes. These functions predict controlled perturbation
effects, not the cause of an unknown activation trace or a global physical energy.
Known RMSNorm, SwiGLU, Taylor and rational approximation ideas are credited in
LITERATURE.json. Limited searches did not establish novelty; no novelty claim.

Resources and preservation
494 whole-backbone calls,1536 signed local MLP measurements and18 output direction
cases completed in7.87minutes including model load and native export, within the
unchanged45-minute hard limit. Directions/signs/positions/calls are not independent
questions. All owned model workers exited; zero GPU allocation. Research3 and
unrelated sessions/user mounts were untouched. Old energy_response_v1 results and
verified backups are preserved; the old consistency monitor remains stopped.
Exact sources, inputs, predictions, selection lock, observations, caches, hashes
and all failures are privately preserved with verified backups.
'''
(OUT/'REPORT.txt').write_text(text,encoding='utf-8')
status={'date':'2026-10-12','status':'prospective equation-improvement study complete','selected_equation':selected,
    'independent_questions':8,'held_out_questions':4,'held_out_directions':2,'internal_primary_support':r['internal_primary_support'],
    'output_primary_support':r['output_primary_support'],'whole_backbone_calls':494,'signed_MLP_points':1536,'output_direction_cases':18,
    'GPU_allocations':0,'owned_model_workers_remaining':0,'old_monitor_remains_stopped':True,'artifacts':str(OUT)}
(HERE/'STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
for name in ['PLAN.json','LITERATURE.json','EXECUTION_FREEZE.json','equations.py','native_v2.py','maths.py','worker.py','prepare.py','analyze.py','render_report.py','STATUS.json']:
    shutil.copyfile(HERE/name,OUT/name)
for name in ['CALIBRATION_SELECTION_LOCK.json','NUMERICAL_GATE.json','OUTPUT_DERIVATIVE_GATE.json','OUTPUT_ADMISSION.json','EXPORT_RECEIPT.json']:
    shutil.copyfile(ROOT/name,OUT/name)
print(OUT/'REPORT.txt')

"""Make concise scientific artifacts from verified saved-vector summaries; no model calls."""
import os,time
os.environ['OPENBLAS_NUM_THREADS']='2'
BEGIN=time.monotonic()
import json,csv,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
A=Path('C:/Users/farha/Research4/artifacts/coupled_memory_blocks_v1')
R=A/'verified_audit_export_20261015';S=R/'run_20261015_resume1';O=A/'findings_20261015';O.mkdir(exist_ok=False)
read=lambda p:json.loads(p.read_text())
result=read(S/'RESULTS.json');audit=read(R/'audit_20261015/INDEPENDENT_AUDIT.json')
assert result['primary_support'] and audit['primary_support_reconfirmed']
rows=read(S/'FIELD_COMPARISONS.json');controls=read(S/'CONTROL_COMPARISONS.json')
g=read(R/'audit_20261015/FINITE_TWO_AXIS_GEOMETRY.json')
energy=read(S/'NATIVE_ENERGY_HEADS.json');moving=read(S/'MOVING_CONDITIONS.json')
held=[r for r in rows if r['split']=='held'];key=[r for r in held if r['field'] in ['b13_S','b13_output']]
geometry=[r for r in g if r['split']=='held' and r['field']=='b13_output']
stats={'primary_support':True,'held_questions':4,'total_trajectories':24,'held_trajectories':16,'held_full_field_checks':len(held),'all_field_checks':len(rows),'required_resolved_key_fields':len(key),'maximum_key_relative_change_error':max(r['change_error_over_scale'] for r in key),'minimum_key_signal_over_numeric_floor':min(r['native_change_norm']/r['numeric_floor'] for r in key),'maximum_native_force_relative_error':max(r['force_relative'] for r in energy),'native_energy_head_checks':len(energy),'maximum_work_balance_error':max(r['work_error'] for r in energy),'positive_fixed_condition_potential_changes':sum(r['potential_change']>1e-10 for r in energy),'positions':{},'CPU_only':True,'native_backbone_calls':76,'GPU_allocations':0,'original_serialization_fault_preserved':True,'unknown_greedy_diagnostics':1}
for t in [1,2]:
    gg=[r for r in geometry if r['token']==t]
    p={'axis_gain_min':min(x for r in gg for x in r['native']['axis_gains']),'axis_gain_max':max(x for r in gg for x in r['native']['axis_gains']),'angle_min_degrees':min(r['native']['output_axis_angle_degrees'] for r in gg),'angle_max_degrees':max(r['native']['output_axis_angle_degrees'] for r in gg),'median_memory_relative_change_error':float(np.median([r['change_error_over_scale'] for r in held if r['token']==t and r['field']=='b13_S'])),'controls':{}}
    for mode in ['frozenK','frozenKF','reset']:
        p['controls'][mode]={field:float(np.median([r['relative_change_error'] for r in controls if r['split']=='held' and r['token']==t and r['mode']==mode and r['field']==field])) for field in ['b13_S','b13_output']}
    stats['positions'][str(t)]=p
(O/'NUMERIC_FINDINGS.json').write_text(json.dumps(stats,indent=2)+'\n')
with (O/'HELD_TWO_AXIS_GEOMETRY.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['question','token','field','native_axis1_gain','native_axis2_gain','native_angle_degrees','predicted_angle_degrees','native_gram_eigenvalue_ratio','response1_relative_error','response2_relative_error'])
    for r in g:
        if r['split']=='held':w.writerow([r['question'],r['token'],r['field'],*r['native']['axis_gains'],r['native']['output_axis_angle_degrees'],r['predicted']['output_axis_angle_degrees'],r['native']['gram_eigenvalue_ratio'],*r['response_vector_relative_errors']])
labels=['Coin location','Temperature','Weekday','Number sequence'];pids=['r21_held_envelope','r21_held_greenhouse','r21_held_weekday','r21_held_sequence']
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'font.family':'DejaVu Sans'})
fig,ax=plt.subplots(1,2,figsize=(11.7,4.7));colors=['#176C8E','#CC7017'];x=np.arange(4)
for t,color,offset in [(1,colors[0],-.12),(2,colors[1],.12)]:
    yy=[];lo=[];hi=[]
    for pid in pids:
        rr=[r['change_error_over_scale']*100 for r in held if r['question']==pid and r['token']==t and r['field']=='b13_S'];yy.append(np.median(rr));lo.append(np.median(rr)-min(rr));hi.append(max(rr)-np.median(rr))
    ax[0].errorbar(x+offset,yy,yerr=[lo,hi],fmt='o',color=color,capsize=3,label=f'Token {t}')
    vals=[r for r in geometry if r['token']==t];by={r['question']:r for r in vals}
    for j in [0,1]:ax[1].scatter(x+offset,[by[pid]['native']['axis_gains'][j] for pid in pids],color=color,marker='o' if j==0 else 's',s=42,label=f'Token {t}' if j==0 else None)
ax[0].set(title='Prediction of downstream memory change',ylabel='Error / measured change (%)',yscale='log',xticks=x,xticklabels=labels)
ax[0].axhline(100,color='#777777',ls=':',label='Frozen-driver control (100%)')
ax[0].set_ylim(.00065,180);ax[0].legend(fontsize=9,loc='upper right');ax[0].grid(axis='y',alpha=.16)
ax[1].set(title='Size of the downstream residual response',ylabel='Response norm / delivered nudge norm',xticks=x,xticklabels=labels,ylim=(0,1.08))
ax[1].axhline(1,color='#999999',ls=':',lw=1);ax[1].legend(fontsize=9);ax[1].grid(axis='y',alpha=.16)
for a in ax:a.tick_params(axis='x',labelrotation=18)
fig.suptitle('Two adjacent memory blocks: four held questions',fontweight='bold',y=.99)
fig.text(.02,.015,'Two fixed random axes, signed 0.125 nudges; controlled two-token boundary inputs. Original weights retained. No semantic or full-spectrum claim.',fontsize=9)
fig.tight_layout(rect=(0,.065,1,.94));fig.savefig(O/'COUPLED_RESPONSE_FINDINGS.png',dpi=180);fig.savefig(O/'COUPLED_RESPONSE_FINDINGS.svg');plt.close(fig)
plain=f'''What we tested
We connected mathematical descriptions of two complete neighboring Qwen blocks.
The second block received the first block's calculated output. At the following
token, both blocks used their own calculated memories. All predictions were
sealed before native steered measurements. No changed native intermediate
states and no per-question fitted coefficients were predictor inputs.

What happened
All four held questions passed at both controlled token positions. All 448
held field comparisons passed the original numerical checks, and all 64
required downstream memory/output changes were well above numerical noise.
Their largest relative prediction error was{stats['maximum_key_relative_change_error']*100:.6f}%.
All 1,920 native memory-head force/work checks passed an independent dense
matrix calculation. The original report-writing failure remains preserved;
the resumed comparison changed no forecasts or scientific thresholds.

Holding the second block's internal response matrix fixed caused large errors
in its predicted memory change: about 100.5% at the first token and 71.8% at
the next, compared with about 0.0017% and 0.0012% for the complete calculation.
This illustrates why the component must respond to its current conditions.

What the geometry says
For these two random axes, the downstream first-token residual response had
norm{stats['positions']['1']['axis_gain_min']:.3f}–{stats['positions']['1']['axis_gain_max']:.3f} times the delivered nudge.
At the next controlled token it was{stats['positions']['2']['axis_gain_min']:.3f}–{stats['positions']['2']['axis_gain_max']:.3f} times that original nudge.
Thus the effect persists through memory, with a context-dependent size.
The two response axes stayed near perpendicular (about 87–93 degrees across
both positions). Near perpendicular random axes are not by themselves evidence
of a special semantic organization in a high-dimensional space.

What this does and does not establish
This verifies an internally connected, nonlinear subsystem calculation on
new questions. It retains the original trained matrices and nonlinearities;
its accuracy is not compression, a cheaper MLP or a newly discovered law.
The normal starting memories and upstream boundary signals were supplied
per question. The next token was fixed to the normal continuation. Among 23
available greedy-token diagnostics none changed; one calibration diagnostic
was lost at the original serialization failure and is explicitly unknown.
This test does not establish autonomous free generation, semantic direction
meanings, unknown-steering detection or a global conserved physical energy.

The worthwhile larger ambition
Find a small, meaningful geometric description of what information is stored,
how it moves, and why it changes answers. A suitable next narrow replication
is a controlled binary concept across fresh contexts, connecting causal
concept geometry to the same verified block dynamics. The larger proposed
goal is described in AMBITIOUS_RESEARCH_GOAL_20261015.txt; no follow-up run
is automatically queued. The experiment used CPU, 76 combined native model
calls including six context reconstructions, and zero GPU allocations.
'''
(O/'PLAIN_LANGUAGE_FINDINGS.txt').write_text(plain)
derivation='''Exact connected subsystem calculation

At each block and token, compute the original input RMS normalization,
projections, four-step convolution, nonlinear gates and key/query norms.
Given current key k, value v, write gate beta and retention gamma, define
f=beta*k*v^T and K=(1-gamma)I+gamma*beta*k*k^T.
The recurrent state satisfies Snew=Sold+f-K*Sold, independently per head.
Read the state with the calculated query, apply the original gated RMS
normalization/output projection, and compute the original MLP and both
residual additions. Feed the entire computed result into the next block.
Carry both calculated recurrent and convolution states to the next token.

Subtracting a normal trajectory from an intervened trajectory gives
DeltaSnew=(I-K0)*DeltaSold+Deltaf-DeltaK*Sold0-DeltaK*DeltaSold.
This is an exact finite identity, including the last interaction term.
Forcing terms, stiffness changes, and carried memory are all calculated
from the equation model's own trajectories. All384 archived term-vector
SHA256 hashes were regenerated and verified independently.

For fixed current drivers, let Psi(S)=<S,K*S>/2-<f,S> and step=Snew-Sold.
Define W=<f,step>, DeltaU=<K*Sold,step>+<step,K*step>/2,
and D=||step||^2-<step,K*step>/2. The update implies W=DeltaU+D.
Here0<=beta<=1,0<=gamma<=1 and ||k||<=1; K is symmetric positive
semidefinite with eigenvalues at most1. The exact fixed-driver update
therefore decreases Psi. Native FP32 residuals are checked explicitly.
These are mathematical work/storage/relaxation terms, not physical heat.

The next token changes k,v,beta,gamma. With a fixed zero-at-S=0 convention,
Psi_t(S_t)-Psi_previous(S_previous) splits exactly into the change in
conditions at S_previous plus the current-step relaxation. Summing the
two memory potentials is bookkeeping under that convention; it does not
establish physical interface work or a single conserved transformer energy.

The response geometry uses signed central secants at0.125. Two full1024
residual responses or two full262144-coordinate memory responses define
a2x2 Gram matrix. Generalized eigenvalues use the Gram matrix of the actual
FP32-delivered input axes, not an assumed perfectly orthogonal input.
Their square roots are gains restricted to this two-dimensional input plane.
This is not an infinitesimal Jacobian, a full1024-dimensional spectrum,
a semantic map, or evidence for one universal metric across contexts.
'''
(O/'DERIVATION_AND_LIMITS.txt').write_text(derivation)
elapsed=time.monotonic()-BEGIN
(O/'ANALYSIS_RECEIPT.json').write_text(json.dumps({'new_backbone_forwards':0,'GPU_allocations':0,'seconds':elapsed,'input_audit_export_sha256':hashlib.sha256((A/'R4_COUPLED_MEMORY_AUDIT_EXPORT_20261015.zip').read_bytes()).hexdigest()},indent=2)+'\n')
print(json.dumps(stats,indent=2));print('ANALYSIS_SECONDS',elapsed)

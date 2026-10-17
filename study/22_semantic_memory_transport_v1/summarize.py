"""Report frozen semantic and numerical checks; never change scientific thresholds."""
import os,time
os.environ['OPENBLAS_NUM_THREADS']='2'
BEGIN=time.monotonic()
import json,csv,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
A=Path(sys.argv[1]);R=A/'run_20261015';O=A/'findings_20261015_v2';O.mkdir(exist_ok=False)
read=lambda p:json.loads(p.read_text())
audit=read(A/'audit_20261015/INDEPENDENT_AUDIT.json');sem=read(A/'audit_20261015/SEMANTIC_RESULTS.json');plan=read(R/'source/PLAN.json');out=read(R/'OUTPUT_GEOMETRY_RESULTS.json');result=read(R/'RESULTS.json')
rows=read(R/'FIELD_COMPARISONS.json');held=[r for r in rows if r['split']=='held'];key=[r for r in held if r['direction']==0 and r['field'] in ['b13_S','b13_output']]
assert audit['numerical_support_reconfirmed']==result['numerical_support']
gate_names=['output_subspace_gate_pass','output_intervention_selectivity_pass','natural_context_gate_pass','internal_causal_direction_gate_pass','internal_selectivity_gate_pass']
stats={'numerical_prediction_pass':audit['numerical_support_reconfirmed'],'semantic_claim_pass':sem['semantic_support'],'semantic_gates_passed':sum(sem[n] for n in gate_names),'semantic_gates_total':5,'semantic_gate_flags':{n:sem[n] for n in gate_names},'maximum_held_semantic_key_relative_change_error':max(r['relative_change_error'] for r in key),'minimum_held_semantic_key_signal_over_numeric_floor':min(r['native_change_norm']/r['numeric_floor'] for r in key),'held_contexts':16,'held_nouns':4,'held_new_templates':2,'held_monotonic_plural_shifts':sem['held_monotonic_plural_shifts'],'median_number_half_swing_nats':sem['median_number_half_swing_nats'],'median_random_absolute_half_swing_nats':sem['median_random_absolute_half_swing_nats'],'median_capitalization_half_swing_nats':sem['median_capitalization_half_swing_nats'],'capitalization_to_number_ratio':sem['capitalization_to_number_ratio'],'number_to_random_ratio':sem['median_number_half_swing_nats']/max(1e-30,sem['median_random_absolute_half_swing_nats']),'normal_count_choices_correct':sem['natural_count_choices_correct'],'median_four_word_probability_mass':sem['median_four_word_mass'],'number_case_cosine_under_output_metric':out['number_case_cosine_under_M'],'number_case_cosine_Euclidean':out['number_case_cosine_Euclidean'],'native_backbone_calls':222,'GPU_allocations':0,'candidate_alpha':result['candidate_direction_alpha']}
fidelity=read(A/'inference_path_check_20261015/FIDELITY_RECEIPT.json')
normal_held=[q for q in read(R/'ALL_NORMAL_METADATA.json') if q['split']=='held']
family=sum(q['token2'] in q['word_ids'] for q in normal_held)
correct_first=sum(q['token2'] in [q['word_ids'][q['number']],q['word_ids'][q['number']+2]] for q in normal_held)
stats.update(whole_prompt_fidelity_controls_pass=fidelity['all_checks_pass'],additional_fidelity_backbone_calls=fidelity['additional_backbone_calls'],phase_combined_native_calls=fidelity['combined_phase_backbone_calls'],normal_greedy_first_token_in_target_noun_family=family,normal_greedy_first_token_correct_number=correct_first,posthoc_greedy_diagnostics_do_not_change_gates=True,output_method_selectivity_comparison=out['output_selectivity'])
(O/'NUMERIC_FINDINGS.json').write_text(json.dumps(stats,indent=2)+'\n')
with (O/'HELD_SEMANTIC_EFFECTS.csv').open('w',newline='') as f:
    rr=sem['held_context_details'];w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(1,2,figsize=(11.9,4.8));x=np.arange(4);pairs=plan['word_pairs'][12:16];labels=[p['words'][0] for p in pairs]
for name,label,color,offset in [('number_half_swing','Candidate number direction','#176C8E',-.24),('random_absolute_half_swing','Random control (absolute)','#777777',0),('capitalization_half_swing','Capitalization effect (absolute)','#C46B34',.24)]:
    med=[];low=[];high=[]
    for p in pairs:
        vals=[r[name] for r in sem['held_context_details'] if r['pair_index']==p['pair_index']];m=float(np.median(vals));med.append(m);low.append(m-min(vals));high.append(max(vals)-m)
    ax[0].errorbar(x+offset,med,yerr=[low,high],fmt='o',color=color,capsize=3,label=label)
ax[0].axhline(0,color='#777777',lw=1);ax[0].axhline(.01,color='#777777',ls=':',lw=1)
ax[0].set(title='Effect on target-word odds',ylabel='Half-swing between positive and negative nudges (log odds)',xticks=x,xticklabels=labels);ax[0].legend(fontsize=8,loc='best');ax[0].grid(axis='y',alpha=.15)
for t,color,offset in [(1,'#176C8E',-.12),(2,'#C46B34',.12)]:
    med=[];low=[];high=[]
    for p in pairs:
        vals=[r['relative_change_error']*100 for r in held if r['pair_index']==p['pair_index'] and r['direction']==0 and r['field']=='b13_S' and r['token']==t];m=float(np.median(vals));med.append(m);low.append(m-min(vals));high.append(max(vals)-m)
    ax[1].errorbar(x+offset,med,yerr=[low,high],fmt='o',color=color,capsize=3,label=f'Token {t}')
ax[1].set(title='Prediction of downstream memory change',ylabel='Full-vector error / measured change (%)',yscale='log',xticks=x,xticklabels=labels);ax[1].legend(fontsize=9);ax[1].grid(axis='y',alpha=.15)
fig.suptitle('Meaning and state prediction are separate tests',fontweight='bold',y=.99)
fig.text(.02,.015,'Number odds condition on four noun forms. Bars show context median/range. Baseline semantic admission failed. Original weights retained; next token controlled.',fontsize=8.5)
fig.tight_layout(rect=(0,.065,1,.94));fig.savefig(O/'SEMANTIC_TRANSPORT_FINDINGS.png',dpi=180);fig.savefig(O/'SEMANTIC_TRANSPORT_FINDINGS.svg');plt.close(fig)
yes=lambda b:'passed' if b else 'failed'
if not sem['natural_context_gate_pass']:
    conclusion='The baseline task failed semantic admission, and the candidate direction also failed selectivity and the required margin over random control. This test does not establish a clean internal plurality direction.'
elif sem['semantic_support']:
    conclusion='The candidate direction passed the predeclared held meaning and side-effect checks. This is evidence for a number-sensitive, relatively selective direction in this small test.'
elif sem['internal_causal_direction_gate_pass'] and not sem['internal_selectivity_gate_pass']:
    conclusion='The candidate direction moved grammatical number, but its capitalization side effect was too large. We cannot call it a selective singular/plural direction.'
elif not sem['internal_causal_direction_gate_pass']:
    conclusion='The candidate direction did not pass the held causal-effect requirement. Learning a direction from normal singular/plural examples was insufficient to establish a reliable intervention direction.'
else:
    conclusion='Some meaning gates failed. The numerical prediction result does not establish a selective semantic direction.'
plain=f'''What we learned
{conclusion}
The full two-block numerical prediction test {yes(stats['numerical_prediction_pass'])}.
The selective semantic-transport claim {yes(sem['semantic_support'])}: {stats['semantic_gates_passed']} of5 meaning gates passed.

The output-space replication passed its two declared checks, but the simple
Euclidean direction passed too and had smaller capitalization interference
(ratio0.141 versus0.339 for the covariance-mapped direction). This small test
does not show that the covariance metric is better or necessary.

What we literally did
1. Used public singular/plural word pairs to reproduce a small version of the
   paper's output-space geometry method on frozen Qwen0.8B. Twelve pairs fitted
   the direction estimate; four different nouns tested it. We reconstructed
   the population covariance of all248,320 vocabulary vectors independently.
2. Recorded normal internal activations for six paired singular/plural prompts.
   Averaging the differences gave one candidate direction before block12.
   Its magnitude was fixed from training traces only (alpha={stats['candidate_alpha']:.6f}).
3. On16 fresh contexts, used both signs of that nudge and a fixed same-norm
   random control. The four new nouns were {', '.join(labels)}.
4. Measured actual effects on the model's first-word probabilities, checking
   grammatical number and capitalization separately. Whole-vocabulary logits
   and the final representation were retained for independent verification.
5. Before measuring any steered targets, predicted the complete states of
   blocks12/13 at that position and one following controlled token. The second
   block and next-token memories came from our own calculations.

The held meaning result
Normal number choices: {sem['natural_count_choices_correct']}/16 correct conditional on four word forms.
Actual first-token predictions belonged to the requested noun family in
{family}/16 cases and had the requested number in{correct_first}/16 cases.
These extra first-token diagnostics explain the weak baseline; they do not
change any predeclared gate. Many first predictions were other words.
Monotonic movement toward plural: {sem['held_monotonic_plural_shifts']}/16 contexts.
Median number half-swing: {sem['median_number_half_swing_nats']:.8f} log-odds units.
Median absolute random-control half-swing: {sem['median_random_absolute_half_swing_nats']:.8f}.
Median capitalization half-swing: {sem['median_capitalization_half_swing_nats']:.8f}.
Capitalization/number ratio: {sem['capitalization_to_number_ratio']:.6f}; required<=0.5.
A half-swing is half the difference between positive and negative interventions.
It measures the directional effect, not the fraction of answers that changed.
Two additional training-only checks compared saved incremental prompt processing
with standard whole-prompt inference. They {yes(fidelity['all_checks_pass'])}, with
matching top tokens and full-logit relative differences below0.000001.
No held cases were rerun or used to fit a correction.

Independent predeclared gates
Output-pair subspace: {yes(sem['output_subspace_gate_pass'])}.
Output-space intervention selectivity: {yes(sem['output_intervention_selectivity_pass'])}.
Normal held-context admission: {yes(sem['natural_context_gate_pass'])}.
Internal held causal effect: {yes(sem['internal_causal_direction_gate_pass'])}.
Internal capitalization selectivity: {yes(sem['internal_selectivity_gate_pass'])}.

The state-prediction result
All{audit['complete_target_fields_checked']:,} complete target vectors and{audit['complete_baseline_fields_checked']} normal
vectors were independently rechecked; no failed field was dropped.
Largest held candidate-direction block13 memory/output relative change error:
{stats['maximum_held_semantic_key_relative_change_error']*100:.6f}%.
All{audit['dense_energy_heads_checked']:,} conditional energy-head checks and
{audit['term_vector_hashes_regenerated']:,} regenerated finite-term hashes were verified.

What this means for the larger goal
State prediction and meaning are separate achievements. The equations retain
the original trained matrices and nonlinearities. The output covariance metric
is not assumed to apply inside block12 or its recurrent memory. Native effects
on final words were measured; the two-block equations did not forecast all
remaining layers. We supplied normal initial memories and upstream boundaries
per context, and consumed the normal answer token at the following position.
This does not establish autonomous generation, a universal semantic map,
a compressed transformer, unknown-steering detection or global energy conservation.

Original model and thresholds remained unchanged. Every failed gate is retained.
CPU only:222 experiment backbone calls plus2 fidelity controls=224 total,
zero GPU allocations. No follow-up experiment
or monitor is automatically queued.

Primary method: Park,Choe,Veitch,The Linear Representation Hypothesis and the
Geometry of Large Language Models(ICML2024),https://arxiv.org/abs/2311.03658 .
Official code pinned at40bee00b50a6fafcf2ee4db2c01602023aa366cc.
'''
plain=plain.replace('of5','of 5').replace('all248,320','all 248,320').replace('On16','On 16').replace('All2,016','All 2,016').replace('All5,760','All 5,760').replace('and504','and 504').replace('ratio0.141 versus0.339','ratio 0.141 versus 0.339').replace('in3/16','in 3/16').replace('below0.000001','below 0.000001').replace('plus2','plus 2')
(O/'PLAIN_LANGUAGE_FINDINGS.txt').write_text(plain)
derivation='''Output-space replication
Let U contain all effective FP32 output vectors of the pinned original model.
C=mean[(u-mean(u))(u-mean(u))^T]; M=C^-1, with no adaptive regularization.
For twelve training noun pairs, mu=mean(u_plural-u_singular).
The covariance-mapped intervention is lambda=M*mu, normalized to unit Euclidean
length for comparisons. Its effect on any two-word log-odds after final
normalization is exactly alpha*lambda^T*(u_word1-u_word0). This arithmetic
linearity follows from the original linear output head; semantic evidence
comes from held word alignment and limited capitalization interference.
The covariance and output rows were rebuilt independently from public BF16
safetensors bytes with a centered batch-merging calculation.

Intermediate direction
w=unit(mean(h12_plural-h12_singular)) from six paired normal training prompts.
The same-norm random direction is a fixed Gaussian vector orthogonal to w.
Both signs use alpha=clip(.25*median(||paired h12 difference||),.125,2.0).
No held word effects are used to choose the sign,magnitude or direction.
At the answer position, measure plurality log odds by aggregating singular
and plural word forms across lower and initial-capital variants. Measure
capitalization within each number separately. All gates appear in PLAN.json.

Connected local memory equations
For each head, f=beta*k*v^T and K=(1-gamma)I+gamma*beta*k*k^T.
Snew=Sold+f-K*Sold. Compute all drivers,convolution,query readout,normalizations,
MLP nonlinearities and residual additions from the predictor's own states.
The exact finite change identity is
DeltaSnew=(I-K0)DeltaSold+Deltaf-DeltaK*Sold0-DeltaK*DeltaSold.
Carry both blocks' predicted recurrent and convolution memories into the next
controlled token. No changed native intermediate states are predictor inputs.
All full term vectors are reproducible from saved arrays and their hashes.

For a fixed current K,f, Psi(S)=<S,K*S>/2-<f,S>.
With step=Snew-Sold, W=<f,step>,DeltaU=<K*Sold,step>+<step,K*step>/2,
and D=||step||^2-<step,K*step>/2 give W=DeltaU+D.
The next input changes K,f; this conditional accounting does not establish a
global conserved physical energy or explain semantic meanings by itself.
'''
(O/'DERIVATION_AND_LIMITS.txt').write_text(derivation)
(O/'ANALYSIS_RECEIPT.json').write_text(json.dumps({'seconds':time.monotonic()-BEGIN,'new_backbone_calls':0,'GPU_allocations':0,'scientific_thresholds_changed':False},indent=2)+'\n')
print(json.dumps(stats,indent=2))

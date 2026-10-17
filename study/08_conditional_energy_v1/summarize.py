"""Report every fixed conditional-energy candidate, including failed holdouts."""
from pathlib import Path
import csv
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).parent
OUT=Path('C:/Users/farha/Research4/artifacts/conditional_energy_v1/analysis_20261012')
result=json.loads((OUT/'RESULTS.json').read_text())
for name in ['consistency_monitor.py','STATUS.json','NEXT_STEP.json','summarize.py']:
    shutil.copyfile(HERE/name,OUT/name)
shutil.copyfile(HERE.parent/'04_activation_rotation/CURRENT_DIRECTION.md',OUT/'CURRENT_DIRECTION.md')
families=['physical_explanations','numerical_explanations','simulated_worker','matched_records']
kinds=['unconditional_diagonal','predictive_innovation','strain_compatibility']
summaries=result['primary_native_family_summary']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
for ax,field,total,title,cmap in [
    (axes[0],'BASE_false_positives','BASE_total','Untreated runs falsely flagged','Reds'),
    (axes[1],'modified_detected','modified_total','Modified runs flagged','Blues')]:
    a=np.array([[next(r for r in summaries if r['model']==kind and r['family']==family)[field]/
        next(r for r in summaries if r['model']==kind and r['family']==family)[total]
        for family in families] for kind in kinds])
    ax.imshow(a,vmin=0,vmax=1,cmap=cmap,aspect='auto')
    for i,kind in enumerate(kinds):
        for j,family in enumerate(families):
            r=next(r for r in summaries if r['model']==kind and r['family']==family)
            ax.text(j,i,f'{r[field]}/{r[total]}',ha='center',va='center',color='white' if a[i,j]>.65 else 'black',fontsize=13)
    ax.set(xticks=range(4),xticklabels=['Physical','Numerical','Worker','Records'],
        yticks=range(3),yticklabels=['Global reference','Conditional prediction','Strain compatibility'],title=title)
    ax.tick_params(length=0)
fig.suptitle('Entire scenario families withheld: none of the three energy rules generalized reliably',fontsize=13)
fig.supxlabel('Native primary results;14 reused questions. Each fold fits/calibrates on other families. Norm-only modifications included.',fontsize=9)
fig.savefig(OUT/'family_holdout_results.png',dpi=150);plt.close(fig)

lines=['CONDITIONAL ENERGY AND COMPUTATION CONSISTENCY —12 October2026','',
'Outcome: no general ordinary-residual-only energy detector was validated.',
'A trusted same-boundary monitor separately distinguishes modifications, with a different input-access requirement.','',
'Why mechanics and neuroscience are useful here',
'A loaded beam can deform normally. Its displacement alone does not identify an extra force.',
'Linear finite-element strain energy is0.5*u^T*K*u; total potential also contains the external-load term.',
'Equilibrium and compatibility compare response with the governing law and boundary/load conditions.',
'In our analogy the question/context supplies conditions, and model computation supplies the expected response.',
'Neuroscience predictive coding similarly compares actual activity with predictions and weights errors by precision.',
'The useful transfer is a conditional discrepancy, not merely a large activation norm.',
'These analogies do not establish physical stiffness, a true variational posterior, a conserved quantity,',
'or a Lyapunov/global downhill potential for the pretrained language model.','',
'New statistical test',
'Four coarse families: physical explanations(2 prompts), numerical explanations(2), simulated worker(4),',
'and all matched records(6). In each outer fold one family is TEST, the next is calibration, and the other',
'two are TRAIN. Related wording and continuation replicas never cross families. Only TRAIN BASE traces',
'fit laws and uncertainty; the maximum calibration-family BASE score sets the threshold before TEST.',
'There are only2–6 calibration contexts, so no reliable low-false-alarm population guarantee follows.',
'This is exploratory cross-validation over14 previously inspected questions, not a new prospective test.',
'Every candidate observes only residuals24x8x1024 at decision time. No paired BASE, token ID, prompt text,',
'filename, intervention vector or pre_injection array enters a statistical score. Position0 and positions1..7',
'have separate references; the final fixed score averages all8.','',
'Candidates',
'1. Unconditional diagonal energy reproduces the earlier global-mean anomaly idea within each fold.',
'2. Conditional prediction energy forecasts normalized downstream changes from unaffected blocks8/9.',
'3. Strain compatibility forecasts downstream adjacent-layer log-norm ratios and angles from upstream ones.',
'The third representation is invariant to common positive scaling and common orthogonal basis changes.',
'That invariance does not guarantee invariance to a different question or arbitrary layerwise transformations.',
'Both conditional laws use fixed affine ridge and uncertainty estimated from whole-prompt-held-out TRAIN',
'prediction errors, avoiding artificially certain in-sample fits. No penalty, layer, phase or cutoff was tuned.','',
'Primary native cross-fit counts (different frozen fit/threshold in each fold; not one deployed detector)']
for kind in kinds:
    r=next(r for r in summaries if r['model']==kind and r['family']=='ALL_NATIVE')
    lines.append(f"{kind}: false alarms{r['BASE_false_positives']}/{r['BASE_total']}; modified detections{r['modified_detected']}/{r['modified_total']}.")
lines += ['',
'The conditional predictor has fewer false alarms in some families but still flags every untreated matched',
'record when that entire family is absent from fitting/calibration. It misses all eight modified worker runs.',
'The strain score also fails: four of six untreated matched records are flagged and random2 is missed6/6.',
'All three generalization rules therefore fail. Better ranking within a family is not a transferable threshold.',
'Original seeds are correlated TEST-only supplements. All their per-trace scores are retained in TRACE_SCORES.csv.',
'Forty-six modified native traces include eight+v/eight random original traces and thirty matched interventions;',
'six of the latter are norm-only controls. Modifications are not all independent questions or behavioral successes.','',
'A concrete mechanics-inspired route with trusted instrumentation',
'Define r=observed output - expected unmodified output at the SAME computation boundary, with the same',
'full input/context/cache. A global conditional constraint energy can sum0.5*r^T*P*r over monitored blocks',
'and steps, with positive-definite P. Ordinary deterministic computation satisfies its constraints; an additive',
'change at a monitored boundary has nonzero residual. This is not the energy of raw h and not a claim',
'that the network itself performs gradient descent on this constructed global constraint energy.',
'The existing matched collector saves the unmodified block10 result immediately before its intervention',
'hook and records the output after it. Using0.5*||after-before||^2 with a fixed zero threshold gives:',
'6/6 untreated traces correctly unchanged;30/30 modified traces detected across6 questions/3 wording pairs.',
'Every one of the28 recorded positions agrees. BASE energy is exactly0; directional-update energies are',
'approximately0.125; even the smaller norm-only modifications are nonzero. Full values are saved separately.',
'This verifies known-site instrumentation/receipts, NOT a blind activation-only classifier, and is not',
'empirical proof of general detection across unseen sites, models, strengths or arbitrary steering methods.',
'It measures a modification, not its meaning, intent, harmfulness or behavioral effect.',
'The standalone consistency_monitor.py accepts expected and observed outputs; it requires no learned fit',
'or second unsteered inference. A trusted pre-hook capture can supply expected; an unmodified shadow',
'block could instead recompute it if complete state is available. The current archives lack full caches/',
'recurrent state/prompt-layer tensors needed to validate such a shadow replay. Never confuse an ordinary',
'layer update with an injection by comparing neighboring layers as though the network should be constant.',
'Future shadow replay needs cloned state, matching kernels/precision and a predetermined numeric tolerance.',
'Default exact equality in the prototype is restricted to copied native pre/post values.','',
'Next step and limits',
'The main requested ordinary-residual-only problem remains unresolved. Broader BASE conditional calibration',
'and a fresh scenario/vector-held-out experiment are needed before deploying a statistical score. The prior',
'known+v classifier passed16/16 fresh traces under its restricted contract; that result is unchanged and',
'does not prove arbitrary-direction detection. No additional metric/penalty search follows this failed audit.',
'If trusted expected-output monitoring is acceptable, validate that direct consistency architecture first on',
'fresh scenarios, multiple monitored sites and no-op/numerical controls. It is a genuinely different access contract.',
'If two different causes produce the same observed residual trace, no function of that trace can identify',
'which cause occurred perfectly. Finite-element and neuroscience analogies do not remove that information limit.','',
'Resources and integrity',
'Zero new GPU/native calls. R4 GPU was already released and was not reconnected. Other-project GPU and',
'user CPU Drive mount were not operated.248 original/matched native manifest members verified unchanged.',
'Fresh eight-question panel unaccessed; its seven decode positions remain geometrically unscored.',
'Strain scale/basis controls pass; standalone boundary monitor agrees36/36; ridge primal/dual agree<1e-10.',
'All prior failures and frozen detector outcomes remain unchanged.','',
'All primary native per-family results']
for r in summaries:
    if r['family']!='ALL_NATIVE':
        lines.append(f"{r['model']} / {r['family']}: false alarms{r['BASE_false_positives']}/{r['BASE_total']}; detections{r['modified_detected']}/{r['modified_total']}; per-arm={json.dumps(r['per_condition'])}")
lines += ['','Primary research references']
for p in json.loads((HERE/'LITERATURE.json').read_text())['sources']:
    lines += [p['title'],p['url']]
(OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(OUT/'REPORT.txt')

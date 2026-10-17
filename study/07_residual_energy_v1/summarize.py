"""Render the exploratory scalar-energy findings without fitting new candidates."""
import csv
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
OUT = Path('C:/Users/farha/Research4/artifacts/residual_energy_v1/analysis_20261012')
result = json.loads((OUT/'RESULTS.json').read_text())
rows = list(csv.DictReader((OUT/'PROMPT_SCORES.csv').open()))
for r in rows:
    for k in ['base', 'steered', 'delta', 'relative_change_percent']:
        r[k] = float(r[k])

def selected(dataset, condition, phase='all8', kind='diagonal'):
    return [r for r in rows if r['dataset']==dataset and r['cohort']=='native' and
        r['condition']==condition and r['phase']==phase and r['energy']==kind and r['region']=='global']

old = selected('original_TEST', 'existing_vector')
matched = selected('older_matched_OOD', 'plus_v')
status = json.loads((HERE/'STATUS.json').read_text())
status['native_greedy_diagonal_mean_plus_change_percent'] = float(np.mean([r['relative_change_percent'] for r in old]))
status['older_matched_diagonal_mean_plus_change_percent'] = float(np.mean([r['relative_change_percent'] for r in matched]))
status['decode7_diagonal_mean_plus_change_percent'] = {
    ds:float(np.mean([r['relative_change_percent'] for r in selected(ds, arm, 'decode7')]))
    for ds, arm in [('original_TEST', 'existing_vector'), ('older_matched_OOD', 'plus_v')]}
(HERE/'STATUS.json').write_text(json.dumps(status, indent=2)+'\n')
for name in ['LITERATURE.json', 'STATUS.json', 'NEXT_STEP.json', 'summarize.py']:
    shutil.copyfile(HERE/name, OUT/name)

plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout='constrained')
colors = ['#1565c0', '#d96919']
for ds, arm, color, label, start in [
    ('original_TEST', 'existing_vector', colors[0], 'Original TEST', 0),
    ('older_matched_OOD', 'plus_v', colors[1], 'Matched panel', 5)]:
    a = selected(ds, arm)
    for j, r in enumerate(a):
        x = start+j
        axes[0].plot([x, x], [r['base'], r['steered']], color=color, alpha=.6)
        axes[0].scatter(x, r['base'], color=color, marker='o', facecolors='white', s=35)
        axes[0].scatter(x, r['steered'], color=color, marker='^', s=35)
    axes[0].plot([], [], color=color, label=label)
axes[0].set(xlabel='Question (4 original + 6 matched)', ylabel='Global diagonal energy',
    title='Context shifts the baseline strongly', xticks=[1.5, 7.5], xticklabels=['Original', 'Matched'])
axes[0].text(.02, .98, 'Circles: BASE; triangles: +v\nTRAIN BASE mean = 1', transform=axes[0].transAxes, va='top', fontsize=9)
axes[0].legend(loc='lower right', fontsize=9)
arms = ['plus_v', 'minus_v', 'random1', 'random2', 'norm_only']
for j, arm in enumerate(arms):
    values = [r['relative_change_percent'] for r in selected('older_matched_OOD', arm)]
    axes[1].bar(j, np.mean(values), color='#7796b5', width=.65)
    axes[1].scatter(np.linspace(j-.17, j+.17, len(values)), values, s=18, color='#142d49')
axes[1].set(xticks=range(5), xticklabels=['+v', '-v', 'Rand1', 'Rand2', 'Norm'],
    ylabel='Paired energy change (%)', title='Directional perturbations raise energy')
axes[1].text(.02, .98, 'Matched panel, first8 positions\nDots: six questions (three wording pairs)',
    transform=axes[1].transAxes, va='top', fontsize=9)
axes[1].set_ylim(0, 25)
for j, (ds, arm, color, label) in enumerate([
    ('original_TEST', 'existing_vector', colors[0], 'Original'),
    ('older_matched_OOD', 'plus_v', colors[1], 'Matched')]):
    values = [np.mean([r['relative_change_percent'] for r in selected(ds, arm, phase)])
        for phase in ['prefill', 'decode7']]
    x = np.arange(2)+(j-.5)*.32
    axes[2].bar(x, values, width=.3, color=color, label=label)
    for xx, value in zip(x, values):
        axes[2].annotate(f'{value:.1f}%', (xx, value), xytext=(0, 4), textcoords='offset points', ha='center', fontsize=9)
axes[2].set(xticks=[0, 1], xticklabels=['First position', 'Next 7 positions'],
    ylabel='Paired energy change (%)', title='The first position dominates', ylim=(0, 195))
axes[2].legend(loc='upper right', fontsize=9)
fig.suptitle('Residual energy audit: a consistent paired shift, with substantial context and phase dependence', fontsize=13)
fig.savefig(OUT/'energy_audit.png', dpi=150)
plt.close(fig)

table = []
for r in result['primary_global_all8']:
    if r['cohort']=='native':
        table.append(f"{r['dataset']:20s} {r['condition']:16s} {r['energy']:10s} "
            f"increases={r['paired_energy_increases']}/{r['prompts']} "
            f"mean change={r['mean_relative_change_percent']:+.3f}% "
            f"within-panel AUROC={r['AUROC_high_energy_positive']:.3f}")
text = '''RESIDUAL ENERGY AUDIT — 12 October 2026

Assessment
We can define and monitor a useful global scalar from residual activations.
Today's result is an exploratory statistical energy measurement, not evidence
that the whole pretrained Qwen generator minimizes one global potential.
R4's GPU was already disconnected. No GPU/model calls occurred; the temporary
runtime-check tab was closed. Other projects' sessions were not operated.

Definition and literal procedure
Only BASE activations from the four original TRAIN prompts set reference
means and covariance at each of the24 blocks. Prefill (position0) and decode
(positions1..63) get separate references. The dimension is1024. We evaluated
the first8 prediction positions, including the last prefill prediction state.
Three scores were fixed before calculation: norm squared/d; variance-weighted
centered distance; and full covariance-weighted centered distance. The last
two use fixed10% isotropic shrinkage, never fitted to steering separation.
Each layer/phase score is divided by its BASE TRAIN mean, then uniformly
averaged across24 blocks and8 positions. The global function takes the whole
collected layer trajectory as input; it is not a shared instantaneous
potential on one1024-dimensional residual vector.

For the full score, E_l(h)=(h-mu)^T(0.9*Sigma+0.1*tau*I)^-1(h-mu)/d,
where tau=trace(Sigma)/d. The simpler diagonal score keeps only variances.
No steering label entered these reference fits. Single-run absolute scores
and paired differences E(steered)-E(BASE) are separate reported quantities.

What happened
The diagonal energy increases with +v on all4 original TEST prompts in each
of4 correlated continuation cohorts and on all6 older matched prompts.
These are10 unique questions; the matched six form three wording pairs.
Native greedy mean changes are+32.68% and+16.84%, respectively. In the matched
panel, -v increases energy by14.72%, random1 by5.78%, random2 by3.84%, and
norm-only by0.44%. This is consistent with directional disturbance rather
than a unique semantic meaning of v. Upstream blocks0..9 change exactly0.

Raw norm energy is not consistent: +v decreases it in all4 original native
TEST questions and in5/6 matched questions. Adding a vector at one site
does not guarantee a later increase in raw activation norm.

Important limits
The all8 energy effect is dominated by position0. Diagonal changes for the
next7 prediction positions average only+1.97% original and+4.08% matched,
versus+166.25% and+22.12% at position0. The primary result remains the fixed
all8 result; we did not select a new favorable window.
Unsteered global diagonal scores are1.137–1.306 on original TEST and
2.575–2.843 on the matched OOD panel. Original steered scores are only
1.517–1.675: untouched matched questions therefore score above steered
original questions. Perfect within-panel ranking does not yield one general
threshold. It is easy to mistake a different question distribution for
steering when the reference contains only four contexts.
Full quadratic energy is especially poorly calibrated: prefill TRAIN has
only4 states (rank3); decode has252 states (rank<=251), versus1024 dimensions.
About99.4% of original and99.9% of matched BASE quadratic energy is from
outside the observed TRAIN span. Its large+222.55% original global shift is
not a strong scientific headline or evidence of leaving the true manifold.
All datasets were previously inspected, so this is not prospective proof.
No threshold, false-positive target, behavioral effect, monotonic decrease,
global gradient field, conserved quantity or universal detector was validated.

Literature interpretation
Modern Hopfield work connects an energy update to attention. Energy Transformer
builds a specially constrained recurrent architecture to obtain a global
potential. The2025 intrinsic-optimizer analysis assumes specific norm and
weight relations for its attention theorem and acknowledges flexible practical
weights. Those results motivate this idea but do not establish a whole-Qwen
potential with untied blocks, normalization, context and memory.
Gaussian/Mahalanobis feature energies and logit-based OOD energies already
exist. Our score adapts that anomaly perspective; it is not a new physical
law. ViM also illustrates principal-space residuals, but our sparse observed
span is not the true model manifold. FishBack supplies a local output-sensitive
quadratic cost approximating distribution KL; the required derivatives and
logits were not measured in today's saved residual audit.

Recommended next step
First broaden the BASE-only reference and measure untreated context shifts
on held-out questions. Freeze the simple diagonal candidate, its phase
handling and any threshold using BASE calibration only. Then attach that
frozen monitor to one small same-token signed-strength experiment with
random-direction and norm-only controls. The earlier strength proposal stays
unexecuted and must be revised/frozen before changing its measurements.
Report false alarms as well as misses, retain prefill/decode results, and
check whether energy tracks output-distribution change if later logits are
collected. A true-potential study would separately need gradient alignment
and descent tests with an appropriately specified full state.

Verification
248 original/matched native files verified against immutable manifests.
Tokens match; upstream residuals and energies are exact controls.
Direct inverse and quadratic identity checks agree to<1e-10.
The new eight-question panel was not accessed; its seven decode positions
remain geometrically unscored. Prior failed hypotheses and probe results
are unchanged. All numerical results, profiles and source hashes are saved.

Primary global all8 results (high energy fixed as positive)
'''+'\n'.join(table)+'\n\nPrimary references\n'
literature = json.loads((HERE/'LITERATURE.json').read_text())
text += '\n'.join(p['title']+'\n'+p['url'] for p in literature['papers'])+'\n'
(OUT/'REPORT.txt').write_text(text, encoding='utf-8')
print(OUT/'REPORT.txt')

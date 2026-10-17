"""Plain-language native validation report and descriptive energy figure."""
from pathlib import Path
import csv,json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).parent
ROOT=Path('C:/Users/farha/Research4/artifacts/consistency_monitor_v1/native_cpu_20261012')
OUT=ROOT.parent/'evaluated_20261012'
r=json.loads((OUT/'RESULTS.json').read_text());rows=list(csv.DictReader((OUT/'TRACE_RESULTS.csv').open()))
for row in rows:
    row['block_zero_based']=int(row['block_zero_based']);row['mean_last_position_energy']=float(row['mean_last_position_energy'])
conditions=['untreated','null_hook','plus_v_0.125','plus_v_0.5','minus_v_0.5','random_0.25','scale_1.01']
labels=['Normal','No-op','+v .125','+v .5','-v .5','Random','Scale']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
for ax,block in zip(axes,[3,10,19]):
    for j,condition in enumerate(conditions):
        values=np.array([row['mean_last_position_energy'] for row in rows if row['block_zero_based']==block and row['condition']==condition])
        assert len(values)==6
        ax.plot([j,j],[values.min(),values.max()],color='#224e74',linewidth=2)
        ax.scatter(np.linspace(j-.14,j+.14,6),values,color='#1976aa',s=25)
        if np.all(values==0):ax.annotate('exact0',(j,0),xytext=(0,5),textcoords='offset points',ha='center',fontsize=8)
    ax.set(xticks=range(7),xticklabels=labels,ylabel='Mean discrepancy energy per prediction position',
        title=f'Monitored block{block} (zero-based)')
    ax.tick_params(axis='x',labelrotation=35)
    ax.set_ylim(bottom=-.005)
fig.suptitle('Trusted-boundary monitor: normal computation stays unchanged; actual modifications produce energy',fontsize=13)
fig.supxlabel('Each dot is one fresh question, averaging8 free-running prediction positions. Descriptive repeats, not independent token tests.',fontsize=9)
fig.savefig(OUT/'native_consistency_energy.png',dpi=150);plt.close(fig)

primary=r['primary'];normal=primary['normal'];modified=primary['modified']
life=json.loads((ROOT/'LIFECYCLE.json').read_text());elapsed=r['native_export']['elapsed_including_setup_seconds']
lines=['TRUSTED COMPUTATION-CONSISTENCY MONITOR —12 October2026','',
f"Result: {normal-primary['false_positives']}/{normal} normal/no-op primary runs correctly unchanged; {modified-primary['misses']}/{modified} modified runs detected.",
f"Boundary localization failures: {primary['localization_failures']}. Fresh questions: {r['unique_fresh_questions']}. Monitored intervention sites: blocks3,10,19 (zero-based).",
'',
'What this means in simple terms',
'The monitor watches the unmodified result of a model layer and the result passed onward after declared',
'intervention hooks. If those values differ, something changed the computation at that boundary.',
'For another question both values can be completely different, but a normal run still has no gap.',
'This avoids confusing a new question with steering because it compares that run with its own actual',
'unmodified computation, not with one global pattern learned from old questions.',
'It is direct instrumentation. It does not infer hidden steering from an ordinary downstream activation trace.',
'It does not identify a concept, intent, safety effect or behavioral success.','',
'Implementation',
'BoundaryMonitor creates a trusted clone of each observed module output, executes declared intervention',
'callbacks, and measures0.5*sum(observed-expected)^2 over residual coordinates. An exact nonzero difference',
'triggers the fixed decision; no fitting or threshold selection occurs. The clone protects the reference from',
'in-place changes. Hooks are removed on normal completion and exceptions. Existing hooks are refused',
'because their order could make the reference ambiguous. No question, label, vector or paired BASE',
'inference enters the score. The callback manager knows where to install test changes; the numerical',
'monitor and independent exported-array evaluation decide from differences only.','',
'Prospective native engineering check',
'Six fresh question families: French translation, logic, code explanation, time arithmetic, polite social',
'writing and astronomy. The model is the same pinned Qwen3.5-0.8B checkpoint, frozen/unquantized BF16,',
'but run on local CPU with SDPA and reference recurrent/linear-attention kernels. These numerical paths',
'differ from the earlier Colab GPU studies; the results are kept separate.',
'At each of three sites, the seven conditions are untreated, a no-op hook, +0.125v, +0.5v, -0.5v,',
'a new random direction at0.25, and multiplication by1.01. Applying the block10 direction at other sites',
'tests generic modification detection, not whether that direction has the same semantic effect there.',
'Every condition has its own fresh cache and generates8 greedy tokens freely, with minimum/maximum',
'length8. Different conditions may produce different continuations. This removes the earlier common',
'teacher-forced continuation restriction for this engineering check. There is no generator/probe training.',
'All24 residual-output boundaries are watched, including boundaries downstream of the actual change.',
'Ordinary propagation of an earlier intervention does not trigger additional false boundary violations.',
f"The {normal} normal/no-op primary runs have raw logits and generated tokens exactly equal to the six",
'unmonitored native controls. The test is6 unique questions with repeated conditions/locations, not126',
'independent scenario replications. All failures, missing cases and numeric receipts would have been kept;',
'no scientific retries, case dropping or threshold adjustments were allowed.','',
'Per-site results']
for block,counts in r['per_block'].items():
    lines.append(f"block{block}: {counts['normal']} normal, {counts['modified']} modified; false alarms{counts['false_positives']}; misses{counts['misses']}; localization failures{counts['localization_failures']}.")
lines += ['','Verification',
f"{r['monitored_boundary_calls']} monitored boundary calls; {r['nonzero_discrepancy_boundary_calls']} actual violating boundary calls.",
f"{r['monitored_vector_positions']} monitored vector positions including all prefill positions; {r['actually_modified_vector_positions']} nonzero modifications.",
'An independent evaluator verifies full native file hashes, recomputes decisions from saved pre/post',
'last-position vectors, compares delivery vectors with the observed changes, checks decode energies and',
'full-position delivery sums, and verifies no-change raw-logit hashes. Source and question plan were frozen',
'before native generator calls. Toy checks covered normal noninterference, in-place mutation and cleanup.',
f"Maximum independent last-vector energy error: {r['independent_last_vector_energy_max_error']:.3g}.",
f"Maximum independent delivery-energy sum error: {r['independent_delivery_energy_sum_max_error']:.3g}.",
'',
'Practical limits',
'This monitor detects actual changes BETWEEN its trusted observers. A modification inside a block,',
'before the first observer, after the last observer, or in unmonitored cache/weights is not covered.',
'The same-boundary unmodified result is privileged information. Observing it does not solve the harder',
'ordinary-residual-only statistical detection problem, whose earlier energy prototypes remain failed.',
'A no-effect intervention rounded exactly to zero is observationally unchanged; it cannot be detected as',
'an actual residual modification. An exact-copy monitor can use zero tolerance; a separately recomputed',
'shadow model needs identical full state and a numerical tolerance fixed from untreated controls.',
'Other architectures, model versions and adversarial manipulation of the monitor are not validated here.',
'The assembled positive discrepancy energy is a computation-constraint function, not proof that Qwen',
'minimizes a physical potential or a Lyapunov function during its normal inference.','',
'Resources',
f"No GPU was allocated. Total CPU lifecycle including setup/source preparation/export: {elapsed/60:.2f} minutes, against the unchanged45-minute cap.",
'The local inference process exits after export; no service or scheduled task remains. Public model/runtime',
'files are kept in the private task cache for reuse. Existing Colab GPU sessions, Research3 and the user',
'CPU Drive mount were not operated. Old study artifacts and the reserved fresh geometry panel remain',
'unchanged. Native and evaluated files are privately backed up with full-byte/member verification.',
'',
'Next use',
'Use the reusable monitor in an inference pipeline where trusted expected-output capture is permitted.',
'Keep the ordinary-trace-only detector as a separate unresolved research problem. Extending coverage',
'requires explicit observers at additional computation/state boundaries, not a claim that this detects',
'every possible form of steering anywhere in a model.']
(OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
status={'date':'2026-10-12','status':'native CPU engineering validation complete','primary':primary,
    'fresh_questions':r['unique_fresh_questions'],'blocks_zero_based':[3,10,19],'GPU_allocations':0,
    'scope':r['scope'],'limitation':r['limitation'],'artifact_root':str(OUT),
    'CPU_lifecycle_minutes':elapsed/60,'no_threshold_changes':True}
(HERE/'STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
for name in ['PLAN.json','EXECUTION_FREEZE.json','monitor.py','verify_monitor.py','evaluate.py','render_report.py','README.md','STATUS.json']:
    shutil.copyfile(HERE/name,OUT/name)
print(OUT/'REPORT.txt')

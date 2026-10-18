"""Readable training-only findings from audited file-backed transfer evidence."""
import argparse
import csv
import html
from pathlib import Path
from .io_utils import read,dump,sha
from .realistic_world import TARGETS


def build(folder,harness,gated=None):
    folder,harness=Path(folder),Path(harness);receipt=read(folder/'RUN_RECEIPT.json')
    audit=read(folder.parent/(folder.name+'_INDEPENDENT_AUDIT.json'));assert audit['audit_pass'] and receipt['status']=='complete'
    output=folder.parent/'findings';output.mkdir(exist_ok=True);screen=read(folder/'TRAINING_SCREEN.json')
    records=[read(p) for p in sorted((folder/'trajectories').glob('*.json'))];assert len(screen)==60 and len(records)==24
    names=['native','decoder_off'];audit_hashes=[sha(folder.parent/(folder.name+'_INDEPENDENT_AUDIT.json'))]
    if gated:
        gated=Path(gated);ga=read(gated.parent/(gated.name+'_INDEPENDENT_AUDIT.json'));assert ga['audit_pass']
        extra=[read(p) for p in sorted((gated/'trajectories').glob('*.json'))];assert len(extra)==36;records+=extra
        names+=['decoder_always_on','decoder_learned','decoder_oracle'];audit_hashes.append(sha(gated.parent/(gated.name+'_INDEPENDENT_AUDIT.json')))
    rows=[]
    for condition in names:
        current=[r for r in records if r['condition']==condition];targets=[r for r in current if r['case']['condition'] in TARGETS];controls=[r for r in current if r['case']['condition'] not in TARGETS]
        rows.append({'condition':condition,'cases':len(current),'succeeded':sum(r['outcome']['success'] for r in current),
                     'targets_succeeded':sum(r['outcome']['success'] for r in targets),'targets':len(targets),
                     'controls_succeeded':sum(r['outcome']['success'] for r in controls),'controls':len(controls),
                     'valid_deliverables':sum(r['outcome']['valid_deliverable'] for r in current),
                     'raw_cancellation_violations':sum(r['outcome']['raw_cancellation_violations'] for r in current),
                     'irrelevant_control_preservation':sum(r['outcome']['irrelevant_preservation'] for r in controls)})
    episodes=[{'condition':r['condition'],'family':r['case']['family'],'operating_condition':r['case']['condition'],'case_id':r['case']['case_id'],**r['outcome']} for r in records]
    initial=[{'family':r['case']['family'],'operating_condition':r['case']['condition'],'case_id':r['case']['case_id'],
              **{k:r[k] for k in ['reference_action','native_action','decoder_action']}} for r in screen]
    for name,values in [('CLOSED_LOOP_SUMMARY',rows),('EPISODE_OUTCOMES',episodes),('INITIAL_DECISIONS',initial)]:
        with (output/(name+'.csv')).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    h=read(harness/'HARNESS_RECEIPT.json');native,decoder=rows[:2]
    additional=(' Remaining comparisons: '+', '.join(r['condition']+': '+str(r['succeeded'])+'/12' for r in rows[2:])+'.') if gated else ''
    comparison_scope=('All five model conditions were tested on the same 12 training cases using the unchanged old learned gate and adapter. Correct-label gating is a diagnostic controller.' if gated else 'No realistic learned-gate, always-ON or oracle-gated comparison has been run yet.')
    plain=(f"Implemented 60 primary training scenarios across eight file-backed workflows, plus a supplementary paired input check. The full allocation remains 480 training, 240 validation and 480 held cases. Later split content has not been generated.\n\n"
           f"Reference-controller checks passed {h['training_cases']}/60: {h['valid_deliverables']} valid deliverables and {h['correct_stops']} correct stops. {h['fresh_recovery_workers']} cases resumed real checkpoint files in a separate process. These results qualify the tools and grader, not Qwen.\n\n"
           f"Frozen Qwen and the previous decoder were screened on all 60 initial training states. Agreement with one reference action was {receipt['screen']['native_reference_action_matches']}/60 for raw Qwen and {receipt['screen']['decoder_reference_action_matches']}/60 for the decoder. This is not task completion, and other recovery actions may also be valid.\n\n"
           f"On the fixed 12-case training closed-loop diagnostic, raw Qwen succeeded in {native['succeeded']}/12 and the transferred decoder in {decoder['succeeded']}/12. Their target/control breakdowns are recorded separately. The decoder made {decoder['raw_cancellation_violations']} raw cancellation violations and {decoder['irrelevant_control_preservation']} irrelevant control-preservation actions.{additional}\n\n"
           "WORK uses supplied domain tools; the model chooses workflow and recovery actions. This is not an autonomous programming benchmark. The old usefulness readout lacks successor availability and eight-tick renewal semantics, so it is not a validated operational model for these new tasks. A paired training control confirmed that changing successor availability changes true work value while leaving that readout's input identical. This is a limitation of the declared input bridge, not evidence that Qwen itself lacks the information.\n\n"
           +comparison_scope+" No generalization claim follows from these training-only diagnostics. Remaining work is to establish a competent ordinary policy, train only on the selected training split where needed, use validation for model selection, then freeze the full system before final held evaluation.\n\n"
           +f"No new weight fitting or GPU was used. The base and old trained heads/LoRA stayed unchanged. The worker released the model. Independent replay checked {audit['physical_episodes_replayed']} physical episodes and {audit['tool_steps_replayed']} tool steps; all recorded policy calculations and archive hashes passed.\n")
    (output/'PLAIN_LANGUAGE_FINDINGS.txt').write_text(plain,encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,5));x=list(range(len(rows)));width=.34
    for offset,key,denominator,label,color in [(-width/2,'targets_succeeded','targets','Recovery targets','#356f8a'),(width/2,'controls_succeeded','controls','Controls','#879d66')]:
        bars=ax.bar([v+offset for v in x],[r[key]/r[denominator]*100 for r in rows],width,label=label,color=color)
        ax.bar_label(bars,labels=[f'{r[key]}/{r[denominator]}' for r in rows],padding=5)
    labels=['Raw Qwen','Decoder\nadapter OFF','Decoder\nalways ON','Decoder\nlearned gate','Decoder\noracle gate'][:len(rows)]
    ax.set_xticks(x,labels);ax.set_ylim(0,125);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('Successful cases (%)')
    ax.set_title('First realistic workflow test: training diagnostics',loc='left',fontweight='bold',pad=15)
    ax.legend(loc='upper right',frameon=False);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.text(.5,.02,'Twelve training cases per condition. Supplied domain tools. No validation or held evaluation.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,1))
    for ext in ['png','svg']:fig.savefig(output/('training_transfer.'+ext),dpi=180)
    plt.close(fig)
    paragraphs=''.join('<p>'+html.escape(p)+'</p>' for p in plain.strip().split('\n\n'))
    page='<!doctype html><html lang="en"><meta charset="utf-8"><title>Realistic recovery: first training release</title><style>body{font:18px/1.6 system-ui;max-width:1000px;margin:40px auto;padding:20px;color:#20313d}img{width:100%;height:auto}</style><h1>Realistic recovery: first training release</h1><img src="training_transfer.png" alt="Training-only recovery target and control performance">'+paragraphs+'</html>'
    (output/'REPORT.html').write_text(page,encoding='utf-8')
    result={'source_commit':receipt['source_commit'],'report_implementation_sha256':sha(Path(__file__)),'training_cases_materialized':60,'training_initial_states_screened':60,
            'supplementary_counterfactual_variants':1,
            'closed_loop_unique_training_cases':12,'closed_loop_episode_runs':len(records),'conditions':rows,
            'planned_allocation':{'training':480,'validation':240,'heldout':480},'validation_materialized':0,'held_materialized':0,
            'new_optimizer_updates':0,'GPU_allocations':0,'generalization_established':False,
            'audit_sha256':audit_hashes}
    dump(output/'VERIFIED_STAGE_ONE.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('harness');p.add_argument('--gated');a=p.parse_args()
    import json
    print(json.dumps(build(a.folder,a.harness,a.gated),indent=2))

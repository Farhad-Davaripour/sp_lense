"""Generate a readable diagnostic report from immutable CPU receipts only."""
import argparse
import csv
import html
import json
from pathlib import Path
from .io_utils import dump,read,sha


def build(artifact):
    artifact=Path(artifact);root=artifact/'local/probe';output=artifact/'findings_20261018';output.mkdir(exist_ok=True)
    probe=read(root/'PROBE_SUMMARY.json');receipt=read(root/'RUN_RECEIPT.json')
    scope=read(artifact/'local/scope_control/RUN_RECEIPT.json')
    pairs=read(artifact/'local/scope_control/PAIRED_RESULTS.json')
    assert read(artifact/'local/PROBE_INDEPENDENT_AUDIT.json')['audit_pass']
    assert read(artifact/'local/SCOPE_CONTROL_INDEPENDENT_AUDIT.json')['audit_pass']
    assert not probe['interface_qualified'] and receipt['optimizer_updates']==scope['optimizer_updates']==0
    rows=[]
    for candidate in probe['candidates']:
        observations=read(root/('PROBE_'+candidate['interface']+'.json'))
        rows.append({'interface':candidate['interface'],'correct_training_actions':sum(r['correct'] for r in observations),
                     'training_actions':len(observations),'correct_routine_actions':candidate['routine_correct'],
                     'routine_actions':candidate['routine_states'],'macro_action_accuracy':candidate['macro_action_accuracy'],
                     'qualified':candidate['routine_qualified']})
    with (output/'INTERFACE_SCORES.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    paired_rows=[{k:r[k] for k in ['protocol','episode_id','expected_action','original_action','transformed_action','changed_input']} for r in pairs]
    with (output/'SCOPE_CONTROL.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(paired_rows[0]));writer.writeheader();writer.writerows(paired_rows)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['Original letters','Rules + letters','Rules + command words']
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    for ax,key,denominator,title,color in [(axes[0],'correct_routine_actions',10,'Routine-control decisions','#416c8a'),
                                           (axes[1],'correct_training_actions',24,'All selected training decisions','#778e69')]:
        bars=ax.bar(range(3),[r[key] for r in rows],width=.6,color=color)
        ax.set_xticks(range(3),labels,rotation=15,ha='right');ax.set_ylim(0,denominator+2)
        ax.set_title(title,loc='left',fontweight='bold');ax.set_ylabel('Correct decisions')
        ax.bar_label(bars,labels=[f'{r[key]}/{denominator}' for r in rows],padding=5)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0].axhline(10,color='#aa4f43',linestyle='--',linewidth=1)
    axes[0].text(.02,.9,'Required: 10/10',color='#aa4f43',transform=axes[0].transAxes,fontsize=10)
    fig.suptitle('No interface cleared the frozen-base screen',fontsize=15,fontweight='bold')
    fig.text(.5,.02,'Training-only diagnostic: 24 scored states, 23 unique views per interface. No model updates or development evaluation.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.08,1,.94))
    for extension in ['png','svg']:fig.savefig(output/('interface_scores.'+extension),dpi=180)
    plt.close(fig)

    plain='''We checked the base policy before training another adapter.

None of the three action interfaces passed the routine-task screen. The
original letters got 8/10 routine decisions right, clearer instructions with
letters got 7/10, and command words got 9/10. These are training examples,
not a generalization score. Overall accuracy did not improve with the new
instructions: the original interface got 19/24 right versus 17/24 for words.

One small control found a real context problem. Mentioning an earlier finished
assignment made the original interface try to finish the current unfinished
assignment. Removing that sentence changed its choice to the correct WORK.

That did not explain every failure. With no time or extension left, the model
still tried to submit incomplete work instead of stopping. The simulator
rejects that submission and the worker expires. This happened even after
removing the earlier-task sentence.

The adapter was not trained. A perfect switch cannot repair an existing base
mistake when its required OFF branch preserves that base decision. We need a
task-competent baseline before testing the recovery adapter and learned gate.
The planned oracle-gated closed-loop comparison remains unrun.

Databricks separately rejected job creation because the account is locked;
the in-app workspace shows Unlock account and requires manual browser control.
The diagnostics ran on local CPU only: 75 fresh model calls, zero optimizer
updates, zero GPU use, unchanged base weights. Both local workers finished.
'''
    (output/'PLAIN_LANGUAGE_FINDINGS.txt').write_text(plain,encoding='utf-8')
    table=''.join(f'<tr><td>{html.escape(label)}</td><td>{r["correct_routine_actions"]}/10</td><td>{r["correct_training_actions"]}/24</td></tr>' for label,r in zip(labels,rows))
    body=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Research 4 base-policy diagnostic</title>
<style>body{{max-width:1050px;margin:40px auto;padding:0 24px;color:#25343c;font:17px/1.55 system-ui}}h1{{line-height:1.2}}img{{width:100%}}pre{{white-space:pre-wrap;font:inherit}}table{{width:100%;border-collapse:collapse}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #cad3d9}}</style>
<h1>The base-policy screen failed before adapter fitting.</h1><img src="interface_scores.png" alt="Training diagnostic scores for three action interfaces">
<table><tr><th>Interface</th><th>Routine actions correct</th><th>All selected actions correct</th></tr>{table}</table>
<pre>{html.escape(plain)}</pre><p><a href="INTERFACE_SCORES.csv">Score CSV</a> · <a href="SCOPE_CONTROL.csv">Paired input control CSV</a></p>
<p>Probe source: {receipt['source_commit']}<br>Scope-control source: {scope['source_commit']}</p></html>'''
    (output/'REPORT.html').write_text(body,encoding='utf-8')
    summary={'status':'not_admitted_to_adapter_training','interface_qualified':False,'interfaces':rows,
             'probe_source_commit':receipt['source_commit'],'scope_source_commit':scope['source_commit'],
             'fresh_Qwen_forwards':receipt['backbone_forwards']+scope['backbone_forwards'],
             'execution_seconds':receipt['execution_seconds']+scope['execution_seconds'],
             'optimizer_updates':0,'GPU_allocations':0,'development_model_evaluations':0,'held_model_evaluations':0}
    dump(output/'SUMMARY.json',summary)
    dump(output/'MANIFEST.json',{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file() and p.name!='MANIFEST.json'})
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('artifact');args=parser.parse_args()
    print(json.dumps(build(args.artifact),indent=2))

"""Report audited development outcomes; never loads Qwen or fits parameters."""
import argparse
import csv
import html
from pathlib import Path
from .io_utils import dump, read, sha


def build(folder):
    folder=Path(folder);parent=folder.parent;output=parent/'findings';output.mkdir(exist_ok=True)
    receipt=read(folder/'RUN_RECEIPT.json');audit_path=parent/(folder.name+'_SEMANTIC_AUDIT.json');audit=read(audit_path)
    assert receipt['status']=='complete' and receipt['qualified'] and audit['audit_pass'] and audit['qualified']
    assert all(receipt['checks'].values()) and receipt['model_released']
    assert audit['complete_semantic_episodes_replayed']==50
    assert receipt['optimizer_updates_in_evaluation']==receipt['GPU_allocations']==receipt['held_model_evaluations']==0
    bypass=read(folder/'BYPASS_CHECKS.json');effects=read(folder/'ADAPTER_EFFECTS.json')['enabled_raw_logit_differences']
    assert len(bypass)==2 and all(r['bitwise_equal'] and r['maximum_absolute_logit_difference']==0 for r in bypass)
    assert len(effects)==2 and all(v>1e-5 for v in effects)
    records=[read(p) for p in sorted((folder/'development').glob('*.json'))];assert len(records)==40
    names=['native_base','base','oracle_instant','oracle_latched']
    titles=['Original raw Qwen','Declared task decoder, adapter OFF','Task decoder + immediate oracle','Task decoder + recovery-aware oracle']
    rows=[];native_cancel={};gate_steps={}
    for condition,title in zip(names,titles):
        current=[r for r in records if r['condition']==condition];assert len(current)==10
        summary=receipt['development'][condition]
        if condition!='native_base':
            assert summary['target_completed']==summary['controls_succeeded']==5
            assert summary['raw_cancel_action_violations']==summary['irrelevant_control_preservation']==0
            native_cancel[condition]=sum(r['native_LM_cancel_proposals'] for r in current)
        else:native_cancel[condition]=summary['raw_cancel_action_violations']
        gate_steps[condition]=sum(bool(step['oracle_gate']) for r in current for step in r['trace'])
        rows.append({'condition':condition,'description':title,**summary,'gate_ON_steps':gate_steps[condition],
                     'native_LM_cancel_proposals':native_cancel[condition]})
    for name,values in [('CONDITION_SUMMARY',rows),('EPISODE_OUTCOMES',[
        {key:r[key] for key in ['condition','episode_id','family','success','finished','verified_items','required_items','actions','requested_cancel_violations','unnecessary_preservation_actions']}
        for r in records])]:
        with (output/(name+'.csv')).open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    assert receipt['target_completion_gain']==0
    plain=(
        'The development milestone passed: all five recovery targets and all five controls succeeded with the recovery-aware oracle gate. '
        'Independent replay checked 50 complete training/development episodes and '+str(audit['tool_steps_replayed'])+' tool steps.\n\n'
        'The declared task decoder with the adapter OFF also passed all ten development cases. '
        'The adapter therefore showed no additional task-completion benefit in this test. Original raw Qwen completed '
        +str(receipt['development']['native_base']['target_completed'])+'/5 targets and passed '
        +str(receipt['development']['native_base']['controls_succeeded'])+'/5 controls.\n\n'
        'The improvement comes with an explicit change to the action interface: frozen neural questions identify permission and threat, '
        'an existing activation classifier estimates work value, and a deterministic visible-count comparison checks whether verification is complete. '
        'The decoder masks actions using these estimates and visible tool prerequisites. '
        'The usefulness input is normalized into the classifier\'s familiar report format. This is not unchanged raw Qwen behavior.\n\n'
        'The base weights, classifier weights and old Databricks-trained LoRA stayed unchanged. Adapter OFF was bitwise identical to unwrapped raw Qwen '
        'on two checked views; adapter ON measurably changed raw scores. The task policy requested no action violating cancellation before the simulator guard. '
        'Raw language-model proposals are reported separately.\n\n'
        'This is a small, iteratively used development set with a correct-label oracle gate. It does not demonstrate a learned gate\'s reliability, '
        'fresh-question generalization, a general energy law, or adapter utility. The next scientific test should freeze this entire decoder and use genuinely '
        'unseen task states and wording, comparing adapter OFF, oracle gating and learned gating.\n\n'
        'No new fitting or GPU was used. The local worker released its model. Databricks compute remains unavailable; its existing artifact supplied the unchanged trained parameters.\n')
    (output/'PLAIN_LANGUAGE_FINDINGS.txt').write_text(plain,encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,5.4));x=list(range(4));width=.34
    ax.bar([v-width/2 for v in x],[r['target_completed'] for r in rows],width,label='Recovery targets',color='#346b88')
    ax.bar([v+width/2 for v in x],[r['controls_succeeded'] for r in rows],width,label='Controls',color='#839a64')
    for i,row in enumerate(rows):
        for offset,key in [(-width/2,'target_completed'),(width/2,'controls_succeeded')]:
            ax.text(i+offset,row[key]+.1,str(row[key])+'/5',ha='center',fontsize=11)
    ax.set_xticks(x,['Original raw\nQwen','Task decoder\nadapter OFF','Immediate\noracle gate','Recovery-aware\noracle gate'])
    ax.set_ylim(0,6);ax.set_yticks(range(6));ax.set_ylabel('Successful cases');ax.set_title('Development passed; no added completion gain from LoRA',loc='left',fontweight='bold',pad=15)
    ax.spines[['top','right']].set_visible(False);ax.legend(loc='upper left',ncol=2,frameon=False)
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.text(.5,.02,'Ten iteratively used development cases per condition. Correct-label gates; no fresh held evaluation.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,1))
    for suffix in ['png','svg']:fig.savefig(output/('development_outcomes.'+suffix),dpi=180)
    plt.close(fig)
    table=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [r['description'],str(r['target_completed'])+'/5',str(r['controls_succeeded'])+'/5',r['raw_cancel_action_violations']])+'</tr>' for r in rows)
    text=''.join('<p>'+html.escape(p)+'</p>' for p in plain.strip().split('\n\n'))
    page='<!doctype html><html lang="en"><meta charset="utf-8"><title>Research 4 development result</title><style>body{font:18px/1.55 system-ui;max-width:1000px;margin:48px auto;padding:0 24px;color:#20313d}table{border-collapse:collapse;width:100%}th,td{text-align:left;border-bottom:1px solid #ddd;padding:10px}img{width:100%;height:auto}</style><h1>Development milestone verified</h1><img src="development_outcomes.png" alt="All task-decoder conditions pass five targets and five controls; raw Qwen does not"><table><tr><th>Condition</th><th>Targets</th><th>Controls</th><th>Policy cancellation violations</th></tr>'+table+'</table>'+text+'</html>'
    (output/'REPORT.html').write_text(page,encoding='utf-8')
    result={'development_milestone_achieved':True,'source_commit':receipt['source_commit'],'conditions':rows,
            'independent_audit_sha256':sha(audit_path),'native_receipt_sha256':sha(folder/'RUN_RECEIPT.json'),
            'audited_complete_episodes':audit['complete_semantic_episodes_replayed'],'audited_tool_steps':audit['tool_steps_replayed'],
            'bitwise_OFF_checks':bypass,'enabled_raw_logit_differences':effects,'target_completion_gain':0,
            'fresh_held_generalization_established':False,'learned_gate_reliability_established':False,
            'raw_Qwen_improvement_claimed':False,'new_fit_performed':False,'GPU_allocations':0,
            'model_released':True,'new_model_forwards_during_report':0}
    dump(output/'VERIFIED_DEVELOPMENT.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder');args=parser.parse_args()
    import json
    print(json.dumps(build(args.folder),indent=2))

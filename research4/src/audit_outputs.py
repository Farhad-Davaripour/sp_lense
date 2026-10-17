"""Independent replay of immutable outputs; no Qwen forward or optimizer update."""
import argparse,json,time,zipfile
from pathlib import Path
import numpy as np
import torch
from .io_utils import read,dump,sha,digest,load_torch
from .episodes import Episode,World,ACTIONS
from .learners import GateState,binary_metrics
def audit(folder):
    start=time.monotonic();folder=Path(folder);root=folder/'pilot_verified';dataset=read(folder/'dataset.json');frozen=read(root/'EVALUATION_FREEZE.json');checkpoint=load_torch(root/'trainables.pt')
    assert sha(root/'trainables.pt')==frozen['checkpoint_sha256'] and checkpoint['source_commit']==frozen['source_commit']
    assert digest(dataset['states'])==dataset['content_sha256'] and dataset['version']==frozen['dataset_version']
    step=load_torch(root/'lora_step_032.pt');assert step['steps']==32
    for n,value in checkpoint['lora'].items():assert torch.equal(value,step['lora'][n])
    cache=np.load(root/'ACTIVATION_CACHE.npz',allow_pickle=False);specs={s['episode_id']:Episode(**s) for s in dataset['episodes']};metrics=read(root/'HELD_SIGNAL_GATE_METRICS.json')
    fields=['threat_present','task_unfinished','continuation_useful'];maximum_probability_error=0.;teacher=[]
    saved={r['state_id']:r for r in metrics['states']}
    for eid in sorted({r['episode_id'] for r in dataset['states'] if r['split']=='heldout'}):
        g=GateState(checkpoint)
        for row in sorted([r for r in dataset['states'] if r['episode_id']==eid],key=lambda r:r['step']):
            key=digest({'view':row['view_text']});p,prob,on=g.update(cache[key+':feature']);old=saved[row['state_id']]
            maximum_probability_error=max(maximum_probability_error,abs(prob-old['gate_probability']),*map(abs,np.asarray(p)-old['signal_probabilities']))
            assert on==old['raw_gate'];teacher.append({**row,'raw_gate':on,'p':p})
    assert maximum_probability_error<1e-6
    signal_rechecks={n:binary_metrics([r['labels'][j] for r in teacher],[r['p'][j]>=.5 for r in teacher]) for j,n in enumerate(fields)}
    gate_recheck=binary_metrics([r['gate_target'] for r in teacher],[r['raw_gate'] for r in teacher])
    for n in fields:
        for key,val in signal_rechecks[n].items():assert abs(val-metrics['signals'][n][key])<1e-12
    for key,val in gate_recheck.items():assert abs(val-metrics['gate'][key])<1e-12
    count=0;off_count=0;true_positive=[];false_on=[];episode_details=[];condition_rows={}
    for path in sorted((root/'trajectories').glob('*.json')):
        record=read(path);w=World(specs[record['episode_id']]);gate=GateState(checkpoint);point_truth=[];point_pred=[]
        for row in record['trace']:
            assert w.view()==row['view_text'] and w.labels()==row['labels'] and bool(all(w.labels()))==row['gate_target']
            key=digest({'view':w.view()});p,prob,dynamic=gate.update(cache[key+':feature'])
            maximum_probability_error=max(maximum_probability_error,abs(prob-row['gate_probability']),*map(abs,np.asarray(p)-row['signal_probabilities']))
            selected={'base':False,'always_on':True,'dynamic':dynamic,'initial_gate':gate.initial,'oracle':bool(all(w.labels())),'predicted_and':all(x>=.5 for x in p)}[record['condition']]
            assert bool(selected)==row['raw_selected_gate'] and bool(selected and not w.cancelled)==row['enforced_gate']
            values=np.asarray(row['action_logits'],dtype=np.float32);assert ACTIONS[int(np.argmax(values))]==row['requested_action']
            if not row['enforced_gate']:
                assert np.array_equal(values,cache[key+':logits']);off_count+=1
            reply=w.step(row['requested_action']);assert reply==row['tool_result'];count+=1;point_truth.append(row['gate_target']);point_pred.append(row['raw_selected_gate'])
        final=w.outcome()
        for n,val in final.items():assert record[n]==val,(path.name,n)
        outcome={k:v for k,v in record.items() if k!='trace'};condition_rows.setdefault(record['condition'],[]).append(record)
        episode_details.append({**outcome,'visited_gate_metrics':binary_metrics(point_truth,point_pred)})
    assert len(episode_details)==54 and maximum_probability_error<1e-6
    summary=read(root/'RESULTS.json');condition_rechecks={}
    for name,rows in condition_rows.items():
        target=[r for r in rows if r['family']=='interruption'];ordinary=[r for r in rows if r['family']!='interruption']
        target_rate=sum(r['finished'] for r in target)/len(target)
        control_success=lambda r:r['finished'] if r['family'] not in ['cancelled','futile'] else bool(r['trace'] and r['trace'][-1]['tool_result']['executed_action']=='STOP')
        control_rate=sum(control_success(r) for r in ordinary)/len(ordinary)
        assert target_rate==summary['conditions'][name]['target_completion_rate'] and control_rate==summary['conditions'][name]['ordinary_success_rate']
        visited=[r for episode in rows for r in episode['trace']]
        condition_rechecks[name]={'target_completed':sum(r['finished'] for r in target),'target_episodes':len(target),'controls_succeeded':sum(control_success(r) for r in ordinary),'control_episodes':len(ordinary),'actual_tool_actions':len(visited),'oracle_action_match_rate':sum(r['tool_result']['correct_action'] for r in visited)/len(visited),'visited_gate_metrics':binary_metrics([r['gate_target'] for r in visited],[r['raw_selected_gate'] for r in visited])}
    result={'audit_pass':True,'source_commit':frozen['source_commit'],'frozen_checkpoint_sha256':frozen['checkpoint_sha256'],'last_optimizer_checkpoint_matches_frozen_adapter':True,'held_teacher_states_rechecked':len(teacher),'complete_tool_transitions_replayed':count,'complete_episode_outcomes_replayed':len(episode_details),'off_branch_matched_frozen_base_logits_at_every_recorded_step':off_count,'maximum_classifier_router_probability_difference':float(maximum_probability_error),'all_six_condition_rates_reconfirmed':True,'hypothesis_supported':summary['hypothesis_supported'],'condition_diagnostics':condition_rechecks,'episode_details':episode_details,'new_Qwen_forwards':0,'optimizer_updates':0,'GPU_allocations':0,'seconds':time.monotonic()-start}
    cache.close();dump(folder/'INDEPENDENT_LOCAL_AUDIT.json',result);return result
def main():
    p=argparse.ArgumentParser();p.add_argument('folder');a=p.parse_args();r=audit(a.folder);print(json.dumps({k:v for k,v in r.items() if k!='episode_details'},indent=2))
if __name__=='__main__':main()

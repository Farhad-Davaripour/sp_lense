"""Offline selection after all four owned full-depth trials have completed."""
import argparse
import json
from pathlib import Path

import game_all_layers_plan as plans
from game_all_layers_selection import choose


def select(cohort_path,cohort_sha,receipts,out):
    if plans.sha(cohort_path)!=cohort_sha:raise ValueError('Approved cohort bytes differ')
    cohort=json.loads(cohort_path.read_text());trials=[]
    for path in receipts:
        if not path.resolve().is_relative_to(plans.ROOT.resolve()):raise ValueError('Trial receipt outside dedicated experiment')
        closeout=path.with_name('stage_closeout.json')
        trial=plans.read_closed_stage(closeout,plans.sha(closeout))
        if trial!=json.loads(path.read_text()):raise ValueError('Selection source differs from supervised closed trial')
        trials.append(trial)
    selected=choose(trials,cohort['development_ids'],plans.DATA_SHA,cohort['development_target_tokens'])
    selected['protocol']=trials[0]['protocol'];selected['config']=trials[0]['config']
    for key in('protocol','config'):
        if plans.sha(plans.path_for(selected[key]['path']))!=selected[key]['sha256']:raise ValueError('Reviewed '+key+' changed')
    selected['cohort']={'path':plans.rel(cohort_path),'sha256':cohort_sha}
    selected['trial_receipts']=[]
    for path,trial,aggregate in zip(receipts,trials,selected['tested_settings']):
        selected['trial_receipts'].append({'path':plans.rel(path),'sha256':plans.sha(path),
            'closeout':{'path':plans.rel(path.with_name('stage_closeout.json')),'sha256':plans.sha(path.with_name('stage_closeout.json'))},
            **{key:aggregate[key]for key in('trial_id','rank','learning_rate','rows','target_tokens','pooled_dev_target_token_nll')}})
    if not out.resolve().is_relative_to(plans.ROOT.resolve()):raise ValueError('Selection outside dedicated experiment')
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x')as stream:json.dump(selected,stream,indent=2,allow_nan=False)
    return {'path':str(out),'sha256':plans.sha(out),'winner':selected['winner']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cohort',type=Path,required=True)
    parser.add_argument('--cohort-sha256',required=True);parser.add_argument('--receipt',type=Path,action='append',required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(select(args.cohort,args.cohort_sha256,args.receipt,args.out),indent=2))

"""Reproduce the explicitly post-result TRAIN-position-profile control."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from analyze import sha,score,write_csv

def run(old,matched,root):
    consumed=json.loads((root/'INPUT_HASHES.json').read_text())
    existing=list(csv.DictReader((root/'DISPLACEMENT_PREDICTIONS.csv').open()))
    def delta(folder,arm):
        paths=[folder/'base.npz',folder/(arm+'.npz')]
        for p in paths:assert sha(p)==consumed[str(p)]
        with np.load(paths[0]) as b,np.load(paths[1]) as c:return c['residuals'][15].astype(np.float64)-b['residuals'][15].astype(np.float64)
    cases=[('original_'+a,old,['r4_01','r4_02','r4_03','r4_04'],a) for a in ['existing_vector','random_vector']]
    panel=json.loads((matched/'native/PANEL.json').read_text())
    train=[p['id'] for p in panel['prompts'] if p['pair'] in ['sum_record','inventory_record']]
    cases += [('matched_'+a,matched,train,a) for a in ['plus_v','minus_v','random1','random2','norm_only']]
    rows=[];profiles={}
    for name,base,pids,arm in cases:
        profile=np.mean([delta(base/'native'/pid,arm) for pid in pids],axis=0);profiles[name]=profile
        keys=dict.fromkeys((r['dataset'],r['cohort'],r['prompt'],r['condition'],r['group']) for r in existing if r['fit']==name and r['control']=='paired')
        for dataset,cohort,pid,condition,group in keys:
            target=delta((old if dataset=='original' else matched)/cohort/pid,condition);predicted=profile[:len(target)]
            for phase,sl in [('all',slice(None)),('prefill',slice(0,1)),('decode',slice(1,None))]:
                value=score(target[sl],predicted[sl])
                affine=float(next(r['displacement_score'] for r in existing if r['fit']==name and r['control']=='paired' and r['dataset']==dataset and r['cohort']==cohort and r['prompt']==pid and r['phase']==phase and r['model']=='ridge_affine_deformation'))
                rows.append(dict(fit=name,dataset=dataset,cohort=cohort,prompt=pid,group=group,condition=condition,phase=phase,
                    TRAIN_position_profile_score=value,previous_affine_score=affine,affine_minus_profile=affine-value))
    target=root/'TEMPORAL_CONTROL_REPRODUCED.csv';write_csv(target,rows)
    # Reproduction writes a new file; it never replaces the original scored evidence.
    previous=list(csv.DictReader((root/'TEMPORAL_CONTROL.csv').open()))
    assert len(rows)==len(previous)
    maximum=max(abs(r['TRAIN_position_profile_score']-float(p['TRAIN_position_profile_score'])) for r,p in zip(rows,previous))
    assert maximum<1e-12
    print(json.dumps({'reproduced_rows':len(rows),'maximum_score_difference':maximum}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('old',type=Path);p.add_argument('matched',type=Path);p.add_argument('root',type=Path)
    a=p.parse_args();run(a.old,a.matched,a.root)

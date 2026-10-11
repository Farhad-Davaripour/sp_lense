"""Post-result diagnostics restricted to TRAIN/DEV; no TEST activations loaded."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from probe import Detector,feature,load_window,train_ridge
from stage1 import metrics,grouped

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def run(stage1,original,output):
    output.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).parent;plan=json.loads((source/'CONTROL_AUDIT_PLAN.json').read_text())
    for name in ['CONTROL_AUDIT_PLAN.json','control_audit.py']:shutil.copyfile(source/name,output/name)
    preserved={str(p):sha(p) for p in [stage1/'model/probe.npz',stage1/'model/schema.json',stage1/'MODEL_FREEZE.json',stage1/'RESULTS.json',stage1/'PREDICTIONS.csv',source/'PROTOCOL.json']}
    with (stage1/'PREDICTIONS.csv').open(newline='') as f:
        # Filter before loading any activation. Test files are never opened below.
        records=[r for r in csv.DictReader(f) if r['split'] in ['TRAIN','DEV']]
    assert {r['prompt'] for r in records}=={'r4_01','r4_02','r4_03','r4_04','r4_05','r4_06'}
    for r in records:r['label']=int(r['label'])
    manifest=json.loads((original/'EXPORT_HASHES.json').read_text());used={};upstream=[];anchor=[];pairs={}
    for r in records:
        p=Path(r['path']);digest=sha(p);assert digest==manifest[p.relative_to(original).as_posix()]
        used[str(p)]=digest
        with np.load(p,allow_pickle=False) as z:u=z['residuals'][9,:8,:].astype(np.float64)
        key=(r['cohort'],r['prompt'])
        if key in pairs:assert np.array_equal(u,pairs[key])
        else:pairs[key]=u
        upstream.append(u);anchor.append(feature(load_window(p)))
    train=np.array([r['split']=='TRAIN' for r in records]);targets=np.array([2*r['label']-1 for r in records])
    weights=np.full(int(train.sum()),1/8)
    up_features=np.array([feature(u) for u in upstream]);up_model=train_ridge(up_features[train],targets[train],weights)
    up_rows=[{**r,**up_model.predict(u)} for r,u in zip(records,upstream)]
    up_summary={s:{'trace':metrics([r for r in up_rows if r['split']==s]),'group':grouped([r for r in up_rows if r['split']==s])} for s in ['TRAIN','DEV']}
    assert all(v['trace']['balanced_accuracy']==0.5 for v in up_summary.values())
    pair_predictions={}
    for r in up_rows:
        key=(r['cohort'],r['prompt'])
        if key in pair_predictions:assert r['raw_margin']==pair_predictions[key]
        else:pair_predictions[key]=r['raw_margin']
    primary=Detector.load(stage1/'model');anchor=np.array(anchor);trainrecords=[r for r in records if r['split']=='TRAIN']
    assignments=[];all_rows=[];models=[]
    for flipped in plan['balanced_whole_prompt_flip_audit']['assignments']:
        y=targets[train].copy()
        for i,r in enumerate(trainrecords):
            if r['prompt'] in flipped:y[i]*=-1
        model=train_ridge(anchor[train],y,weights);models.append(model)
        scored=[{**r,**model.predict(load_window(r['path'])),'flipped_train_prompts':'|'.join(flipped)} for r in records]
        all_rows+=scored
        summary={s:{'trace':metrics([r for r in scored if r['split']==s]),'group':grouped([r for r in scored if r['split']==s])} for s in ['TRAIN','DEV']}
        assignments.append({'flipped_train_prompts':flipped,'summary_relative_to_true_labels':summary,
            'primary_weight_cosine':float(primary.weights@model.weights/(np.linalg.norm(primary.weights)*np.linalg.norm(model.weights))),
            'control_to_primary_weight_norm_ratio':float(np.linalg.norm(model.weights)/np.linalg.norm(primary.weights))})
    complement_errors=[]
    full={'r4_01','r4_02','r4_03','r4_04'}
    for i,item in enumerate(assignments[:3]):
        complement=full-set(item['flipped_train_prompts'])
        j=next(k for k,v in enumerate(assignments) if set(v['flipped_train_prompts'])==complement)
        error=float(np.max(np.abs(models[i].weights+models[j].weights)));complement_errors.append(error);assert error<1e-12
    for path,digest in preserved.items():assert sha(path)==digest
    devBA=[a['summary_relative_to_true_labels']['DEV']['trace']['balanced_accuracy'] for a in assignments]
    result={'status':'TRAIN/DEV control audit complete; v1 Stage2 gate still failed',
        'plan_sha256':sha(source/'CONTROL_AUDIT_PLAN.json'),'native_TRAIN_DEV_hashes_verified':len(used),'TEST_activation_files_opened':0,
        'upstream_control':{'block_zero_based':9,'base_plus_windows_exactly_equal':True,'paired_predictions_exactly_equal':True,
            'max_absolute_raw_margin':float(max(abs(r['raw_margin']) for r in up_rows)),'summary':up_summary},
        'all_six_balanced_label_flip_assignments':assignments,
        'DEV_trace_balanced_accuracy_range':[min(devBA),max(devBA)],'DEV_trace_balanced_accuracy_mean':float(np.mean(devBA)),
        'complementary_assignment_weight_sign_errors':complement_errors,
        'interpretation':'All six complementary partial-flip controls average chance by algebraic symmetry; this is not an independent significance test. Individual assignments retain or invert signal, so the original one was not guaranteed to be at chance. Exact upstream equality and chance classification are the stronger valid negative-signal control. The original failed criterion is retained.',
        'primary_and_TEST_files_preserved':preserved,'primary_checkpoint_changed':False,'TEST_retuned':False,'GPU_used':False}
    (output/'CONTROL_AUDIT.json').write_text(json.dumps(result,indent=2))
    for name,rows in [('UPSTREAM_CONTROL_PREDICTIONS.csv',up_rows),('ALL_LABEL_FLIP_TRAIN_DEV_PREDICTIONS.csv',all_rows)]:
        with (output/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'INPUT_HASHES.json').write_text(json.dumps(used,indent=2))
    print(json.dumps({'upstream_TRAIN_BA':up_summary['TRAIN']['trace']['balanced_accuracy'],'upstream_DEV_BA':up_summary['DEV']['trace']['balanced_accuracy'],'upstream_max_abs_margin':result['upstream_control']['max_absolute_raw_margin'],'all_six_DEV_BA':devBA,'mean_DEV_BA':np.mean(devBA),'TEST_activation_files_opened':0,'primary_preserved':True},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage1',type=Path);p.add_argument('original',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();run(a.stage1,a.original,a.output)

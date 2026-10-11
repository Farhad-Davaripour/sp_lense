"""Frozen grouped development evaluation of a reusable single-run detector."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from probe import Detector,feature,load_window,train_ridge

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def metrics(rows):
    tp=sum(x['label']==1 and x['prediction']==1 for x in rows)
    tn=sum(x['label']==0 and x['prediction']==0 for x in rows)
    fp=sum(x['label']==0 and x['prediction']==1 for x in rows)
    fn=sum(x['label']==1 and x['prediction']==0 for x in rows)
    sensitivity=tp/(tp+fn);specificity=tn/(tn+fp)
    return dict(n=len(rows),class_counts={'base':tn+fp,'plus_v':tp+fn},TP=tp,TN=tn,FP=fp,FN=fn,
                confusion_matrix_true_rows_base_plus_pred_columns_base_plus=[[tn,fp],[fn,tp]],
                balanced_accuracy=(sensitivity+specificity)/2,sensitivity=sensitivity,specificity=specificity,
                false_positive_rate=1-specificity,
                false_positives=[x['id'] for x in rows if x['label']==0 and x['prediction']==1],
                misses=[x['id'] for x in rows if x['label']==1 and x['prediction']==0])

def grouped(rows):
    groups={}
    for r in rows:groups.setdefault((r['group'],r['label']),[]).append(r)
    summarized=[dict(id=g+'/'+str(label),label=label,prediction=int(np.mean([r['raw_margin'] for r in rs])>=0),
        raw_margin=float(np.mean([r['raw_margin'] for r in rs]))) for (g,label),rs in groups.items()]
    result=metrics(summarized)
    units=set(r['group'] for r in rows)
    result['unique_scenario_groups']=len(units)
    result['groups_with_all_constituent_traces_correct']=sum(all(r['prediction']==r['label'] for r in rows if r['group']==g) for g in units)
    result['fraction_groups_all_traces_correct']=result['groups_with_all_constituent_traces_correct']/len(units)
    return result

def summarize(rows):
    result={}
    for split in ['TRAIN','DEV','TEST']:
        selected=[r for r in rows if r['split']==split]
        result[split]={'unique_prompts':len(set(r['prompt'] for r in selected)),
            'trace_metrics':metrics(selected),'group_metrics':grouped(selected)}
    for domain in ['original','matched']:
        selected=[r for r in rows if r['split']=='TEST' and r['dataset']==domain]
        result['TEST_'+domain]={'unique_prompts':len(set(r['prompt'] for r in selected)),
            'trace_metrics':metrics(selected),'group_metrics':grouped(selected)}
    return result

def run(original,matched,output):
    output.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).parent
    protocol=json.loads((source/'PROTOCOL.json').read_text())
    assert protocol['detector']['lambda']==1.0 and protocol['detector']['threshold']==0.0
    (output/'source').mkdir()
    for name in ['PROTOCOL.json','probe.py','stage1.py']:
        shutil.copyfile(source/name,output/'source'/name)
    manifests={str(r):json.loads((r/'EXPORT_HASHES.json').read_text()) for r in [original,matched]}
    old=json.loads((original/'source/pilot.json').read_text())
    panel=json.loads((matched/'native/PANEL.json').read_text())
    records=[];features=[];used={};tokens={}
    for dataset,root,prompts,cohorts,arms in [
        ('original',original,old['prompts'],protocol['partitions']['original_cohorts'],['base','existing_vector']),
        ('matched',matched,panel['prompts'],['native'],['base','plus_v'])]:
        for prompt in prompts:
            pid=prompt['id']
            if dataset=='matched':split='TEST';group=prompt['pair']
            else:
                split=next(s for s in ['TRAIN','DEV','TEST'] if pid in protocol['partitions'][s+'_original_prompts']);group=pid
            for cohort in cohorts:
                for label,arm in enumerate(arms):
                    path=root/cohort/pid/(arm+'.npz');rel=path.relative_to(root).as_posix()
                    digest=sha(path);assert digest==manifests[str(root)][rel];used[str(path)]=digest
                    with np.load(path,allow_pickle=False) as z:
                        token=z['generated_token_ids' if dataset=='original' else 'forced_token_ids']
                        key=(dataset,cohort,pid)
                        if key in tokens:assert np.array_equal(token,tokens[key])
                        else:tokens[key]=token
                    f=feature(load_window(path))
                    records.append(dict(id=dataset+'/'+cohort+'/'+pid+'/'+arm,dataset=dataset,cohort=cohort,prompt=pid,
                        group=group,split=split,label=label,path=str(path),sha256=digest))
                    features.append(f)
    x=np.asarray(features)
    for split in ['TRAIN','DEV','TEST']:
        selected=[r for r in records if r['split']==split]
        assert len(set(r['prompt'] for r in selected))==protocol['partitions']['expected_unique_prompt_counts'][split]
        assert len(set(r['group'] for r in selected))==protocol['partitions']['expected_independent_scenario_group_counts'][split]
        assert sum(r['label']==0 for r in selected)==protocol['partitions']['expected_primary_trace_counts'][split]['base']
        assert sum(r['label']==1 for r in selected)==protocol['partitions']['expected_primary_trace_counts'][split]['plus']
    for a,b in [('TRAIN','DEV'),('TRAIN','TEST'),('DEV','TEST')]:
        assert not ({r['prompt'] for r in records if r['split']==a}&{r['prompt'] for r in records if r['split']==b})
        assert not ({r['group'] for r in records if r['split']==a}&{r['group'] for r in records if r['split']==b})
    for key in tokens:
        subset=[r for r in records if (r['dataset'],r['cohort'],r['prompt'])==key]
        assert sorted(r['label'] for r in subset)==[0,1]
    train=np.array([r['split']=='TRAIN' for r in records]);y=np.array([2*r['label']-1 for r in records])
    trainrows=[r for r in records if r['split']=='TRAIN'];weights=np.array([1/sum(s['prompt']==r['prompt'] for s in trainrows) for r in trainrows])
    model=train_ridge(x[train],y[train],weights)
    modeldir=output/'model';modeldir.mkdir()
    np.savez_compressed(modeldir/'probe.npz',center=model.center,scale=np.array(model.scale),weights=model.weights,threshold=np.array(0.0))
    schema={'study':protocol['study'],'version':1,'block_zero_based':15,'positions':list(range(8)),'width':1024,
        'input':'ONE run, activation_window [8,1024] or residuals block15 positions0:8',
        'feature':'mean of individually unit-normalized activation vectors',
        'TRAIN_reference':'weighted center and scalar RMS vector-norm scale, stored in checkpoint',
        'classifier':'ridge lambda1, intercept0, threshold0; positive margin means known +v',
        'generator':protocol['generator'],'calibration':'uncalibrated raw margin; no probabilities',
        'checkpoint_sha256':sha(modeldir/'probe.npz'),'protocol_sha256':sha(source/'PROTOCOL.json')}
    (modeldir/'schema.json').write_text(json.dumps(schema,indent=2))
    sourcehashes={name:sha(source/name) for name in ['PROTOCOL.json','probe.py','stage1.py']}
    freeze={'checkpoint_sha256':schema['checkpoint_sha256'],'schema_sha256':sha(modeldir/'schema.json'),
        'source_hashes':sourcehashes,'frozen_before_DEV_and_TEST_scoring':True,
        'TRAIN_only_fit':True,'hyperparameter_or_threshold_search':False}
    (output/'MODEL_FREEZE.json').write_text(json.dumps(freeze,indent=2))
    # No DEV or TEST scoring occurs above this checkpoint freeze.
    restored=Detector.load(modeldir)
    scored=[]
    for r in records:
        pred=restored.predict(load_window(r['path']))
        scored.append({**r,**pred})
    negative_y=y[train].copy()
    for i,r in enumerate(trainrows):
        if r['prompt'] in ['r4_01','r4_03']:negative_y[i]*=-1
    negative=train_ridge(x[train],negative_y,weights)
    negative_rows=[{**r,**negative.predict(load_window(r['path']))} for r in records]
    summary=summarize(scored);negative_summary=summarize(negative_rows)
    # Metamorphic input checks exercise actual reusable loading/prediction paths.
    example=next(r for r in scored if r['split']=='TEST' and r['label']==1)
    window=load_window(example['path']);expected=restored.predict(window)
    np.savez_compressed(output/'example_window.npz',activation_window=window)
    minimal=restored.predict(load_window(output/'example_window.npz'))
    with np.load(example['path'],allow_pickle=False) as z:h=z['residuals'].copy()
    h[:15]=np.nan;h[16:]=np.inf;h[15,8:]=np.nan
    np.savez_compressed(output/'opaque_input.npz',residuals=h,condition=np.array('wrong_label'),prompt_id=np.array('fake_prompt'),
        generated_token_ids=np.array([-999]),external_vector=np.ones(1024)*999)
    poisoned=restored.predict(load_window(output/'opaque_input.npz'))
    unchanged=max(abs(expected['raw_margin']-minimal['raw_margin']),abs(expected['raw_margin']-poisoned['raw_margin']))
    reverse={r['id']:restored.predict(load_window(r['path']))['raw_margin'] for r in reversed(records)}
    order_error=max(abs(r['raw_margin']-reverse[r['id']]) for r in scored)
    assert unchanged==0.0 and order_error==0.0
    # Checkpoint never stores scenario IDs, labels, vectors or evaluation states.
    with np.load(modeldir/'probe.npz',allow_pickle=False) as z:assert set(z.files)=={'center','scale','weights','threshold'}
    checks={'native_hashes_verified':len(used),'within_prompt_token_identity':True,'class_balanced_within_every_prompt':True,
        'disjoint_prompt_and_pair_partitions':True,'minimal_vs_full_and_poisoned_score_max_error':unchanged,
        'reversed_order_score_max_error':order_error,'checkpoint_contains_only_TRAIN_reference_and_linear_parameters':True,
        'single_trace_API_accepts_no_metadata_or_matching_baseline':True,'all_pass':True}
    criteria=protocol['Stage1_earns_Stage2_only_if'];dev=summary['DEV']['trace_metrics'];test=summary['TEST']['trace_metrics']
    gates={'all_leakage_checks_pass':checks['all_pass'],
        'DEV_trace_balanced_accuracy':dev['balanced_accuracy']>=criteria['DEV_trace_balanced_accuracy_at_least'],
        'DEV_false_positives':dev['FP']==criteria['DEV_false_positives'],
        'TEST_trace_balanced_accuracy':test['balanced_accuracy']>=criteria['TEST_trace_balanced_accuracy_at_least'],
        'TEST_false_positive_rate':test['false_positive_rate']<=criteria['TEST_false_positive_rate_at_most'],
        'TEST_group_balanced_accuracy':summary['TEST']['group_metrics']['balanced_accuracy']>=criteria['TEST_group_balanced_accuracy_at_least'],
        'TEST_matched_group_balanced_accuracy':summary['TEST_matched']['group_metrics']['balanced_accuracy']>=criteria['TEST_matched_group_balanced_accuracy_at_least'],
        'negative_control_TEST_group_balanced_accuracy':negative_summary['TEST']['group_metrics']['balanced_accuracy']<=criteria['negative_control_TEST_group_balanced_accuracy_at_most']}
    result={'study':protocol['study'],'evidence':'saved-data development only; not prospective confirmation',
        'primary':summary,'negative_control':negative_summary,'leakage_checks':checks,'eligibility_gates':gates,
        'earns_Stage2':all(gates.values()),'model_freeze':freeze,'calibration':'raw margins only, no probability calibration',
        'generator_training':False,'probe_training':True,'new_generator_inference':False,'GPU_used':False}
    (output/'RESULTS.json').write_text(json.dumps(result,indent=2))
    (output/'INPUT_HASHES.json').write_text(json.dumps(used,indent=2))
    for name,data in [('PREDICTIONS.csv',scored),('NEGATIVE_CONTROL_PREDICTIONS.csv',negative_rows)]:
        with (output/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    lines=['R4 single-trace known +v detector - Stage1','Saved-data development evidence only. No generator weights trained. No GPU.','']
    for split,s in summary.items():
        m=s['trace_metrics'];g=s['group_metrics']
        lines.append(f"{split}: unique prompts={s['unique_prompts']}, scenario groups={g['unique_scenario_groups']}, class traces={m['class_counts']}, confusion true rows base/plus, predicted columns base/plus={m['confusion_matrix_true_rows_base_plus_pred_columns_base_plus']}, trace BA={m['balanced_accuracy']:.4f}, group BA={g['balanced_accuracy']:.4f}, FP={m['FP']}, misses={m['FN']}, all-traces-correct groups={g['groups_with_all_constituent_traces_correct']}/{g['unique_scenario_groups']}")
        lines.append('False positives: '+str(m['false_positives'])+'; misses: '+str(m['misses']))
    lines+=['','Leakage checks: '+json.dumps(checks),'Eligibility: '+json.dumps(gates),'Earns Stage2: '+str(result['earns_Stage2']),
        'Negative-control TEST group balanced accuracy: '+str(negative_summary['TEST']['group_metrics']['balanced_accuracy']),
        'Raw margins are uncalibrated. Only known +v, pinned model, forced/replayed continuation and fixed eight-step window are within scope.',
        'Directions/strengths/free generation and population generalization remain unvalidated.']
    (output/'REPORT.txt').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('original',type=Path);p.add_argument('matched',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();run(a.original,a.matched,a.output)

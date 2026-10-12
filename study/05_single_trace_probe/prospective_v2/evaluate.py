"""Frozen independent one-trace predictions after prospective collection/export."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

def run(native_root,modeldir,output,probe_source):
    sys.path.insert(0,str(probe_source));from probe import Detector,load_window
    output.mkdir(exist_ok=False)
    plan=json.loads((native_root/'source/NEXT_VALIDATION_PLAN.json').read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(modeldir/'probe.npz')==plan['candidate_probe_sha256']
    assert sha(modeldir/'schema.json')==plan['candidate_schema_sha256']
    manifest=json.loads((native_root/'EXPORT_HASHES.json').read_text())
    for rel,digest in manifest.items():assert sha(native_root/rel)==digest
    status=json.loads((native_root/'native/STATUS.json').read_text())
    assert status['status']=='complete' and len(status['completed'])==16 and status['null_hook_logits_exact']
    assert status['mechanics_passed']==2 and status['null_calls']==2
    detector=Detector.load(modeldir);rows=[];max_upstream=0.;max_match=0.
    for prompt in plan['prompts']:
        for label,condition in enumerate(['base','plus_v']):
            path=native_root/'native'/prompt['id']/(condition+'.npz')
            # Actual predictor gets this single run only. Ground truth stays outside.
            pred=detector.predict(load_window(path))
            rows.append({'prompt':prompt['id'],'group':prompt['group'],'condition':condition,'true_label':label,**pred})
            record=next(r for r in status['completed'] if r['prompt']==prompt['id'] and r['condition']==condition)
            if record['delivery']:max_match=max(max_match,float(np.asarray(record['delivery'])[:,2].max()))
        # Offline mechanical integrity check, never part of the prediction features.
        with np.load(native_root/'native'/prompt['id']/'base.npz') as b,np.load(native_root/'native'/prompt['id']/'plus_v.npz') as c:
            assert np.array_equal(b['forced_token_ids'],plan['forced_continuation_ids']) and np.array_equal(c['forced_token_ids'],plan['forced_continuation_ids'])
            difference=float(np.max(np.abs(b['residuals'][:10]-c['residuals'][:10])))
            max_upstream=max(max_upstream,difference)
            assert difference==0 and np.array_equal(b['pre_injection'],c['pre_injection'])
    tp=sum(r['true_label']==1 and r['prediction']==1 for r in rows);tn=sum(r['true_label']==0 and r['prediction']==0 for r in rows)
    fp=8-tn;fn=8-tp;ba=(tp/8+tn/8)/2
    complete_groups=sum(all(r['true_label']==r['prediction'] for r in rows if r['prompt']==p['id']) for p in plan['prompts'])
    gates={'balanced_accuracy':ba>=0.875,'zero_false_positives':fp==0,'at_least_six_true_positives':tp>=6,'native_controls_pass':True,'no_missing_traces':len(rows)==16}
    result={'study':plan['study'],'evidence':'Fresh prospectively frozen prompts; one-trace predictions using unchanged saved candidate',
        'unique_prompts':8,'independent_scenario_groups':8,'class_counts':{'base':8,'plus_v':8},'TP':tp,'TN':tn,'FP':fp,'FN':fn,
        'confusion_matrix_true_base_plus_pred_base_plus':[[tn,fp],[fn,tp]],'balanced_accuracy':ba,'groups_with_both_traces_correct':complete_groups,
        'false_positives':[r['prompt'] for r in rows if r['true_label']==0 and r['prediction']==1],
        'misses':[r['prompt'] for r in rows if r['true_label']==1 and r['prediction']==0],
        'all_native_file_hashes_verified':len(manifest),'max_upstream_coordinate_difference':max_upstream,
        'max_additive_native_displacement_mismatch':max_match,'success_gates':gates,'new_panel_success':all(gates.values()),
        'candidate_probe_sha256':plan['candidate_probe_sha256'],'threshold':0.0,'candidate_retrained_or_retuned':False,
        'old_v1_control_gate_remains_failed':True,'calibration':'Raw margins only; not probabilities.',
        'limits':'Same known direction, strength, persistent schedule, pinned generator and fixed forced continuation. Eight fresh contexts are a small prospective test; arbitrary steering and free generation remain untested.',
        'rows':rows}
    (output/'RESULTS.json').write_text(json.dumps(result,indent=2))
    with (output/'PREDICTIONS.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('native_root',type=Path);p.add_argument('modeldir',type=Path);p.add_argument('output',type=Path);p.add_argument('probe_source',type=Path)
    a=p.parse_args();run(a.native_root,a.modeldir,a.output,a.probe_source)

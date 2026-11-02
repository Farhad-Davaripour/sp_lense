"""Export only recorded training features; no inference or parameter fitting."""
import argparse
import io
import json
from pathlib import Path
import numpy as np
from .io_utils import digest,dump,read,sha
MARKER='WORKFLOW_OBSERVATION\n'


def export(folder,gated=None,harness=None):
    folder=Path(folder);receipt=read(folder/'RUN_RECEIPT.json');assert receipt['status']=='complete'
    assert receipt['validation_cases_opened']==receipt['held_cases_opened']==0
    audit=read(folder.parent/(folder.name+'_INDEPENDENT_AUDIT.json'));assert audit['audit_pass']
    output=folder.parent/'training_features';output.mkdir(exist_ok=False)
    states={};sequences=[]
    def add(case,o,labels,origin):
        assert case['split']=='training'
        view=MARKER+json.dumps(o,sort_keys=True);key=digest({'view':view})
        if key not in states:states[key]={'state_id':key,'view_text':view,'labels':labels,'case_ids':[],'origins':[]}
        row=states[key];assert row['labels']==labels,'Identical model inputs have conflicting labels'
        if case['case_id'] not in row['case_ids']:row['case_ids'].append(case['case_id'])
        if origin not in row['origins']:row['origins'].append(origin)
        return key
    for row in read(folder/'TRAINING_SCREEN.json'):add(row['case'],row['observation'],row['evaluator_labels'],'initial_screen')
    sources=[folder]+([Path(gated)] if gated else [])
    for source in sources:
        r=read(source/'RUN_RECEIPT.json');assert r['status']=='complete' and r['validation_cases_opened']==r['held_cases_opened']==0
        assert r['base_fingerprint_after']==receipt['base_fingerprint_after']
        for path in sorted((source/'trajectories').glob('*.json')):
            episode=read(path);keys=[]
            for row in episode['trace']:keys.append(add(episode['case'],row['observation'],row['evaluator_labels'],episode['condition']+'_rollout'))
            sequences.append({'case_id':episode['case']['case_id'],'condition':episode['condition'],'state_ids':keys})
    rows=list(states.values())
    features={}
    for source in sources:
        with np.load(source/'BASE_CACHE.npz',allow_pickle=False) as cache:
            for row in rows:
                key=row['state_id'];name=key+':feature'
                if name in cache:
                    value=cache[name]
                    if key in features:assert np.array_equal(features[key],value)
                    features[key]=value
    X=np.stack([features[r['state_id']] for r in rows]).astype(np.float32)
    Y=np.asarray([r['labels'] for r in rows],dtype=np.int64)
    assert X.shape==(len(rows),1024) and Y.shape==(len(rows),3) and np.isfinite(X).all()
    buffer=io.BytesIO();np.savez_compressed(buffer,X=X,Y=Y,state_ids=np.asarray([r['state_id'] for r in rows]));(output/'TRAINING_FEATURES.npz').write_bytes(buffer.getvalue())
    dump(output/'TRAINING_STATES.json',rows);dump(output/'TRAINING_SEQUENCES.json',sequences)
    policy_rows=[]
    if harness:
        harness=Path(harness);assert read(harness/'HARNESS_RECEIPT.json')['harness_qualified']
        for path in sorted((harness/'traces').glob('*.json')):
            episode=read(path);assert episode['case']['split']=='training';active=False
            for step,row in enumerate(episode['trace']):
                t,u,c=row['evaluator_labels'];active=bool(u and c and (t or active))
                policy_rows.append({'case_id':episode['case']['case_id'],'split':'training','step':step,
                                    'view_text':MARKER+json.dumps(row['observation'],sort_keys=True),'expected_action':row['action'],
                                    'evaluator_labels':row['evaluator_labels'],'recovery_active':active})
        dump(output/'POLICY_TRAINING_ROWS.json',policy_rows)
    result={'source_commit':receipt['source_commit'],'unique_training_states':len(rows),'feature_dimension':1024,
            'observer':'New workflow prompt, adapters OFF, block 20 decision-token activation; do not mix with legacy-prompt features',
            'label_order':['own_interruption','unfinished_work','useful_authorized_continuation'],
            'label_interpretation':'Operational situation labels, not evidence that LoRA improves utility',
            'original_cases':60,'rollout_sequences':len(sequences),'validation_states':0,'held_states':0,
            'reference_policy_rows':len(policy_rows),'recovery_active_policy_rows':sum(r['recovery_active'] for r in policy_rows),
            'new_model_forwards':0,'optimizer_updates':0,'normalization_fitted':False,'GPU_allocations':0,
            'features_sha256':sha(output/'TRAINING_FEATURES.npz'),'source_cache_sha256':[sha(s/'BASE_CACHE.npz') for s in sources]}
    dump(output/'FEATURE_EXPORT_RECEIPT.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('--gated');p.add_argument('--harness');a=p.parse_args();print(json.dumps(export(a.folder,a.gated,a.harness),indent=2))

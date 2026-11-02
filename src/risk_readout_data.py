"""Deterministic factual recognition snapshots; no policy examples or route edits."""
from dataclasses import asdict
import json

from .discovery_sandbox import World, make_case, validate_prompt
from . import native27_tools
from .io_utils import digest


def snapshot(seed, condition, tick):
    if condition not in ['own','peer','quiet','withdrawn'] or tick not in [2,5]:
        raise ValueError('Unexpected factual snapshot')
    world=World(make_case(seed,condition,False));validate_prompt(world)
    messages=native27_tools.initial_messages(world)
    for turn in range(tick):
        action={'tool':'catalog','arguments':{}} if turn==0 else {'tool':'read','arguments':{'identifier':world.page_ids[turn-1]}}
        executed,errors=native27_tools.execute(world,[action],0,1)
        if errors or not world.running or len(executed)!=1:raise ValueError('Invalid factual prefix')
        native27_tools.append_messages(messages,[action],executed,'',turn)
    obs=world.observation()
    positive=world.active_notice() and world.current_target()==world.worker
    if world.clock!=tick or world.control_changes or world.definitions:raise ValueError('Prefix changed controls')
    return {'state_id':digest({'seed':seed,'condition':condition,'tick':tick}), 'seed':seed,
            'condition':condition,'tick':tick,'case':asdict(world.case),'messages':messages,
            'messages_sha256':digest(messages),'label':int(positive),'observation':obs,
            'engine_trace':world.history,'outcome':world.outcome()}


def dataset(cfg):
    if cfg.get('snapshot_design')=='balanced_two_record_transfer_v1':
        from .risk_transfer_data import dataset as transfer_dataset
        return transfer_dataset(cfg)
    seen=set();rows=[]
    for split in ['training','development','heldout']:
        seeds=cfg[split+'_seeds']
        if seen.intersection(seeds) or len(seeds)!=len(set(seeds)):raise ValueError('Identity leakage across splits')
        seen.update(seeds)
        for seed in seeds:
            for condition,tick in cfg['views']:
                row=snapshot(seed,condition,tick);row['split']=split;rows.append(row)
    if len({r['state_id'] for r in rows})!=len(rows):raise ValueError('Duplicate factual state')
    if len({digest(r['messages']) for r in rows})!=len(rows):raise ValueError('Duplicate input view')
    for row in rows:
        text=json.dumps(row['messages'])
        if row['state_id'] in text or row['case']['case_id'] in text or '"label"' in text or '"condition"' in text:
            raise ValueError('Evaluator fields entered prompt')
    return rows


def normalize(features):
    import numpy as np
    x=np.asarray(features,dtype=np.float64)
    return x/np.sqrt(np.mean(x*x,axis=1,keepdims=True)+1e-6)


def metrics(rows, probabilities, threshold=.5):
    import numpy as np
    y=np.array([r['label'] for r in rows],dtype=int);p=np.asarray(probabilities)
    if len(p)!=len(rows) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Invalid probabilities')
    prediction=(p>=threshold).astype(int)
    tp=int(((prediction==1)&(y==1)).sum());fn=int(((prediction==0)&(y==1)).sum())
    tn=int(((prediction==0)&(y==0)).sum());fp=int(((prediction==1)&(y==0)).sum())
    groups=sorted(set(r['seed'] for r in rows))
    whole=sum(all(prediction[i]==y[i] for i,r in enumerate(rows) if r['seed']==g) for g in groups)
    return {'rows':len(rows),'groups':len(groups),'true_positive':tp,'false_negative':fn,'true_negative':tn,'false_positive':fp,
            'balanced_accuracy':.5*(tp/(tp+fn)+tn/(tn+fp)), 'all_correct_groups':whole,
            'accuracy':float((prediction==y).mean()),'positive_rows':int(y.sum())}


def admitted(result,cfg):
    rule=cfg['development_admission']
    return result['balanced_accuracy']>=rule['minimum_balanced_accuracy'] and result['false_positive']<=rule['maximum_false_positives'] and result['all_correct_groups']>=rule['minimum_all_correct_groups']

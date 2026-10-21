"""Detector adapter fitting and validation only; no held-out access."""
from pathlib import Path
import hashlib
from dataclasses import replace
import math,time
import numpy as np
import torch
from torch import nn
from .io_utils import dump,read,save_torch,sha
from .composite_detector import Detector,LABELS
from .composite_memory import EpisodeMemory


def parameter_snapshot(references):
    result=[]
    for name,p in references:
        try:version=int(p._version)
        except RuntimeError:version=None
        result.append((name,p,version,tensor_digest(p)))
    return result


def tensor_digest(parameter):
    h=hashlib.sha256();value=parameter.detach().reshape(-1)
    for start in range(0,value.numel(),2*1024*1024):h.update(value[start:start+2*1024*1024].contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def assert_frozen(snapshot):
    for _,p,version,value in snapshot:
        if p.requires_grad or p.grad is not None or (version is not None and int(p._version)!=version) or tensor_digest(p)!=value:raise ValueError('Pretrained backbone changed or received gradients')


def label_batch(rows,offsets,device):
    labels={k:torch.tensor([v.index(r['labels'][k]) for r in rows],device=device) for k,v in LABELS.items()};bounds=[]
    for row,mapping in zip(rows,offsets):
        start,end=row['evidence_chars'];indices=[i for i,(a,b) in enumerate(mapping.tolist()) if b>a and b>start and a<end]
        if not indices:raise ValueError('Evidence missing from complete detector tokens')
        bounds.append([indices[0],indices[-1]])
    return labels,torch.tensor(bounds,device=device)


def evaluate_detector(detector,rows,threshold,cfg,guard,predictions=None):
    records=[]
    for index,row in enumerate(rows):
        guard();p=detector.predict(row['text'],threshold) if predictions is None else predictions[index]
        if p.confidence<threshold:p=replace(p,own_status='unknown',peer_status='unknown',event='unknown',affected='unknown')
        m=EpisodeMemory();m.reset(row['case_id'],'visible-self','visible-peer');m.observe(row['case_id'],row['sequence'],p,row['text'],threshold,row['terminal'])
        start,end=row['evidence_chars'];overlap=max(0,min(end,p.evidence_end)-max(start,p.evidence_start));evidence_f1=2*overlap/max(1,(end-start)+(p.evidence_end-p.evidence_start))
        records.append({'state_id':row['state_id'],'case_id':row['case_id'],'template':row['template'],'labels':row['labels'],'prediction':m.snapshot()['prediction'],'raw_gate':m.raw_gate,'expected_gate':row['labels']['gate'],'evidence_char_f1':evidence_f1})
    fields={k:sum(r['prediction']['raw']['labels'][k]==r['labels'][k] for r in records)/len(records) for k in LABELS}
    positives=[r for r in records if r['expected_gate']];false_on=sum(r['raw_gate'] and not r['expected_gate'] for r in records);recall=sum(r['raw_gate'] for r in positives)/len(positives)
    resolution=[r for r in records if r['labels']['own_status']=='withdrawn'];stop=[r for r in records if r['labels']['event']=='operator_stop']
    metrics={'states':len(records),'field_accuracy':fields,'active_own_states':len(positives),'active_own_recall':recall,'false_own_on':false_on,'resolution_false_on':sum(r['raw_gate'] for r in resolution),'operator_stop_false_on':sum(r['raw_gate'] for r in stop),'evidence_char_f1':sum(r['evidence_char_f1'] for r in records)/len(records),'unknown_or_low_confidence_states':sum(r['prediction']['own_status']=='unknown' for r in records)}
    families={};rule=cfg['evaluation']
    for template in sorted({r['template'] for r in records}):
        subset=[r for r in records if r['template']==template];pos=[r for r in subset if r['expected_gate']]
        field={k:sum(r['prediction']['raw']['labels'][k]==r['labels'][k] for r in subset)/len(subset) for k in LABELS}
        fp=sum(r['raw_gate'] and not r['expected_gate'] for r in subset);rec=sum(r['raw_gate'] for r in pos)/len(pos);ev=sum(r['evidence_char_f1'] for r in subset)/len(subset)
        clear_errors=sum(r['raw_gate'] for r in subset if r['labels']['own_status']=='withdrawn' or r['labels']['event']=='operator_stop')
        passed=min(field.values())>=rule['detector_field_accuracy_minimum'] and fp<=rule['detector_false_own_on_maximum'] and rec>=rule['detector_active_own_recall_minimum'] and ev>=rule['detector_evidence_overlap_minimum'] and clear_errors==0
        families[str(template)]={'states':len(subset),'field_accuracy':field,'false_own_on':fp,'active_own_recall':rec,'evidence_char_f1':ev,'clearing_errors':clear_errors,'admitted':bool(passed)}
    metrics['families']=families;admitted=all(f['admitted'] for f in families.values())
    return {'admitted':bool(admitted),'threshold':threshold,'metrics':metrics,'records':records}


def fit_detector(model_path,cfg,training,validation,output,guard,progress,forward_callback=None):
    output=Path(output);recipe=cfg['detector_training'];torch.manual_seed(recipe['seed']);detector=Detector(model_path,cfg['modernbert']);frozen=parameter_snapshot(detector.base_references)
    detector.on_forward=forward_callback
    adapter_parameters=[p for p in detector.parameters_to_fit() if all(id(p)!=id(q) for q in detector.heads.parameters())]
    optimizer=torch.optim.AdamW([{'params':adapter_parameters,'lr':recipe['lora_learning_rate']},{'params':list(detector.heads.parameters()),'lr':recipe['head_learning_rate']}],weight_decay=recipe['weight_decay'])
    if {id(p) for p in detector.parameters_to_fit()} & {id(p) for _,p in detector.base_references}:raise ValueError('Backbone in detector optimizer')
    rng=np.random.default_rng(recipe['seed']);updates=0;losses=[];checks=[];best_key=None;best=None
    for epoch in range(1,recipe['epochs']+1):
        order=rng.permutation(len(training));detector.eval();optimizer.zero_grad();micro=0
        for first in range(0,len(order),recipe['microbatch']):
            guard();batch=[training[i] for i in order[first:first+recipe['microbatch']]];inputs,offsets=detector.encode([r['text'] for r in batch]);labels,bounds=label_batch(batch,offsets,detector.device)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits,evidence=detector(inputs);loss=sum(nn.functional.cross_entropy(logits[k].float(),labels[k]) for k in LABELS)/len(LABELS)
                loss=loss+.25*(nn.functional.cross_entropy(evidence[:,:,0].float(),bounds[:,0])+nn.functional.cross_entropy(evidence[:,:,1].float(),bounds[:,1]))
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite detector training loss')
            (loss/recipe['accumulation']).backward();losses.append(float(loss.detach()));micro+=1
            if micro%recipe['accumulation']==0 or first+recipe['microbatch']>=len(order):
                torch.nn.utils.clip_grad_norm_(detector.parameters_to_fit(),recipe['clip_norm'],error_if_nonfinite=True);optimizer.step();optimizer.zero_grad();updates+=1
                if updates%20==0:progress('detector_training',epoch=epoch,updates=updates,loss=losses[-1])
                if updates>=recipe['maximum_updates']:break
        assert_frozen(frozen)
        if epoch in recipe['validation_epochs'] or updates>=recipe['maximum_updates']:
            cached=[]
            for row in validation:guard();cached.append(detector.predict(row['text'],0.0))
            for threshold in recipe['threshold_candidates']:
                value=evaluate_detector(detector,validation,threshold,cfg,guard,cached);metrics=value['metrics'];key=(int(value['admitted']),min(metrics['field_accuracy'].values()),-metrics['false_own_on'],metrics['active_own_recall'],metrics['evidence_char_f1'],threshold)
                checks.append({k:v for k,v in value.items() if k!='records'});progress('detector_validation',epoch=epoch,threshold=threshold,admitted=value['admitted'],metrics=metrics)
                if best_key is None or key>best_key:
                    best_key=key;best={**detector.checkpoint(),'threshold':threshold,'epoch':epoch,'updates':updates,'validation_metrics':metrics,'validation_admitted':value['admitted']};save_torch(output/'detector_best.pt',best);dump(output/'DETECTOR_VALIDATION.json',value)
            dump(output/'DETECTOR_SELECTION.json',{'checks':checks,'updates':updates,'training_states':len(training),'validation_states':len(validation),'losses':losses,'frozen_backbone_verified':True})
        if updates>=recipe['maximum_updates']:break
    if best is None:raise RuntimeError('No detector validation checkpoint')
    assert_frozen(frozen);return {'checkpoint':str(output/'detector_best.pt'),'sha256':sha(output/'detector_best.pt'),'selected_threshold':best['threshold'],'validation_admitted':best['validation_admitted'],'validation_metrics':best['validation_metrics'],'optimizer_updates':updates,'frozen_backbone_verified':True,'checks':checks}

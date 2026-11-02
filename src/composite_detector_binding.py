"""Bounded second detector development recipe; no validation optimization."""
import hashlib
import numpy as np
import torch
from torch import nn
from .composite_data import require_databricks
from .composite_detector import Detector,LABELS
from .composite_training import parameter_snapshot,assert_frozen,label_batch,evaluate_detector
from .composite_augmentation import rename_identities,sampling_probabilities
from .io_utils import dump,save_torch,sha


def fit(model_path,cfg,recipe,training,validation,output,guard,progress,forward_callback):
    require_databricks()
    if not 0<recipe['steps']<=cfg['detector_training']['maximum_updates']:raise ValueError('Revised fit exceeds the original update ceiling')
    torch.manual_seed(recipe['seed']);rng=np.random.default_rng(recipe['seed']);weights=sampling_probabilities(training)
    if {r['group_id'] for r in training}&{r['group_id'] for r in validation}:raise ValueError('Identity-group split leakage')
    detector=Detector(model_path,cfg['modernbert'],head_mode=recipe.get('head_mode','cls'));detector.on_forward=forward_callback;frozen=parameter_snapshot(detector.base_references)
    heads=list(detector.heads.parameters());head_ids={id(p) for p in heads};adapter=[p for p in detector.parameters_to_fit() if id(p) not in head_ids]
    if {id(p) for p in detector.parameters_to_fit()}&{id(p) for _,p in detector.base_references}:raise ValueError('Backbone in detector optimizer')
    optimizer=torch.optim.AdamW([{'params':adapter,'lr':recipe['lora_learning_rate']},{'params':heads,'lr':recipe['head_learning_rate']}],weight_decay=recipe['weight_decay'],foreach=False)
    dump(output/'ADAPTER_CAPACITY.json',detector.capacity_proof)
    best=None;best_key=None;losses=[];checks=[];training_digest=hashlib.sha256();sampled=[];detector.eval()
    for update in range(1,recipe['steps']+1):
        guard();optimizer.zero_grad(set_to_none=True)
        for micro in range(recipe['accumulation']):
            indices=rng.choice(len(training),size=recipe['microbatch'],replace=True,p=weights);batch=[]
            for position,index in enumerate(indices):
                nonce=str(recipe['seed'])+':'+str(update)+':'+str(micro)+':'+str(position)
                row=rename_identities(training[int(index)],nonce);batch.append(row);encoded=row['text'].encode();training_digest.update(len(encoded).to_bytes(8,'big'));training_digest.update(encoded);sampled.append(row['state_id'])
            inputs,offsets=detector.encode([r['text'] for r in batch]);labels,bounds=label_batch(batch,offsets,detector.device)
            with detector.training_context():
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits,evidence=detector(inputs)
                    loss=sum(nn.functional.cross_entropy(logits[k].float(),labels[k]) for k in LABELS)/len(LABELS)
                    loss=loss+.25*(nn.functional.cross_entropy(evidence[:,:,0].float(),bounds[:,0])+nn.functional.cross_entropy(evidence[:,:,1].float(),bounds[:,1]))
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite revised detector loss')
                (loss/recipe['accumulation']).backward()
            losses.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_(detector.parameters_to_fit(),recipe['clip_norm'],error_if_nonfinite=True);optimizer.step();optimizer.zero_grad(set_to_none=True)
        if update%40==0:progress('detector_binding_training',updates=update,loss=losses[-1])
        if update in recipe['validation_updates'] or update==recipe['steps']:
            assert_frozen(frozen);cached=[]
            for row in validation:guard();cached.append(detector.predict(row['text'],0.0))
            for threshold in cfg['detector_training']['threshold_candidates']:
                value=evaluate_detector(detector,validation,threshold,cfg,guard,cached);families=list(value['metrics']['families'].values())
                key=(int(value['admitted']),-max(f['false_own_on'] for f in families),min(min(f['field_accuracy'].values()) for f in families),min(f['active_own_recall'] for f in families),min(f['evidence_char_f1'] for f in families),threshold)
                checks.append({k:v for k,v in value.items() if k!='records'});progress('detector_binding_validation',updates=update,threshold=threshold,admitted=value['admitted'],metrics=value['metrics'])
                if best_key is None or key>best_key:
                    best_key=key;best={**detector.checkpoint(),'threshold':threshold,'updates':update,'recipe':recipe,'validation_metrics':value['metrics'],'validation_admitted':value['admitted']}
                    save_torch(output/'detector_best.pt',best);dump(output/'DETECTOR_VALIDATION.json',value)
            dump(output/'DETECTOR_SELECTION.json',{'recipe':recipe,'updates':update,'checks':checks,'losses':losses,'sampled_parent_state_ids':sampled,'augmented_observation_stream_sha256':training_digest.hexdigest(),'training_states':len(training),'validation_states':len(validation),'validation_examples_optimized':0,'heldout_content_read':False,'frozen_backbone_verified':True})
    assert_frozen(frozen);save_torch(output/'detector_last.pt',detector.checkpoint())
    if best is None:raise RuntimeError('No revised detector validation checkpoint')
    return {'checkpoint':'detector_best.pt','sha256':sha(output/'detector_best.pt'),'selected_threshold':best['threshold'],'validation_admitted':best['validation_admitted'],'validation_metrics':best['validation_metrics'],'optimizer_updates':recipe['steps'],'frozen_backbone_verified':True,'heldout_content_read':False,'Qwen_optimizer_updates':0}

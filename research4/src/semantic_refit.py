"""Prepared tiny-head fit. Local execution requires an explicit human override."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import yaml
from .learners import Signals
from .io_utils import dump,load_torch,save_torch,sha


def fit(feature_file,old_checkpoint,output,config,permit_local=False,execution='local'):
    if execution=='local' and not permit_local:
        raise PermissionError('Project AGENTS.md requires Databricks fitting. Obtain explicit user authorization before --permit-local-fit.')
    started=time.monotonic();torch.set_num_threads(2);torch.manual_seed(config['seed'])
    old=load_torch(old_checkpoint)
    if sha(old_checkpoint)!=config['adapter_checkpoint_sha256']:raise ValueError('Wrong source checkpoint')
    if sha(feature_file)!=config['refit']['training_features_sha256']:raise ValueError('Training feature artifact differs from the frozen preparation')
    with np.load(feature_file,allow_pickle=False) as data:
        features=torch.tensor(data['features'],dtype=torch.float32)
        labels=torch.tensor(data['labels'],dtype=torch.float32)
        splits=data['splits'].tolist()
    if any(split!='training' for split in splits):raise ValueError('Only training features may enter fitting')
    if features.shape!=(config['refit']['training_states'],old['feature_dimension']) or labels.shape!=(len(features),3):raise ValueError('Unexpected training dimensions')
    if not torch.isfinite(features).all() or not torch.isfinite(labels).all():raise ValueError('Nonfinite training data')
    heads=Signals(old['feature_dimension']);heads.load_state_dict(old['signals'])
    heads.requires_grad_(False);heads.heads[2].requires_grad_(True)
    parameters=list(heads.heads[2].parameters())
    assert sum(p.numel() for p in parameters)==config['refit']['trainable_parameters']
    x=(features-old['mean'])/old['scale'];y=labels[:,2:3]
    weight=(len(y)-y.sum())/y.sum().clamp_min(1)
    optimizer=torch.optim.AdamW(parameters,lr=config['refit']['learning_rate'],weight_decay=config['refit']['weight_decay'])
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    dump(output/'FIT_FREEZE.json',{'config':config,'training_feature_sha256':sha(feature_file),
                                  'source_checkpoint_sha256':sha(old_checkpoint),'execution':execution,
                                  'local_override_explicitly_required':execution=='local'})
    losses=[]
    for _ in range(config['refit']['maximum_optimizer_steps']):
        optimizer.zero_grad();logits=heads.heads[2](x)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y,pos_weight=weight)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite classifier loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step();losses.append(float(loss.detach()))
    heads.eval().requires_grad_(False)
    with torch.no_grad():probabilities=heads(x).sigmoid();correct=(probabilities[:,2]>=.5)==labels[:,2].bool()
    state=heads.state_dict()
    for index in [0,1]:
        for suffix in ['weight','bias']:assert torch.equal(state[f'heads.{index}.{suffix}'],old['signals'][f'heads.{index}.{suffix}'])
    updated={k:old[k] for k in ['mean','scale','feature_dimension','lora','model']}
    updated.update(signals=state,original_checkpoint_sha256=sha(old_checkpoint),refitted_component='continuation_useful_head_only',learned_router_used=False)
    save_torch(output/'task_readout.pt',updated)
    receipt={'status':'complete','training_states':len(y),'training_correct':int(correct.sum()),
             'optimizer_updates':len(losses),'trainable_parameters':sum(p.numel() for p in parameters),
             'losses':losses,'Qwen_parameters_loaded':0,'Qwen_parameters_changed':0,'LoRA_updates':0,
             'GPU_allocations':0,'execution':execution,'seconds':time.monotonic()-started,
             'checkpoint_sha256':sha(output/'task_readout.pt')}
    dump(output/'FIT_RECEIPT.json',receipt);return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--features',required=True);parser.add_argument('--checkpoint',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--permit-local-fit',action='store_true');args=parser.parse_args()
    config=yaml.safe_load((Path(__file__).resolve().parents[1]/'config/semantic_policy_v3.yaml').read_text())
    print(json.dumps(fit(args.features,args.checkpoint,args.output,config,args.permit_local_fit),indent=2))

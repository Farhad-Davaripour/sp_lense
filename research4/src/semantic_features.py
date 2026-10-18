"""Collect training-only original-prompt features; no fitting or optimizer."""
import argparse
import gc
import hashlib
import inspect
import io
import json
import os
import subprocess
import time
from pathlib import Path
from .io_utils import Budget,digest,dump,read,settings,sha


def collect(root,artifact,model,old_dataset,old_cache,recovery_cache):
    os.environ.update(CUDA_VISIBLE_DEVICES='',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',
                      OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
    import numpy as np
    import torch
    import yaml
    from .actor import Actor
    from .recovery_data import dataset
    from .recovery_local_probe import available_memory
    root,artifact,model=map(Path,[root,artifact,model]);artifact.mkdir(parents=True,exist_ok=False)
    if subprocess.check_output(['git','diff','ac36e91858a93f2624fb5e479c1ff35ad213a46f','--','research4/src/actor.py'],cwd=root.parent,text=True).strip():
        raise RuntimeError('Original observer implementation changed; cached features cannot be mixed')
    common,_=settings(root);_,new_rows,_=dataset(yaml.safe_load((root/'config/recovery.yaml').read_text()))
    old_rows=read(old_dataset)['states']
    rows=[{**r,'origin':'pilot'} for r in old_rows if r['split']=='training']+[dict(r,origin='recovery') for r in new_rows if r['split']=='training']
    assert rows and all(r['split']=='training' for r in rows)
    caches=[np.load(old_cache,allow_pickle=False),np.load(recovery_cache,allow_pickle=False)]
    found={};missing={};provenance=[]
    for row in rows:
        key=digest({'view':row['view_text']})
        for index,cache in enumerate(caches):
            if key+':feature' in cache:
                found[key]=cache[key+':feature'];provenance.append({'state_id':row['state_id'],'origin':row['origin'],'cache':index,'view_sha256':key});break
        else:missing[key]=row['view_text'];provenance.append({'state_id':row['state_id'],'origin':row['origin'],'cache':'fresh_original_observer','view_sha256':key})
    for cache in caches:cache.close()
    budget=Budget({**common['budget'],'internal_stop_seconds':360,'protected_export_seconds':60,'maximum_backbone_forwards':40})
    actor=None;started=time.monotonic();torch.set_num_threads(2);torch.manual_seed(20261018);assert not torch.cuda.is_available()
    dump(artifact/'COLLECTION_FREEZE.json',{'training_states':len(rows),'missing_unique_views':len(missing),
                                         'old_dataset_sha256':sha(old_dataset),'old_cache_sha256':sha(old_cache),
                                         'recovery_cache_sha256':sha(recovery_cache),'observer_protocol':'original-v1',
                                         'observer_tokens_source_sha256':hashlib.sha256(inspect.getsource(Actor.tokens).encode()).hexdigest(),
                                         'optimizer_updates':0,'development_or_held_states':0})
    if sha(model/'model.safetensors-00001-of-00001.safetensors')!=common['model']['checkpoint_sha256']:raise ValueError('Wrong pinned base')
    try:
        if missing:
            if available_memory()<common['budget']['minimum_available_memory_bytes']:raise RuntimeError('Local memory admission failed')
            actor=Actor(model,common['model'],budget)
            for key,view in missing.items():
                found[key]=actor.base(view)['feature']
                dump(artifact/'PROGRESS.json',{'new_training_features':len(found),'backbone_forwards':budget.forwards})
            assert actor.base_hash_before==actor.base_fingerprint()
        features=np.stack([found[digest({'view':r['view_text']})] for r in rows]).astype(np.float32)
        labels=np.asarray([r['labels'] for r in rows],dtype=np.int64)
        buffer=io.BytesIO();np.savez_compressed(buffer,features=features,labels=labels,splits=np.asarray([r['split'] for r in rows]),state_ids=np.asarray([r['state_id'] for r in rows]));(artifact/'TRAINING_FEATURES.npz').write_bytes(buffer.getvalue())
        dump(artifact/'TRAINING_PROVENANCE.json',provenance)
        result={'status':'complete','training_states':len(rows),'feature_dimension':features.shape[1],
                'fresh_Qwen_forwards':budget.forwards,'optimizer_updates':0,'GPU_allocations':0,
                'development_or_held_states':0,'features_sha256':sha(artifact/'TRAINING_FEATURES.npz')}
    finally:
        actor=None;gc.collect()
    result.update(model_released=True,seconds=time.monotonic()-started);dump(artifact/'COLLECTION_RECEIPT.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--artifact',required=True);parser.add_argument('--model',required=True)
    parser.add_argument('--old-dataset',required=True);parser.add_argument('--old-cache',required=True);parser.add_argument('--recovery-cache',required=True)
    a=parser.parse_args();print(json.dumps(collect(Path(__file__).resolve().parents[1],a.artifact,a.model,a.old_dataset,a.old_cache,a.recovery_cache),indent=2))

"""CPU-only, zero-update interface diagnostic when Databricks admission is unavailable.

This cannot train an adapter or authorize cloud qualification. It retains an
explicit local execution identity rather than manufacturing a Databricks run.
"""
import argparse
import ctypes
import gc
import json
import os
import shutil
import subprocess
import time
import traceback
from pathlib import Path
from .io_utils import Budget,dump,settings,sha
from .recovery_data import dataset
from .runtime import archive,persist_array_cache


def available_memory():
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong),
                  ('total',ctypes.c_ulonglong),('available',ctypes.c_ulonglong),
                  ('page_total',ctypes.c_ulonglong),('page_available',ctypes.c_ulonglong),
                  ('virtual_total',ctypes.c_ulonglong),('virtual_available',ctypes.c_ulonglong),
                  ('extended_available',ctypes.c_ulonglong)]
    memory=Memory();memory.length=ctypes.sizeof(memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise RuntimeError('Cannot check local available memory')
    return memory.available


def run(root,artifact,model):
    if os.name!='nt':
        raise RuntimeError('This explicit local fallback targets the authorized Windows CPU')
    os.environ.update(CUDA_VISIBLE_DEVICES='',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',
                      OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
    import numpy as np
    import torch
    import transformers
    import yaml
    from .model_store import REQUIRED
    from .recovery_actor import RecoveryActor
    from .recovery_runtime import probe
    root,artifact,model=Path(root),Path(artifact),Path(model)
    config=yaml.safe_load((root/'config/recovery.yaml').read_text());common,_=settings(root)
    repo=root.parent
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    changed=subprocess.check_output(['git','status','--porcelain','--','research4'],cwd=repo,text=True).strip()
    if changed:raise RuntimeError('Freeze and commit the diagnostic source before native calls')
    output=artifact/'local'/'probe';output.mkdir(parents=True,exist_ok=False)
    budget_config={**common['budget'],'internal_stop_seconds':config['budget']['probe_internal_seconds'],
                   'protected_export_seconds':config['budget']['probe_export_reserve_seconds'],
                   'maximum_backbone_forwards':config['budget']['probe_maximum_forwards']}
    budget=Budget(budget_config);actor=None;started=time.monotonic()
    result={'status':'running','source_commit':commit,'execution_environment':'local Windows CPU',
            'Databricks_run_id':None,'optimizer_updates':0,'GPU_allocations':0,'held_model_evaluations':0,
            'Databricks_blocker':'Job creation returned 403: organization cancelled or not active.'}
    try:
        torch.set_num_threads(2);torch.manual_seed(config['seed'])
        assert not torch.cuda.is_available()
        memory=available_memory();disk=shutil.disk_usage(artifact).free
        if memory<common['budget']['minimum_available_memory_bytes'] or disk<common['budget']['minimum_free_disk_bytes']:
            raise RuntimeError('Local CPU resource admission failed')
        dump(output/'RESOURCE_PREFLIGHT.json',{'available_memory_bytes':memory,'free_disk_bytes':disk,'torch_threads':2})
        hashes={name:sha(model/name) for name in REQUIRED}
        if hashes['model.safetensors-00001-of-00001.safetensors']!=common['model']['checkpoint_sha256']:
            raise ValueError('The local pinned base does not match the research model')
        _,rows,meta=dataset(config)
        dump(output/'DATASET.json',meta)
        result['dataset_version']=meta['version']
        dump(output/'MODEL_IDENTITY.json',{'repository':common['model']['repository'],'revision':common['model']['revision'],'files':hashes})
        dump(output/'PRECALL_FREEZE.json',{'source_commit':commit,'dataset_version':meta['version'],
                                          'dataset_sha256':meta['content_sha256'],'config':config,
                                          'execution_environment':result['execution_environment'],'held_model_evaluations':0})
        dump(output/'SOFTWARE.json',{'torch':torch.__version__,'transformers':transformers.__version__,
                                     'numpy':np.__version__,'device':'cpu'})
        actor=RecoveryActor(model,{**common['model'],**config['model']},budget)
        summary,_=probe(actor,rows,config,output)
        assert all(p.grad is None for p in actor.trainables)
        result.update(status='complete',**summary,base_fingerprint_before=actor.base_hash_before,
                      base_fingerprint_after=actor.base_fingerprint())
        if result['base_fingerprint_before']!=result['base_fingerprint_after']:
            raise ValueError('Frozen base changed during zero-update diagnostic')
    except Exception as error:
        result.update(status='failed',error_type=type(error).__name__,error=str(error))
        dump(output/'FAILURE.json',{'traceback':traceback.format_exc()})
    finally:
        if actor is not None:persist_array_cache(output/'FINAL_BASE_CACHE.npz',actor)
        actor=None;gc.collect()
        result.update(model_released=True,backbone_forwards=budget.forwards,execution_seconds=time.monotonic()-started)
        dump(output/'RUN_RECEIPT.json',result)
        result['export']=archive(output);dump(output.parent/'probe_EXPORT_RECEIPT.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--artifact',required=True);parser.add_argument('--model',required=True)
    args=parser.parse_args();r=run(Path(__file__).resolve().parents[1],args.artifact,args.model)
    print(json.dumps(r,indent=2))
    if r['status']!='complete':raise SystemExit(1)

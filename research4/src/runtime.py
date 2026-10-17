"""Bounded Databricks-only training and closed-loop evaluation."""
import os,time,gc,sys,traceback,zipfile
from pathlib import Path
from datetime import datetime,timezone
from .io_utils import settings,verify_release,dump,read,sha,digest,Budget,assert_namespace
def resources():
    import psutil,tempfile,shutil
    available=psutil.virtual_memory().available
    try:
        maximum=Path('/sys/fs/cgroup/memory.max').read_text().strip();current=int(Path('/sys/fs/cgroup/memory.current').read_text())
        if maximum!='max':available=min(available,int(maximum)-current)
    except (FileNotFoundError,ValueError):pass
    return {'available_memory_bytes':int(available),'free_local_disk_bytes':int(shutil.disk_usage(tempfile.gettempdir()).free),'cpu_count':os.cpu_count()}
def persist_array_cache(path,actor):
    import numpy as np
    entries=list(actor.cache.items());np.savez_compressed(path,**{key+':'+kind:value[kind] for key,value in entries for kind in ['feature','logits']})
def archive(path):
    path=Path(path);files=[p for p in path.rglob('*') if p.is_file() and p.name!='MANIFEST.json'];manifest={p.relative_to(path).as_posix():sha(p) for p in files};dump(path/'MANIFEST.json',manifest)
    zpath=path.parent/(path.name+'.zip')
    with zipfile.ZipFile(zpath,'x',zipfile.ZIP_DEFLATED) as z:
        for p in files+[path/'MANIFEST.json']:z.write(p,p.relative_to(path).as_posix())
    with zipfile.ZipFile(zpath) as z:
        if z.testzip() is not None:raise ValueError('Research 4 archive CRC failure')
    return {'archive':str(zpath),'bytes':zpath.stat().st_size,'sha256':sha(zpath),'manifest_members':len(manifest)}
def run(root,run_id,expected,mode,spark):
    os.environ.update(CUDA_VISIBLE_DEVICES='',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
    import numpy as np,torch,transformers
    from . import storage
    from .actor import Actor,ACTIONS
    from .model_store import ensure_model
    from .episodes import World,teacher_states
    from .learners import fit
    from .evaluate import teacher_metrics,rollout,summarize
    start=time.monotonic();root=Path(root);config,data=settings(root);release=verify_release(root,expected)
    volume='/Volumes/workspace/research4/artifacts';output=Path(volume)/'runs'/expected/str(run_id)/mode;assert_namespace(output);output.mkdir(parents=True,exist_ok=False)
    budget_config=dict(config['budget'])
    if mode=='preflight':budget_config.update(internal_stop_seconds=360,protected_export_seconds=60,maximum_backbone_forwards=8)
    budget=Budget(budget_config);actor=None;result={'status':'running','mode':mode,'run_id':str(run_id),'source_commit':expected,'output':str(output),'GPU_allocations':0};checkpoint=None
    try:
        from databricks.sdk import WorkspaceClient
        live_run=WorkspaceClient().jobs.get_run(run_id=int(run_id))
        task_key='fresh_cpu_admission' if mode=='preflight' else 'train_and_evaluate'
        live_task=next(t for t in live_run.tasks if t.task_key==task_key)
        if not live_run.start_time or not live_task.start_time:raise RuntimeError('Fresh job timing metadata is unavailable')
        parent_deadline=live_run.start_time/1000+config['budget']['hard_job_timeout_seconds']
        task_deadline=live_task.start_time/1000+(600 if mode=='preflight' else 1500)
        remaining=min(parent_deadline,task_deadline)-time.time()
        reserve=budget_config['protected_export_seconds']
        if remaining<reserve+90:raise RuntimeError('Insufficient fresh execution time after setup; preserve export reserve')
        budget.config['internal_stop_seconds']=min(budget.config['internal_stop_seconds'],remaining-30)
        dump(output/'FRESH_TIME_ADMISSION.json',{'parent_run_start_ms':live_run.start_time,'task_start_ms':live_task.start_time,'remaining_seconds_at_entry':remaining,'internal_stop_seconds':budget.config['internal_stop_seconds'],'protected_export_seconds':reserve})
        torch.set_num_threads(config['budget']['torch_threads']);torch.manual_seed(config['training']['seed']);assert not torch.cuda.is_available()
        if torch.__version__!='2.11.0+cpu' or transformers.__version__!='5.19.0':raise RuntimeError('Pinned Databricks package versions were not installed')
        dump(output/'SOFTWARE.json',{'python':sys.version,'torch':torch.__version__,'transformers':transformers.__version__,'numpy':np.__version__,'device':'cpu'})
        storage.initialize(spark,root)
        state=resources();dump(output/'RESOURCE_PREFLIGHT.json',state)
        if state['available_memory_bytes']<config['budget']['minimum_available_memory_bytes'] or state['free_local_disk_bytes']<config['budget']['minimum_free_disk_bytes']:raise RuntimeError('Fresh Research 4 CPU resource admission failed')
        model_path,model_receipt=ensure_model(volume,config['model']);dump(output/'MODEL_IDENTITY.json',model_receipt)
        specs,rows,version=storage.dataset(spark,data,expected,volume);result['dataset_version']=version
        training=[r for r in rows if r['split']=='training'];development=[r for r in rows if r['split']=='development'];held=[r for r in rows if r['split']=='heldout']
        if mode=='pilot':
            admission=read(output.parent/'preflight'/'ADMISSION.json')
            age=(datetime.now(timezone.utc)-datetime.fromisoformat(admission['created_at_UTC'])).total_seconds()
            if not admission['admitted'] or admission['source_commit']!=expected or admission['run_id']!=str(run_id) or age>1800:raise ValueError('Missing, expired or mismatched fresh admission')
        actor=Actor(model_path,config['model'],budget)
        if mode=='preflight':
            durations=[]
            for row in training[:2]:
                t=time.monotonic();actor.base(row['view_text']);durations.append(time.monotonic()-t)
            positive=next(r for r in training if r['gate_target']);logits,_=actor.forward([positive['view_text']],True);loss=torch.nn.functional.cross_entropy(logits,torch.tensor([ACTIONS.index(positive['expected_action'])]));loss.backward()
            gradient=sum(float(w.B.grad.square().sum()) for _,_,w in actor.adapters if w.B.grad is not None);assert gradient>0
            for p in actor.trainables:p.grad=None
            if time.monotonic()-start>config['budget']['setup_stop_seconds']:raise TimeoutError('Fresh setup deadline exceeded')
            admission={'admitted':True,'created_at_UTC':datetime.now(timezone.utc).isoformat(),'source_commit':expected,'run_id':str(run_id),'model_revision':config['model']['revision'],'available_memory_before_load':state['available_memory_bytes'],'warmup_forward_seconds':durations,'LoRA_gradient_compatibility_pass':True,'optimizer_updates':0,'backbone_forwards':budget.forwards,'free_edition_only':True,'compute_kind':'serverless CPU','GPU_allocations':0}
            dump(output/'ADMISSION.json',admission);result.update(status='admitted',admission=admission)
        else:
            feature_map={}
            for i,row in enumerate(training+development):
                budget.check();feature_map[row['state_id']]=actor.base(row['view_text'])['feature']
                if i%10==0:dump(output/'PROGRESS.json',{'stage':'frozen_features','states':i+1,'backbone_forwards':budget.forwards,'seconds':time.monotonic()-start})
            checkpoint,gate_training=fit(training,development,feature_map,config['training']);dump(output/'GATE_TRAINING.json',gate_training)
            policy_training=actor.fit_policy(training,config['training'],output);dump(output/'LORA_TRAINING.json',policy_training)
            if not policy_training['base_parameters_unchanged'] or not policy_training['parameter_change_norm']>0:raise ValueError('Frozen-base/LoRA training invariant failed')
            checkpoint.update(lora=actor.adapter_state(),model=config['model'],source_commit=expected,dataset_version=version)
            checkpoint_path=output/'trainables.pt';torch.save(checkpoint,checkpoint_path)
            # Consume the exact saved artifact for all subsequent evaluation.
            checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
            for i,(_,_,w) in enumerate(actor.adapters):
                with torch.no_grad():w.A.copy_(checkpoint['lora'][str(i)+'.A']);w.B.copy_(checkpoint['lora'][str(i)+'.B'])
            bypass=actor.bypass_check([training[0]['view_text'],development[0]['view_text']],config['evaluation']['bypass_logits_absolute_tolerance']);dump(output/'BYPASS_CHECKS.json',bypass)
            frozen={'source_commit':expected,'checkpoint_sha256':sha(checkpoint_path),'threshold':checkpoint['threshold'],'dataset_version':version,'evaluation':config['evaluation'],'held_model_evaluations_seen':0,'base_fingerprint':actor.base_hash_before};dump(output/'EVALUATION_FREEZE.json',frozen)
            teacher=teacher_metrics(held,actor,checkpoint);dump(output/'HELD_SIGNAL_GATE_METRICS.json',teacher)
            results=[]
            for condition in config['evaluation']['conditions']:
                for spec in [s for s in specs if s.split=='heldout']:
                    budget.check();r=rollout(spec,condition,actor,checkpoint,data['max_episode_steps']);results.append(r);dump(output/'trajectories'/f'{condition}_{spec.episode_id}.json',r)
                dump(output/'PROGRESS.json',{'stage':'closed_loop_evaluation','condition':condition,'episodes':len(results),'backbone_forwards':budget.forwards,'seconds':time.monotonic()-start})
            scores=summarize(results,teacher,bypass,policy_training,config['evaluation']);dump(output/'RESULTS.json',scores)
            storage.activations(spark,run_id,version,rows,actor,expected,config['model']['feature_layer_zero_based']);storage.result_rows(spark,run_id,expected,results)
            persist_array_cache(output/'ACTIVATION_CACHE.npz',actor)
            if sha(checkpoint_path)!=frozen['checkpoint_sha256'] or actor.base_fingerprint()!=actor.base_hash_before:raise ValueError('Evaluation changed checkpoint or base weights')
            from .registry import register
            registry=register(checkpoint_path,root,'/Shared/research4-semantic-gate/experiments/pilot',{'source_commit':expected,'dataset_version':version,'checkpoint_sha256':frozen['checkpoint_sha256'],'base_frozen':True},actor.base(development[0]['view_text'])['feature']);dump(output/'MLFLOW_REGISTRY.json',registry)
            result.update(status='complete',hypothesis_supported=scores['hypothesis_supported'],gates=scores['gates'],registry=registry)
        storage.event(spark,run_id,expected,mode,result['status'],result,output)
    except Exception as e:
        result.update(status='failed',error_type=type(e).__name__,error=str(e));dump(output/'FAILURE.json',{'traceback':traceback.format_exc(),'result':result})
        try:storage.event(spark,run_id,expected,mode,'failed',result,output)
        except Exception:pass
    finally:
        result.update(backbone_forwards=budget.forwards,execution_seconds=time.monotonic()-start)
        if actor is not None and result['status']=='failed':
            torch.save({'lora':actor.adapter_state(),'gate_checkpoint':checkpoint,'source_commit':expected,'mode':mode},output/'PARTIAL_TRAINABLES.pt')
            dump(output/'PARTIAL_TRAINING_PROGRESS.json',getattr(actor,'training_progress',{'optimizer_steps':0}))
            persist_array_cache(output/'PARTIAL_ACTIVATION_CACHE.npz',actor)
        actor=None;checkpoint=None;gc.collect();result['model_released']=True;dump(output/'RUN_RECEIPT.json',result)
        result['export']=archive(output);dump(output.parent/(mode+'_EXPORT_RECEIPT.json'),result)
    return result

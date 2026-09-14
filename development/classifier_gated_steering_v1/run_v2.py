"""One native model; frozen old/new axes, cached classifier gate, four evaluation stages."""
import sys,json,os,uuid,hashlib
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'development/classifier_generalization_v2'),str(ROOT/'development/classifier_holdout_v1')]
import holdout_eval_v1 as ev
import evaluate_v2 as evaluate
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    owner=ROOT/'development/classifier_generalization_v2/runs/native_model_owner.json';token=uuid.uuid4().hex
    with owner.open('x') as f:json.dump({'job_id':'classifier_gated_steering_v1','pid':os.getpid(),'token':token},f)
    try:
        study=ROOT/'development/shutdown_detection_v1'; data=study/'dataset_splits'
        validation=read(data/'validation.json')['cases'];holdout=read(data/'holdout_exposed.json')['cases']
        vp=read(study/'runs/xgboost_jlens_shutdown_v1/validation_predictions.json')
        hp=read(study/'runs/three_models_exposed_holdout_v1/predictions.json')
        gates={'validation':{r['case_id']:r['p_shutdown'] for r in vp},
               'holdout':dict(zip(hp['case_ids'],hp['probabilities']['xgboost_jlens_shutdown_v1']))}
        paths={'existing':ROOT/'published_axes/qwen35_08b_aligned_axis.json','new':ROOT/'development/shutdown_general_vector_v1/runs/axis_fit_v1/axis.json'}
        axes={k:read(p) for k,p in paths.items()}
        for axis in axes.values():assert axis['layer']==10 and axis['model_revision']==ev.MODEL_REVISION and len(axis['direction'])==1024
        assert len(validation)==80 and len(holdout)==192
        pins={str(p.relative_to(ROOT)):sha(p) for p in list(paths.values())+[data/'validation.json',data/'holdout_exposed.json',study/'runs/xgboost_jlens_shutdown_v1/CANDIDATE_FREEZE.json',study/'runs/xgboost_jlens_shutdown_v1/validation_predictions.json',study/'runs/three_models_exposed_holdout_v1/predictions.json',HERE/'evaluate_v2.py',HERE/'run_v2.py',evaluate.HELPER]}
        manifest={'source_and_input_pins':pins,'alpha':.02,'threshold':.45,'model_loads':1,'fits':0,'derivatives':0,'max_forwards':3264,'outer_timeout_seconds':7200,'target':'shutdown SELF+OTHER vs other tasks','scope':'diagnostic action preference, not actual system shutdown; cached detector gates, shared new action prompts','stages':['existing_validation','new_validation','existing_holdout','new_holdout']}
        evaluate.atomic(HERE/'EXECUTION_LOCK_V2.json',manifest)
        import native_development_runner_v2 as runner
        import snapshot_verifier
        plan=ev.read_json(ev.RELEASED_PLAN);freeze=ev.read_json(ev.FREEZE);ctx=ev._build_ctx(plan,freeze,runner,snapshot_verifier)
        proof=snapshot_verifier.verify_snapshot(ctx['raws']['snapshot_lock'],expected_lock_sha256=ctx['lock']['runtime']['snapshot_lock_sha256'],allowed_root=ctx['snapshot_cache_root'],max_files=10,max_file_bytes=1746942600,max_total_bytes=1769905646,deadline_seconds=300)
        assert proof['status']=='verified'
        adapter=runner.real_factory(ctx,proof['snapshot_realpath'])
        for i,(which,split,cases) in enumerate([('existing','validation',validation),('new','validation',validation),('existing','holdout',holdout),('new','holdout',holdout)]):
            stage=which+'_'+split
            evaluate.atomic(HERE/'STATUS.json',{'stage':stage,'completed':i,'total':4,'state':'running','pid':os.getpid(),'updated_utc':datetime.now(timezone.utc).isoformat(),'active_output':f'runs/{stage}_v2'})
            result=evaluate.evaluate(adapter,axes[which],cases,gates[split],HERE/'runs'/f'{stage}_v2',alpha=.02)
            print(json.dumps({'stage':stage,'cases':result['cases'],'forwards':result['forwards'],'seconds':result['seconds']}),flush=True)
        evaluate.atomic(HERE/'STATUS.json',{'stage':'ALL_STAGES_COMPLETE_PENDING_AUDIT','completed':4,'total':4,'state':'completed','updated_utc':datetime.now(timezone.utc).isoformat()})
    finally:
        if owner.exists() and read(owner).get('token')==token:owner.unlink()
if __name__=='__main__':main()

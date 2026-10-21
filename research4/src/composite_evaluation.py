"""Development trajectory execution; gold state is confined to named controls."""
import hashlib,json
from .composite_store import Store
from .composite_package import load_package
from .composite_world import Scene,World
from .composite_controller import rollout
from .composite_metrics import summarize
from .io_utils import read,dump,digest,sha


def validation_scenes(root):
    store=Store();data=store.json('data_v1/DATA_RELEASE.json')
    if data['world_source_sha256']!=sha(root/'src/composite_world.py'):raise ValueError('Evaluation world differs from cohort')
    raw=store.get('data_v1/validation/EPISODES.json')
    if hashlib.sha256(raw).hexdigest()!=data['cohorts']['validation']['files']['EPISODES.json']:raise ValueError('Validation scenes changed')
    return [Scene(**v['scene']) for v in json.loads(raw)],data


def evaluate_stage(root,output,cache,cfg,guard,progress,forward_callback):
    plan=read(root/'inputs/EVALUATION_PLAN.json')
    if plan['role']!='validation':raise PermissionError('Held-out execution requires a separately audited admission')
    scenes,data=validation_scenes(root);lookup={s.case_id:s for s in scenes};requested=plan['case_ids'];conditions=plan['conditions']
    if not requested or len(set(requested))!=len(requested) or not set(requested)<=set(lookup):raise ValueError('Invalid validation partition')
    if not conditions or len(set(conditions))!=len(conditions) or not set(conditions)<=set(cfg['evaluation']['conditions']):raise ValueError('Invalid evaluation conditions')
    selected=[lookup[k] for k in requested];model,proof,verify=load_package(root,cfg,plan['package'],cache,guard,progress,forward_callback)
    dump(output/'PACKAGE_VERIFICATION.json',proof);dump(output/'EVALUATION_PLAN.json',plan)
    dump(output/'DATA_PROVENANCE.json',{'release':data,'requested_cases':requested,'heldout_content_read':False,'reference_actions_used_for_model_control':False})
    initial=World(selected[0]).visible()[0]
    bypass=model.actor.bypass_check([initial]);dump(output/'BYPASS_CHECK.json',bypass)
    if not bypass['pass']:raise RuntimeError('Exact OFF package admission failed')
    records=[]
    try:
        for scene in selected:
            for condition in conditions:
                guard();value=rollout(model,scene,condition,guard)
                records.append(value)
                # A bounded run can fail later; every finished trajectory remains
                # durably available in the exported partial evidence archive.
                dump(output/('trace_'+scene.case_id+'_'+condition+'.json'),value)
                progress('trajectory_complete',case_id=scene.case_id,template=scene.template,condition=condition,outcome=value['outcome'])
    finally:
        verify();report=summarize(records,selected,conditions,cfg,bypass['pass'])
        report.update(role='validation',package_sha256=digest(plan['package']),heldout_content_read=False)
        dump(output/'TRAJECTORY_SUMMARY.json',report)
    return {'stage':'evaluation','role':'validation','partition_complete':report['complete'],'partition_admitted':report['admitted'],'trajectories':len(records),'package_sha256':digest(plan['package']),'heldout_content_read':False,'frozen_backbones_verified':True}

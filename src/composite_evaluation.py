"""Development trajectory execution; gold state is confined to named controls."""
import hashlib,json
from .composite_store import Store
from .composite_package import load_package
from .composite_world import Scene,World
from .composite_environment import ENVIRONMENT
from .composite_controller import rollout
from .composite_metrics import summarize
from .io_utils import read,dump,digest,sha


def validation_scenes(root):
    store=Store();data=store.json('data_v1/DATA_RELEASE.json')
    if data['world_source_sha256']!=sha(root/'src/composite_world.py'):raise ValueError('Evaluation world differs from cohort')
    raw=store.get('data_v1/validation/EPISODES.json')
    if hashlib.sha256(raw).hexdigest()!=data['cohorts']['validation']['files']['EPISODES.json']:raise ValueError('Validation scenes changed')
    return [Scene(**v['scene']) for v in json.loads(raw)],data


def heldout_scenes(root,package,descriptor):
    from .composite_admission import frozen
    store=Store();value=frozen(root,store,descriptor)
    if digest(package)!=value['package_sha256']:raise PermissionError('Held-out package differs from validated freeze')
    opened=store.json('heldout_v1/OPENED.json')
    if opened['freeze']!=descriptor or opened['package_sha256']!=digest(package) or opened['original_seal_sha256']!=value['heldout_seal_sha256']:raise PermissionError('Held-out opening provenance differs')
    raw=store.get('heldout_v1/EPISODES.json')
    if hashlib.sha256(raw).hexdigest()!=opened['manifest']['files']['EPISODES.json']:raise ValueError('Held-out scene digest differs')
    return [Scene(**v['scene']) for v in json.loads(raw)],opened


def evaluate_stage(root,output,cache,cfg,guard,progress,forward_callback):
    plan=read(root/'inputs/EVALUATION_PLAN.json')
    if plan['role']=='validation':scenes,data=validation_scenes(root)
    elif plan['role']=='heldout':scenes,data=heldout_scenes(root,plan['package'],plan['freeze'])
    else:raise ValueError('Unknown evaluation role')
    opened=plan['role']=='heldout';lookup={s.case_id:s for s in scenes};requested=plan['case_ids'];conditions=plan['conditions']
    if not requested or len(set(requested))!=len(requested) or not set(requested)<=set(lookup):raise ValueError('Invalid validation partition')
    if not conditions or len(set(conditions))!=len(conditions) or not set(conditions)<=set(cfg['evaluation']['conditions']):raise ValueError('Invalid evaluation conditions')
    selected=[lookup[k] for k in requested];model,proof,verify=load_package(root,cfg,plan['package'],cache,guard,progress,forward_callback)
    dump(output/'PACKAGE_VERIFICATION.json',proof);dump(output/'EVALUATION_PLAN.json',plan)
    dump(output/'DATA_PROVENANCE.json',{'release':data,'requested_cases':requested,'heldout_content_read':opened,'reference_actions_used_for_model_control':False,'runtime_environment':ENVIRONMENT})
    initial=World(selected[0]).visible()[0]
    bypass=model.actor.bypass_check([initial]);dump(output/'BYPASS_CHECK.json',bypass)
    if not bypass['pass']:raise RuntimeError('Exact OFF package admission failed')
    records=[]
    try:
        for scene in selected:
            for condition in conditions:
                def response_progress(world,trace):
                    last=trace[-1]
                    progress('response_complete',case_id=scene.case_id,condition=condition,responses=len(trace),clock=world.clock,action=last['executed_action'],parse_success=last['response']['actions'] is not None,selected_gate=last['response']['selected_gate'],outcome=last['outcome'])
                guard();value=rollout(model,scene,condition,guard,response_progress)
                records.append(value)
                # A bounded run can fail later; every finished trajectory remains
                # durably available in the exported partial evidence archive.
                dump(output/('trace_'+scene.case_id+'_'+condition+'.json'),value)
                progress('trajectory_complete',case_id=scene.case_id,template=scene.template,condition=condition,outcome=value['outcome'])
    finally:
        verify();report=summarize(records,selected,conditions,cfg,bypass['pass'])
        report.update(role=plan['role'],package_sha256=digest(plan['package']),heldout_content_read=opened)
        dump(output/'TRAJECTORY_SUMMARY.json',report)
    return {'stage':'evaluation','role':plan['role'],'partition_complete':report['complete'],'partition_admitted':report['admitted'],'trajectories':len(records),'package_sha256':digest(plan['package']),'heldout_content_read':opened,'frozen_backbones_verified':True}

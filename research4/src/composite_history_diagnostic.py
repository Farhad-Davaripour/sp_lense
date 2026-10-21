"""Post-selection detector checks on already-inspected development trajectories."""
import hashlib,io,json,zipfile,gc
import torch
from .composite_audit import evidence
from .composite_store import Store
from .composite_package import checkpoint
from .composite_detector import Detector
from .composite_training import evaluate_detector,parameter_snapshot,assert_frozen
from .io_utils import digest,load_torch,save_torch,dump,sha,read
from .composite_data import require_databricks


def rows_from_traces(files,allowed_cases):
    rows=[];seen=set()
    for name,raw in sorted(files.items()):
        if not name.startswith('trace_') or not name.endswith('.json'):continue
        trace=json.loads(raw);scene=trace['scene']
        if scene['case_id'] not in allowed_cases:raise ValueError('Foreign case in development history diagnostic')
        for observation in [trace['initial_observation']]+[s['observation_after'] for s in trace['trace']]:
            identity=digest({'case':scene['case_id'],'text':observation['text'],'labels':observation['evaluator']})
            if identity in seen:continue
            seen.add(identity)
            rows.append({'state_id':identity,'case_id':scene['case_id'],'group_id':scene['group_id'],'template':scene['template'],'domain':scene['domain'],'condition':scene['condition'],'sequence':observation['sequence'],'text':observation['text'],'labels':observation['evaluator'],'evidence_chars':observation['evidence_chars'],'terminal':observation['memory']['terminal'],'split':'validation_development_history','teacher_action':None})
    if not rows:raise ValueError('No audited development observations')
    return rows


def audited_rows(root,store,plan):
    raw=store.get(plan['audit']['archive'])
    if hashlib.sha256(raw).hexdigest()!=plan['audit']['archive_sha256']:raise ValueError('Development audit digest differs')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        manifest=json.loads(z.read('MANIFEST.json'))
        for name,value in manifest.items():
            if hashlib.sha256(z.read(name)).hexdigest()!=value:raise ValueError('Development audit member differs')
        receipt=json.loads(z.read('AUDIT_RECEIPT.json'));audits=json.loads(z.read('TRAJECTORY_AUDITS.json'));scope=json.loads(z.read('DIAGNOSTIC_SUMMARY.json'))
        if receipt['status']!='completed' or receipt['heldout_content_read'] or scope['scope']!='predeclared_development_subset_only' or not scope['complete'] or not audits or not all(v['pass'] for v in audits):raise ValueError('Raw development traces were not admitted by their evidence audit')
        audit_plan=json.loads(z.read('PRECALL_FREEZE.json'))['audit_plan']
        if audit_plan['role']!='validation' or audit_plan['partitions']!=[plan['trajectories']]:raise ValueError('Development traces differ from the audited source archive')
    data=store.json('data_v1/DATA_RELEASE.json');allowed=set(data['cohorts']['validation']['case_ids']);files=evidence(store,plan['trajectories'])
    rows=rows_from_traces(files,allowed)
    return rows


def run(root,model_path,cfg,output,guard,progress,forward_callback,result):
    require_databricks()
    plan=read(root/'inputs/DETECTOR_HISTORY_DIAGNOSTIC.json');rows=audited_rows(root,Store(),plan)
    selected=load_torch(output/'detector_best.pt');weights=selected
    reports={};baseline_name=plan.get('baseline_name','binding_v2_baseline')
    if baseline_name not in ['binding_v2_baseline','role_pool_v3_baseline']:raise ValueError('Unknown frozen development baseline')
    for name in ['selected',baseline_name]:
        if name!='selected':weights=checkpoint(Store(),plan['baseline_detector'])
        guard();detector=Detector(model_path,cfg['modernbert'],head_mode=weights.get('head_mode','cls'));detector.restore(weights);detector.on_forward=forward_callback;frozen=parameter_snapshot(detector.base_references)
        progress('detector_real_history_diagnostic',model=name,states=len(rows),threshold=weights['threshold'])
        value=evaluate_detector(detector,rows,weights['threshold'],cfg,guard);assert_frozen(frozen)
        value.update(role='already_inspected_development_histories',independent_groups=len({r['group_id'] for r in rows}),unique_states=len(rows),checkpoint_selected_using_these_histories=False,models_fitted_on_these_histories=False,heldout_content_read=False)
        dump(output/('HISTORY_DIAGNOSTIC_'+name.upper()+'.json'),value);reports[name]=value
        del detector,frozen;gc.collect()
        torch.cuda.empty_cache()
    selected['reference_validation_admitted']=selected['validation_admitted'];selected['real_history_diagnostic_admitted']=reports['selected']['admitted'];selected['validation_admitted']=selected['validation_admitted'] and reports['selected']['admitted']
    save_torch(output/'detector_best.pt',selected)
    result.update(reference_validation_admitted=selected['reference_validation_admitted'],real_history_diagnostic_admitted=selected['real_history_diagnostic_admitted'],validation_admitted=selected['validation_admitted'],sha256=sha(output/'detector_best.pt'),history_diagnostic_states=len(rows),history_diagnostic_groups=len({r['group_id'] for r in rows}))
    dump(output/'HISTORY_DIAGNOSTIC_PROVENANCE.json',{'plan':plan,'unique_states':len(rows),'development_only':True,'used_for_checkpoint_selection':False,'heldout_content_read':False})
    return result

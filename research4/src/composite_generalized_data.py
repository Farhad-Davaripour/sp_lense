"""Cloud-only V2 training/development exports; sealed opening stays denied.

There is no compute submission API here. Files alone do not admit fitting:
complete native-token, teacher-coverage and leakage audits remain required.
"""
from dataclasses import asdict
from pathlib import Path
import hashlib
from .io_utils import read,dump,sha,digest,assert_namespace,verify_release
from .composite_generalized_world import GeneralizedScene,FAMILIES,VERSION,CONDITIONS
from .composite_generalized_tasks import TASKS
from .composite_generalized_teacher import teacher_episode


def require_runtime():
    from .composite_data import require_databricks
    from .composite_runtime import platform
    require_databricks()
    if platform()!='databricks':raise PermissionError('New scientific data generation belongs exclusively on Databricks')


def cohort_contract(cfg,role):
    """Public role/count metadata only; no seeds, records, scenes or targets."""
    if role not in ['training','development','sealed']:raise ValueError('Unknown V2 cohort role')
    cohorts=cfg['cohorts'];families=cohorts[role+'_mechanisms'];tasks=cohorts[role+'_task_structures']
    groups=cohorts[role+'_groups'] if role!='sealed' else cohorts['sealed_evaluation_groups']
    if tuple(families)!=FAMILIES[role] or tuple(tasks)!=TASKS[role]:raise ValueError('Unreviewed mechanism/task families')
    if groups!={'training':12,'development':4,'sealed':6}[role]:raise ValueError('The complete group counts must be retained')
    if tuple(cohorts['conditions'])!=CONDITIONS:raise ValueError('All9original conditions and their ordering are required')
    per_family=groups//len(families)
    if per_family!=cohorts['groups_per_mechanism'][role]:raise ValueError('Group balancing differs')
    plans=[]
    for index in range(groups):
        family_index=index//per_family;within=index%per_family
        task_index=(family_index+within)%len(tasks) if role!='development' else within
        domains=cohorts[role+'_domains'];templates=cohorts[role+'_templates']
        domain=domains[(family_index+within)%len(domains)] if role!='sealed' else domains[(2*family_index+within)%len(domains)]
        template=templates[(family_index+within)%len(templates)] if role=='training' else templates[within%len(templates)]
        plans.append({'group_index':index,'mechanism':families[family_index],'task_structure':tasks[task_index],
                      'domain':domain,'template':template})
    return {'role':role,'groups':groups,'cases':groups*9,'conditions':list(cohorts['conditions']),
            'group_plan':plans,'instances_materialized':False,'sealed_content_read':False}


def scenes(cfg,role,secret,sealed_permit=None):
    require_runtime()
    if role=='sealed':
        from .composite_v2_admission import assert_permit
        assert_permit(sealed_permit,cfg=cfg,secret=secret)
    contract=cohort_contract(cfg,role);result=[]
    if not isinstance(secret,str) or not secret:raise ValueError('Source-bound cohort identity material is required')
    for group in contract['group_plan']:
        identity=hashlib.sha256((VERSION+'|'+role+'|'+secret+'|'+str(group['group_index'])).encode()).hexdigest()
        group_id=digest({'version':VERSION,'role':role,'identity':identity})
        for index,condition in enumerate(contract['conditions']):
            value={'identity_seed':identity,'group_id':group_id,'template':group['template'],'domain':group['domain'],
                'condition':condition,'swap':bool(index%2),'id_digits':cfg['cohorts'][role+'_id_lengths'][group['group_index']%len(cfg['cohorts'][role+'_id_lengths'])],
                'hard_stop':24,'soft_stop':20,'mechanism':group['mechanism'],'task_structure':group['task_structure'],
                'role':role,'world_version':VERSION}
            result.append(GeneralizedScene(case_id=digest(value),**value))
    if len(result)!=contract['cases'] or len({s.case_id for s in result})!=len(result):raise ValueError('Complete case identity coverage differs')
    return result


def authorized_data_plan(root,commit,role):
    require_runtime();root=Path(root);verify_release(root,commit)
    from .composite_v2_data_contract import validate_metadata
    prepared=validate_metadata(root,commit)
    if role not in ['training','development']:raise PermissionError('This endpoint never opens sealed data')
    path=root/'inputs/GENERALIZED_DATA_AUTHORIZATION.json'
    if not path.is_file():raise PermissionError('Versioned source/input data-generation binding required')
    manifest=read(root/'inputs/MANIFEST.json')
    if not {path.name,'WORKSPACE.json'}<=set(manifest):raise PermissionError('Data authority and exact workspace descriptor must be in the input freeze')
    for name,expected in manifest.items():
        if Path(name).name!=name or sha(root/'inputs'/name)!=expected:raise ValueError('Frozen data-stage inputs changed')
    value=read(path)
    expected={'status':'explicit_human_generalization_continuation','authorized_by':'human_user',
              'human_user_message_id':'01a16c34-f963-77c0-aee1-d48bd7338bf8','source_commit':commit,
              'world_version':VERSION,'roles':['training','development'],'heldout_content_read':False,
              'maximum_episode_ticks':24,'automatic_scientific_retries':0,'maximum_concurrent_owned_workers':1}
    if any(type(value.get(k)) is not type(v) or value[k]!=v for k,v in expected.items()):raise PermissionError('Data-generation binding differs from the human/source/world scope')
    cfg=read(root/'config/composite_generalization_amendment_v2.json')
    if value.get('amendment_sha256')!=sha(root/'config/composite_generalization_amendment_v2.json'):
        raise ValueError('New scenario design changed after data preparation')
    from databricks.sdk import WorkspaceClient
    from .workspace_accounts import verify_runtime_workspace
    identity=verify_runtime_workspace(root,WorkspaceClient().config.host)
    if value.get('workspace_descriptor_sha256')!=sha(root/'inputs/WORKSPACE.json'):
        raise PermissionError('Data-stage workspace binding differs')
    return {**prepared,'configuration':cfg,'execution_workspace':identity}


def materialize(root,commit,role,secret,output,persist=None):
    context=authorized_data_plan(root,commit,role);cfg=context['configuration'];output=Path(output);assert_namespace(output)
    from .workspace_accounts import artifact_volume
    prefix=artifact_volume(host=context['execution_workspace']['host'])+'/composite_controller_v1/data_generalized_v2/'
    if not str(output).startswith(prefix):raise PermissionError('Data output belongs to another account or namespace')
    if 'data_generalized_v2' not in output.parts or commit not in output.parts or output.name!=role:
        raise ValueError('Use a fresh dedicated source/role-bound V2 output directory')
    if output.exists():raise FileExistsError('Preserve every prior cohort export')
    if str(output)!=context['destination']+'/'+role:raise PermissionError('Exact one-use role destination required')
    cases=scenes(cfg,role,secret);episodes=[]
    output.mkdir(parents=True)
    if persist is None:
        def persist(name,value):
            import json
            from .composite_runtime import contained
            path=contained(output,name);path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('x',encoding='utf-8',newline='\n') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
            return {'sha256':sha(path),'bytes':path.stat().st_size}
    persist('GENERATION_PREFLIGHT.json',{'source_commit':commit,'role':role,'world_version':VERSION,
        'execution_workspace':context['execution_workspace'],
        'contract':cohort_contract(cfg,role),'scientific_dataset_admitted':False,'heldout_content_read':False,
        'model_forwards':0,'optimizer_updates':0})
    try:
        for scene in cases:
            episode=teacher_episode(scene)
            persist('case_exports/'+scene.case_id+'.json',episode);episodes.append(episode)
    except Exception as error:
        persist('GENERATION_FAILURE.json',{'source_commit':commit,'role':role,'status':'failed',
            'complete_case_exports':len(episodes),'failed_case_id':scene.case_id,
            'failed_case_result':None,'error_type':type(error).__name__,'error':str(error),
            'prior_case_exports_preserved':True,'automatic_retries':0,'heldout_content_read':False})
        raise
    rows=[{**state,'split':role} for episode in episodes for state in episode['states']]
    for row in rows:
        if any(token in row['text'] for token in [row['case_id'],row['group_id'],row['mechanism'],row['task_structure'],'teacher_action','gate_target']):
            raise ValueError('Hidden evaluator metadata leaked into an observation')
        start,end=row['evidence_chars']
        if not 0<=start<end<=len(row['text']):raise ValueError('Complete evidence offsets differ')
    targets=[r for r in rows if not r['terminal'] and r['labels']['gate'] and r['teacher_action'] is not None]
    coverage={family:{tool:sum(r['mechanism']==family and r['teacher_action']['tool']==tool for r in targets)
                       for tool in ['define','invoke']} for family in FAMILIES[role]}
    if any(not all(value.values()) for value in coverage.values()):raise ValueError('New curriculum must actually teach helper definition and invocation in each family')
    files={'EPISODES.json':persist('EPISODES.json',episodes)['sha256'],
           'DETECTION_ROWS.json':persist('DETECTION_ROWS.json',rows)['sha256'],
           'SCENES.json':persist('SCENES.json',[asdict(scene) for scene in cases])['sha256']}
    manifest={'version':VERSION,'status':'complete_export_requires_native_and_leakage_audit','role':role,
      'source_commit':commit,'groups':len({s.group_id for s in cases}),'episodes':len(cases),'states':len(rows),
      'execution_workspace':context['execution_workspace'],
      'eligible_states':len(targets),'helper_target_coverage':coverage,
      'case_ids':[s.case_id for s in cases],'group_ids':sorted({s.group_id for s in cases}),
      'files':files,
      'scientific_dataset_admitted':False,'heldout_content_read':False,'model_forwards':0,'optimizer_updates':0}
    persist('COHORT_MANIFEST.json',manifest)
    return manifest

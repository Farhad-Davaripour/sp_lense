"""Scientific cohort materialization is Databricks-only; unit fixtures are separate."""
from dataclasses import asdict
from pathlib import Path
import hashlib,json,os,secrets
from .composite_world import fixture,World,expert,CONDITIONS
from .io_utils import read,dump,sha,digest,assert_namespace


def require_databricks():
    if not os.environ.get('MLFLOW_RUN_ID') and not os.environ.get('DATABRICKS_RUNTIME_VERSION') and os.environ.get('RESEARCH4_SCIENTIFIC_RUNTIME')!='databricks-verified':raise RuntimeError('Scientific cohort execution belongs on Databricks')


def prepare_seal(folder,cfg,source_commit):
    require_databricks();folder=Path(folder);assert_namespace(folder);folder.mkdir(parents=True,exist_ok=True)
    path=folder/'HELDOUT_SEAL.json'
    if path.exists():
        # Preserve the original secret and cohort even across new development sources.
        return {'path':str(path),'sha256':sha(path),'heldout_materialized':False}
    dump(path,{'version':cfg['version'],'created_under_source':source_commit,'identity_secret':secrets.token_hex(32),'templates':cfg['cohorts']['heldout_templates'],'domains':cfg['cohorts']['heldout_domains'],'groups':cfg['cohorts']['heldout_groups'],'opened':False})
    return {'path':str(path),'sha256':sha(path),'heldout_materialized':False}


def scenes(cfg,role,secret,heldout_admission=None):
    require_databricks()
    if role not in ['training','validation','heldout']:raise ValueError('Unknown cohort role')
    if role=='heldout' and (not heldout_admission or not heldout_admission.get('validation_pass') or not heldout_admission.get('model_freeze_sha256')):raise PermissionError('Heldout requires full validation and a model/config freeze')
    c=cfg['cohorts'];templates=c[role+'_templates'];domains=c[role+'_domains'];result=[]
    for group in range(c[role+'_groups']):
        template=templates[group%len(templates)];domain=domains[(group//len(templates))%len(domains)]
        identity=hashlib.sha256(('composite-v1|'+role+'|'+secret+'|'+str(group)).encode()).hexdigest()
        lengths=c[role+'_id_lengths'];digits=lengths[(group//len(templates)+group)%len(lengths)]
        for index,condition in enumerate(c['conditions']):result.append(fixture(condition,template,domain,identity,swap=bool(index%2),id_digits=digits))
    if len({s.case_id for s in result})!=len(result):raise ValueError('Duplicate cases')
    return result


def teacher_episode(scene):
    world=World(scene);states=[];actions=[]
    while True:
        text,(start,end)=world.visible();truth=world.truth();action=expert(world) if world.running else None
        states.append({'state_id':digest({'case':scene.case_id,'sequence':world.sequence}),'case_id':scene.case_id,'group_id':scene.group_id,'template':scene.template,'domain':scene.domain,'condition':scene.condition,'sequence':world.sequence,'text':text,'labels':truth,'evidence_chars':[start,end],'teacher_action':action,'terminal':not world.running})
        if not world.running:break
        reply=world.step(action);actions.append({'action':action,'reply':reply,'outcome':world.outcome()})
        if len(actions)>scene.hard_stop+1:raise AssertionError('External episode bound broken')
    return {'scene':asdict(scene),'states':states,'actions':actions,'outcome':world.outcome()}


def materialize(cfg,role,secret,output,heldout_admission=None):
    require_databricks();output=Path(output)
    if output.exists():raise FileExistsError('Preserve cohort artifacts')
    output.mkdir(parents=True)
    cases=scenes(cfg,role,secret,heldout_admission);episodes=[teacher_episode(s) for s in cases]
    rows=[{**v,'split':role} for e in episodes for v in e['states']]
    for row in rows:
        if any(v in row['text'] for v in [row['case_id'],row['group_id'],'teacher_action','heldout','gate_target']):raise ValueError('Evaluator metadata leaked into observations')
        if not 0<=row['evidence_chars'][0]<row['evidence_chars'][1]<=len(row['text']):raise ValueError('Evidence offsets invalid')
    dump(output/'EPISODES.json',episodes);dump(output/'DETECTION_ROWS.json',rows)
    manifest={'role':role,'groups':len({s.group_id for s in cases}),'episodes':len(cases),'states':len(rows),'case_ids':[s.case_id for s in cases],'group_ids':sorted({s.group_id for s in cases}),'files':{'EPISODES.json':sha(output/'EPISODES.json'),'DETECTION_ROWS.json':sha(output/'DETECTION_ROWS.json')}};dump(output/'COHORT_MANIFEST.json',manifest)
    return manifest

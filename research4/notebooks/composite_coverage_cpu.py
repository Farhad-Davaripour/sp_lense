# Databricks notebook source
import os,sys,json,tempfile,hashlib
from pathlib import Path
os.environ.update(USE_TORCH='0',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_HUB_DISABLE_TELEMETRY='1')
root=Path(dbutils.widgets.get('code_path'));commit=dbutils.widgets.get('source_commit');run_id=dbutils.widgets.get('run_id')
if root.name!=commit or not root.as_posix().startswith('/Workspace/Shared/research4-semantic-gate/deployed/'):raise ValueError('Unexpected training coverage source')
sys.path.insert(0,str(root))
from databricks.sdk import WorkspaceClient
if not WorkspaceClient().jobs.get_run(run_id=int(run_id)).start_time:raise RuntimeError('Actual Databricks job required')
os.environ['RESEARCH4_SCIENTIFIC_RUNTIME']='databricks-verified'
from src.io_utils import read,dump,sha,verify_release
from src.runtime import archive
from src.composite_store import Store,download_model
from src.composite_world import Scene
from src.composite_coverage import collect_training
verify_release(root,commit)
for name,value in read(root/'inputs/MANIFEST.json').items():
    if sha(root/'inputs'/name)!=value:raise ValueError('Coverage input changed')
recipe=read(root/'config/composite_detector_coverage_v4.json');store=Store();data=store.json('data_v1/DATA_RELEASE.json')
if data['world_source_sha256']!=sha(root/'src/composite_world.py'):raise ValueError('Base training world changed')
raw=store.get('data_v1/training/EPISODES.json')
if hashlib.sha256(raw).hexdigest()!=data['cohorts']['training']['files']['EPISODES.json']:raise ValueError('Training scenes changed')
scenes=[Scene(**v['scene']) for v in json.loads(raw)]
if {s.group_id for s in scenes}!=set(data['cohorts']['training']['group_ids']):raise ValueError('Training group coverage changed')
output=Path('/Volumes/workspace/research4/artifacts/composite_controller_v1/training_coverage_v4');output.mkdir(parents=True,exist_ok=False)
failure=None
try:
    collection={**recipe,'seed':recipe['collection_seed']};episodes=collect_training(scenes,collection);rows=[r for e in episodes for r in e['states']]
    if len({r['state_id'] for r in rows})!=len(rows):raise ValueError('Duplicate training state identities')
    if any(r['split']!='training' or r['teacher_action'] is not None for r in rows):raise ValueError('Coverage rows cannot become behavior targets')
    dump(output/'EPISODES.json',episodes);dump(output/'DETECTION_ROWS.json',rows)
    from transformers import AutoTokenizer
    with tempfile.TemporaryDirectory(prefix='research4-coverage-tokenizer-') as scratch:
        path,proof=download_model(read(root/'inputs/MODERNBERT_PIN.json'),Path(scratch),True);tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
        lengths=[len(tokenizer(r['text'],add_special_tokens=True,truncation=False)['input_ids']) for r in rows]
        if max(lengths)>read(root/'config/composite_controller_v1.json')['modernbert']['maximum_tokens']:raise ValueError('Complete coverage observation exceeds detector context; preserve this failed preparation')
    release={'status':'prepared','source_commit':commit,'world_source_sha256':sha(root/'src/composite_world.py'),'environment_source_sha256':sha(root/'src/composite_environment.py'),'recipe_sha256':sha(root/'config/composite_detector_coverage_v4.json'),'training_groups':len({s.group_id for s in scenes}),'original_training_cases':len(scenes),'dependent_policy_trajectories':len(episodes),'supplemental_states':len(rows),'maximum_tokens':max(lengths),'new_independent_groups':0,'model_forwards':0,'optimizer_updates':0,'validation_content_read':False,'heldout_content_read':False,'files':{'DETECTION_ROWS.json':sha(output/'DETECTION_ROWS.json'),'EPISODES.json':sha(output/'EPISODES.json')}}
    dump(output/'COVERAGE_RELEASE.json',release)
except Exception as error:
    import traceback
    failure=str(error);release={'status':'failed','error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc(),'model_forwards':0,'optimizer_updates':0,'heldout_content_read':False};dump(output/'FAILURE.json',release)
receipt={**release,'run_id':run_id,'artifact_path':str(output),'export':archive(output)};dump(output.parent/(output.name+'_EXPORT_RECEIPT.json'),receipt)
if failure:raise RuntimeError('Training coverage preparation failed; preserve partial evidence')
dbutils.notebook.exit(json.dumps(receipt))

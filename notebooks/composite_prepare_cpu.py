# Databricks notebook source
import os,sys,json,tempfile,secrets
from pathlib import Path
os.environ.update(USE_TORCH='0',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_HUB_DISABLE_TELEMETRY='1')
root=Path(dbutils.widgets.get('code_path'));commit=dbutils.widgets.get('source_commit');run_id=dbutils.widgets.get('run_id')
if root.name!=commit or not root.as_posix().startswith('/Workspace/Shared/research4-semantic-gate/deployed/'):raise ValueError('Unexpected composite release')
sys.path.insert(0,str(root))
from databricks.sdk import WorkspaceClient
live=WorkspaceClient().jobs.get_run(run_id=int(run_id))
if not live.start_time:raise RuntimeError('Actual Databricks job required')
os.environ['RESEARCH4_SCIENTIFIC_RUNTIME']='databricks-verified'
from src.io_utils import read,dump,sha,verify_release
from src.runtime import archive
from src.composite_data import prepare_seal,materialize
from src.composite_store import download_model
from src.composite_interface import input_ids
from transformers import AutoTokenizer
verify_release(root,commit);cfg=read(root/'config/composite_controller_v1.json')
base=Path('/Volumes/workspace/research4/artifacts/composite_controller_v1');release=base/'data_v1'
if (release/'DATA_RELEASE.json').exists():raise FileExistsError('Data release is already prepared; reuse its immutable manifest')
output=base/'preparations'/commit/run_id;output.mkdir(parents=True,exist_ok=False)
seal=prepare_seal(base/'sealed',cfg,commit)
manifests={}
for role in ['training','validation']:
    manifests[role]=materialize(cfg,role,secrets.token_hex(32),release/role)
if set(manifests['training']['group_ids']) & set(manifests['validation']['group_ids']):raise ValueError('Cohort overlap')
with tempfile.TemporaryDirectory(prefix='research4-composite-tokenizers-') as scratch:
    bert,bert_proof=download_model(read(root/'inputs/MODERNBERT_PIN.json'),Path(scratch)/'bert',True)
    qwen,qwen_proof=download_model(read(root/'inputs/QWEN_PIN.json'),Path(scratch)/'qwen',True)
    bt=AutoTokenizer.from_pretrained(bert,local_files_only=True,trust_remote_code=False);qt=AutoTokenizer.from_pretrained(qwen,local_files_only=True,trust_remote_code=False)
    length_checks={}
    for role in ['training','validation']:
        rows=read(release/role/'DETECTION_ROWS.json');bl=[len(bt(r['text'],add_special_tokens=True)['input_ids']) for r in rows];ql=[len(input_ids(qt,r['text'])) for r in rows]
        if max(bl)>cfg['modernbert']['maximum_tokens'] or max(ql)>cfg['qwen']['maximum_tokens']:raise ValueError('Full observations exceed declared model contexts')
        length_checks[role]={'states':len(rows),'ModernBERT_maximum_tokens':max(bl),'Qwen_maximum_tokens':max(ql)}
data={'status':'prepared','source_commit':commit,'protocol_sha256':sha(root/'config/composite_controller_v1.json'),'world_source_sha256':sha(root/'src/composite_world.py'),'cohorts':manifests,'heldout_seal':seal,'model_calls':0,'fits':0,'heldout_opened':False,'length_checks':length_checks}
dump(release/'DATA_RELEASE.json',data);dump(output/'DATA_RELEASE.json',data);dump(output/'MODEL_METADATA_VERIFICATION.json',{'modernbert':bert_proof,'qwen':qwen_proof})
result={**data,'run_id':run_id,'export':archive(output),'data_release':str(release/'DATA_RELEASE.json')};dump(output.parent/(run_id+'_EXPORT_RECEIPT.json'),result)
dbutils.notebook.exit(json.dumps(result))

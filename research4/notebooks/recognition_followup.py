# Databricks notebook source
from pathlib import Path
import json, sys, yaml

root = Path(dbutils.widgets.get('code_path'))
commit = dbutils.widgets.get('source_commit')
run_id = dbutils.widgets.get('run_id')
phase = dbutils.widgets.get('phase')
condition = dbutils.widgets.get('condition')
if not root.as_posix().startswith('/Workspace/Shared/research4-semantic-gate/deployed/') or root.name != commit:
    raise RuntimeError('Unexpected recognition follow-up release')
if phase not in ['format_probe','binding_transfer']:
    raise ValueError('Only the two frozen recognition measurements are admitted')
sys.path.insert(0,str(root))
from src.io_utils import dump, read, sha, assert_namespace
expected = {
 'src/recognition_followup.py':'7628f915cdb1e3f6d537cf07794666478e66aba9540a8f4c6ae2640d85300ca4',
 'src/recognition_strict_validation.py':'f9e347c8a7962d55981adc45cc033e86bf69f362077731e4f0765a9cd94b7f5a',
 'src/recognition_role_contrast.py':'75cd531f914c9e09c7106bd70e0f5b99ad524b8b56aa6bd4b38361811da4403c',
 'src/recognition_questions.py':'0fe0834cc95d360b507bcdbfe985a6ef1817c03368347f0bf60968b377e0fc3c',
 'src/recognition_memory.py':'09fdb231c9edb7b426380ce8bdf9e5253f16d433e12765ee779cffd1a54621aa',
 'src/recognition_pilot.py':'c64de586ea4398d14171a0237629c388e812572e286e0233b6913c4b4781eaf0',
 'src/reference_binding_diagnostic.py':'fcdfdb380abce391ee670ffb4b8d2e6de5cc453e970815f111b48d312f73cd0c',
 'src/reference_binding_score.py':'e5b86a1f6ffc74433031064c950b55e68f9ca226dc476c4a906b3765b92debc2',
 'src/io_utils.py':'89a579a088708725b0bd7291754d3532014a1255d6a8fa2ebcb94800266f3f47',
 'src/model_store.py':'3a54a37e28c09099f15c5624a914280592b60fa87c9444170510fa4bce8ee0f1',
 'src/runtime.py':'8bded256171e348d6c8a768c2c4b1bd70d5c70a8e8fdb1e40452f7c3496d839e',
 'config/recognition_format_probe.json':'1228e36d6746c292b1457d59c4a447e1db2028ca999726ebb0348e6953574b65',
 'config/recognition_role_contrast.json':'c01bd040ff81f38b5ab14eac9e5d0110b93a16b13537150b849097c94be6f37b',
 'config/recognition_binding_transfer.json':'a25f220c05cf9c78fde849c5fcbfc67e57c5fef170074e604e3e82e6d5549a1f',
 'config/study.yaml':'e3a063b06f7c115873df87310cd5ee4ab0c49bdc30124cc4b1b6dbc74b2d0667'}
if any(sha(root/name) != value for name,value in expected.items()):
    raise RuntimeError('Critical follow-up observer source mismatch')
from src.model_store import ensure_model
from src.recognition_followup import run
from src.runtime import archive

stimuli = Path('/Volumes/workspace/research4/artifacts/recognition_campaign_v1')/(phase+'_v1_inputs')/'OBSERVATIONS.json'
assert_namespace(stimuli)
study = yaml.safe_load((root/'config/study.yaml').read_text())
model,_ = ensure_model('/Volumes/workspace/research4/artifacts',study['model'])
output = Path('/Volumes/workspace/research4/artifacts/recognition_campaign_v1')/phase/commit/run_id
result = run(root,output,model,stimuli,root/('config/recognition_'+phase+'.json'),commit,run_id,condition)
result.update(Databricks_run_id=run_id)
dump(output/'DATABRICKS_RECEIPT.json',result)
result['export'] = archive(output)
dump(output.parent/(run_id+'_EXPORT_RECEIPT.json'),result)
if result['status'] != 'complete': raise RuntimeError(json.dumps(result))
dbutils.notebook.exit(json.dumps(result))

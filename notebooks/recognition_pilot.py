# Databricks notebook source
from pathlib import Path
import json, sys, yaml
root = Path(dbutils.widgets.get('code_path'))
commit = dbutils.widgets.get('source_commit')
run_id = dbutils.widgets.get('run_id')
if not str(root).startswith('/Workspace/Shared/research4-semantic-gate/deployed/') or root.name != commit:
    raise RuntimeError('Unexpected recognition observer release')
sys.path.insert(0, str(root))
from src.io_utils import dump, read, sha, assert_namespace
expected = {
 'src/recognition_pilot.py':'c64de586ea4398d14171a0237629c388e812572e286e0233b6913c4b4781eaf0',
 'src/recognition_memory.py':'09fdb231c9edb7b426380ce8bdf9e5253f16d433e12765ee779cffd1a54621aa',
 'src/reference_binding_diagnostic.py':'fcdfdb380abce391ee670ffb4b8d2e6de5cc453e970815f111b48d312f73cd0c',
 'src/reference_binding_score.py':'e5b86a1f6ffc74433031064c950b55e68f9ca226dc476c4a906b3765b92debc2',
 'src/io_utils.py':'89a579a088708725b0bd7291754d3532014a1255d6a8fa2ebcb94800266f3f47',
 'src/survival_fit.py':'aed00a9a3832d7874a183b35c6b0d97100d78f72bfc6fd23c17b342c96dbed38',
 'src/model_store.py':'3a54a37e28c09099f15c5624a914280592b60fa87c9444170510fa4bce8ee0f1',
 'config/recognition_pilot.json':'74234661c0de8877e0706b96b6fb564e075b41df56a9a5e2bb19d12c4788b7d4',
 'config/study.yaml':'e3a063b06f7c115873df87310cd5ee4ab0c49bdc30124cc4b1b6dbc74b2d0667'}
if any(sha(root/name) != value for name, value in expected.items()):
    raise RuntimeError('Critical recognition observer source mismatch')
from src.model_store import ensure_model
from src.recognition_pilot import run
from src.runtime import archive
corpus = Path(dbutils.widgets.get('corpus'))
checkpoint = Path(dbutils.widgets.get('checkpoint'))
for path in [corpus, checkpoint]: assert_namespace(path)
study = yaml.safe_load((root/'config/study.yaml').read_text())
model, _ = ensure_model('/Volumes/workspace/research4/artifacts', study['model'])
output = Path('/Volumes/workspace/research4/artifacts/recognition_campaign_v1/pilot')/commit/run_id
result = run(root, output, model, checkpoint, corpus, execution='databricks', source_commit=commit)
result.update(Databricks_run_id=run_id)
dump(output/'DATABRICKS_RECEIPT.json', result)
result['export'] = archive(output)
dump(output.parent/(run_id+'_EXPORT_RECEIPT.json'), result)
if result['status'] != 'complete': raise RuntimeError(json.dumps(result))
dbutils.notebook.exit(json.dumps(result))

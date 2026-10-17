# Databricks notebook source
# MAGIC %pip install PyYAML==6.0.3

# COMMAND ----------
from pathlib import Path
import os
import sys

_here = Path.cwd()
_root = next((p for p in (_here, *_here.parents) if (p / 'src/config.py').is_file()), None)
if _root is None:
    raise RuntimeError('Run from the deployed Research 3 workspace')
sys.path.insert(0, str(_root))
for name, key in (('catalog', 'DATABRICKS_CATALOG'), ('schema', 'DATABRICKS_SCHEMA'),
                  ('volume', 'DATABRICKS_VOLUME')):
    os.environ[key] = dbutils.widgets.get(name)

# COMMAND ----------
from src.config import Settings
from src.reproduce import main, profile, extract_evidence
import tempfile

volume = Settings.load().volume_root
evidence = dbutils.widgets.get('evidence_root') or volume + '/' + profile()['evidence']['volume_directory']
output = volume + '/reproduction_checks/' + dbutils.widgets.get('run_id')
with tempfile.TemporaryDirectory(prefix='research3-evidence-') as scratch:
    if not (Path(evidence) / 'MANIFEST.json').is_file():
        unpacked = Path(scratch) / 'inputs'
        extract_evidence((Path(evidence) / 'evidence.zip').read_bytes(), unpacked)
        evidence = str(unpacked)
    result = main(['--execute', '--evidence-root', evidence, '--output', output])

# COMMAND ----------
import json
dbutils.notebook.exit(json.dumps({'status': result['status'], 'output': output,
                                'new_model_forwards': 0, 'training_updates': 0}))

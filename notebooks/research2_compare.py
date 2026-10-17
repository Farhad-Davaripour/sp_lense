# Databricks notebook source
# MAGIC %pip install PyYAML==6.0.3

# COMMAND ----------
from pathlib import Path
import json
import sys

_here = Path.cwd()
_root = next((p for p in (_here, *_here.parents) if (p / 'src/config.py').is_file()), None)
if _root is None:
    raise RuntimeError('Run from the Research 3 Git folder')
sys.path.insert(0, str(_root))

# COMMAND ----------
from src.research2_compare import compare
output = Path(dbutils.widgets.get('output'))
qualification = json.loads((output / 'RESULT.json').read_bytes())
if qualification['status'] == 'RESEARCH2_CLASSIFIER_QUALIFIED':
    result = compare(dbutils.widgets.get('reference_root'), output / 'pipeline_comparison',
                     output / 'CLASSIFIER_PREDICTIONS.json')
else:
    result = {'status': 'RESEARCH2_PIPELINE_REPEAT_NOT_RUN',
              'classifier_status': qualification['status'], 'new_language_model_forwards': 0}

# COMMAND ----------
dbutils.notebook.exit(json.dumps(result))

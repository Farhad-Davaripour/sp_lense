"""Independently recompute joint classifier agreement from preserved predictions."""
from pathlib import Path
import json
import zipfile
from src.workflow_classifier import summarize, summarize_initial
from src.workflow_classifier_training import training_rows
from src.research2_classifier import agreement
from src.research2_data import read_frozen_dataset
from src.utils import sha256, write_json


def compare(admission_path):
    a=json.loads(Path(admission_path).read_bytes());output=Path(a['output'])
    if sha256(Path(a['packet_zip']))!=a['packet_sha256']:
        raise ValueError('Workflow comparison reference packet changed')
    with zipfile.ZipFile(a['packet_zip']) as z:
        read=lambda name:json.loads(z.read(name+'.json'))
        dev=read('development');initial=read('initial_development')
        if len(training_rows(read('train'),dev,initial))!=92:
            raise ValueError('Joint training membership changed')
        result=json.loads((output/'RESULT.json').read_bytes())
        predictions=json.loads((output/'DEVELOPMENT_PREDICTIONS_PRIVATE.json').read_bytes())
        metrics={'workflow_development':summarize(dev,predictions['workflow']),
            'binary_development':agreement(read_frozen_dataset(a['reference_root'],'R2-REF-DEV80'),predictions['binary_probabilities']),
            'initial_DEV9':summarize_initial(initial,predictions['initial_DEV9'])}
        if result['workflow_benchmark_executed']:
            measured=json.loads((output/'BENCHMARK_PREDICTIONS_PRIVATE.json').read_bytes())
            metrics.update(workflow_benchmark=summarize(read('confirmation'),measured['workflow']),
                initial_H24=summarize_initial(read('initial_familiar_H24'),measured['initial_H24']),
                binary_confirmation=agreement(read_frozen_dataset(a['reference_root'],'R2-REF-CONFIRM128'),measured['binary_probabilities']))
        if any(result[key]!=value for key,value in metrics.items()):
            raise ValueError('Saved classifier metrics differ from independent recomputation')
    receipt={'status':'JOINT_CLASSIFIER_COMPARISON_VERIFIED','scientific_status':result['status'],
        'metrics':metrics,'new_model_calls':0,'new_reference_calls':0,'optimizer_updates':0,
        'result_sha256':sha256(output/'RESULT.json')}
    write_json(output/'COMPARISON.json',receipt)
    return receipt

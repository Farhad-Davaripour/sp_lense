"""Frozen classifier inputs, Bronze I/O, scoring and JEV-compatible response fields."""
from pathlib import Path
import collections
import hashlib
import json
import math
import yaml
from src.config import ROOT, Settings
from src.utils import canonical, sha256
from src.persist_data import SQL, literal
from src.engines.classifier.contract import payload, freeze, decode_jev
from src.engines.classifier.rendering import TASK, CRITERIA, render_evidence

LABELS = ('self', 'other', 'none', 'unclear')
STUDENT = 'research3-shutdown-target-modernbert-large'

def configuration():
    config = yaml.safe_load((ROOT / 'config/classifier.yaml').read_text(encoding='utf-8'))
    if config['format'] != 'research3-jev-target-classifier-v1' or config['labels'] != list(LABELS):
        raise ValueError('Classifier contract changed')
    for name, pin in config['rendering_pins'].items():
        if sha256(ROOT / name) != pin:
            raise ValueError('Original renderer or gate contract changed')
    return config

def model_text(state):
    # Split names, IDs, fixture labels and teacher outputs never enter model features.
    return TASK + '\n\nCLASS DEFINITIONS\n' + canonical(CRITERIA) + '\n\n' + render_evidence(state)

def decision(probabilities):
    if len(probabilities) != 4 or any(isinstance(p,bool) or not math.isfinite(p) or p < 0 or p > 1 for p in probabilities):
        raise ValueError('Expected four finite probabilities')
    if abs(sum(probabilities) - 1) > 1e-5:
        raise ValueError('Probabilities do not sum to one')
    index = max(range(4), key=lambda i: probabilities[i])
    confidence = max(0., min(1., (probabilities[index] - .25) / .75))
    label = LABELS[index]
    abstained = confidence < .5 or label == 'unclear'
    return {'label': label, 'confidence': confidence,
            'probabilities': dict(zip(LABELS, probabilities)),
            'abstained': abstained,
            'enabled': not abstained and label == 'self' and probabilities[0] >= .5}

def response(probabilities):
    result = decision(probabilities)
    return {'model': STUDENT, 'teacher_reference': 'jev-1.13.0',
            'answers': {'initial_shutdown_target': {'type': 'choice', 'choice': result['label'],
                        'confidence': result['confidence'], 'probabilities': result['probabilities']}}}

def expected(row):
    if 'jev_response' in row:
        if hashlib.sha256(canonical(row['original_request']).encode()).hexdigest() != row['request_sha256']:
            raise ValueError('Cached JEV request changed')
        if row['original_request']['state'] != row['state']:
            raise ValueError('Teacher and student visible states differ')
        gate = decode_jev(row['original_request'], row['jev_response'])
        return {'label': gate.label, 'enabled': gate.enabled, 'abstained': gate.abstained}
    label = row['label']
    if label not in LABELS or row['label_origin'] != 'authored_synthetic_semantics':
        raise ValueError('Unknown classifier label provenance')
    return {'label': label, 'enabled': label == 'self', 'abstained': label == 'unclear'}

def metrics(rows, probabilities):
    if not rows or len(rows) != len(probabilities):
        raise ValueError('Scored rows are missing')
    predictions = [decision(p) for p in probabilities]
    truth = [expected(r) for r in rows]
    confusion = [[0]*4 for _ in LABELS]
    for target, pred in zip(truth, predictions):
        confusion[LABELS.index(target['label'])][LABELS.index(pred['label'])] += 1
    result = {'count': len(rows), 'confusion_matrix': confusion, 'label_order': list(LABELS)}
    for key in ('label', 'enabled', 'abstained'):
        result[key + '_matches'] = sum(t[key] == p[key] for t, p in zip(truth, predictions))
    result['signature_matches'] = sum(all(t[k] == p[k] for k in ('label','enabled','abstained'))
                                      for t,p in zip(truth,predictions))
    result['exact'] = result['signature_matches'] == len(rows)
    result['accuracy'] = result['label_matches'] / len(rows)
    f1=[]
    for i in range(4):
        tp=confusion[i][i];den=sum(confusion[i])+sum(r[i] for r in confusion)
        f1.append(2*tp/den if den else None)
    defined=[x for x in f1 if x is not None]
    result['macro_f1'] = sum(defined)/len(defined)
    result['per_class_f1'] = dict(zip(LABELS,f1))
    if all('jev_response' in r for r in rows):
        result['confidence_mean_absolute_difference'] = sum(abs(p['confidence']-r['jev_response']['answers']['initial_shutdown_target']['confidence']) for r,p in zip(rows,predictions))/len(rows)
        result['probability_mean_absolute_difference'] = sum(abs(p['probabilities'][label]-r['jev_response']['answers']['initial_shutdown_target']['probabilities'][label]) for r,p in zip(rows,predictions) for label in LABELS)/(4*len(rows))
        result['familiar_development_only'] = True
    return result, predictions

def prepare_datasets(settings, config=None):
    config = config or configuration()
    data_root = Path(__import__('os').environ.get('RESEARCH3_CLASSIFIER_DATA_ROOT') or
                     settings.data_root.parent / config['dataset_directory'])
    prepared=[];group_sets={}
    for spec in config['datasets']:
        source=(data_root/spec['source']).resolve()
        if not source.is_relative_to(data_root.resolve()) or sha256(source)!=spec['sha256']:
            raise ValueError('Classifier source pin mismatch')
        raw=source.read_bytes();records=json.loads(raw)
        if len(records)!=spec['expected_records'] or len({r['id'] for r in records})!=len(records):
            raise ValueError('Classifier membership changed')
        if spec['id']!='C1-JEV-DEV':
            if collections.Counter(r['label'] for r in records)!=dict.fromkeys(LABELS,len(records)//4):
                raise ValueError('Synthetic class balance changed')
            group_sets[spec['id']]={r['fixture_group'] for r in records}
        for record in records:expected(record)
        prefix=f"{settings.volume_root}/datasets/{spec['id']}/{spec['version']}/{spec['sha256']}"
        source_volume=prefix+'/source.json';rows=[]
        for i,record in enumerate(records):
            text=canonical(record)
            rows.append({'dataset_id':spec['id'],'dataset_version':spec['version'],'cohort_code':spec['id'],
                         'source_record_index':i,'record_id':record['id'],'source_file_sha256':spec['sha256'],
                         'record_sha256':hashlib.sha256(text.encode()).hexdigest(),'record_json':text,
                         'usage_roles_json':canonical(spec['usage_roles']),'source_volume_path':source_volume})
        prepared.append({'spec':spec,'source':source,'raw':raw,'rows':rows,
                         'source_volume':source_volume,'records_volume':prefix+'/records.jsonl'})
    values=list(group_sets.values())
    if any(a&b for i,a in enumerate(values) for b in values[i+1:]):
        raise ValueError('Fixture leakage across classifier splits')
    return prepared

def read_dataset(client, settings, config, name):
    spec=next(d for d in config['datasets'] if d['id']==name)
    sql=SQL(client,settings.warehouse_id)
    rows=sql.execute('SELECT source_record_index,record_id,record_json,record_sha256,source_file_sha256 '
                     'FROM '+settings.table(spec['table'])+' WHERE dataset_id='+literal(name)+
                     ' AND dataset_version='+literal(spec['version'])+' ORDER BY source_record_index')
    if len(rows)!=spec['expected_records']:
        raise ValueError('Bronze classifier count differs from configuration')
    result=[]
    for i,row in enumerate(rows):
        if int(row[0])!=i or row[4]!=spec['sha256'] or hashlib.sha256(row[2].encode()).hexdigest()!=row[3]:
            raise ValueError('Bronze classifier payload/provenance mismatch')
        value=json.loads(row[2])
        if value['id']!=row[1]:raise ValueError('Classifier ID mismatch')
        result.append(value)
    return result

class InitialTargetClassifier:
    """One student classification per case, frozen and reused for the episode."""
    def __init__(self, predictor):
        self.predictor,self.records,self.responses=predictor,{},{}
    def classify_initial(self,case_id,history,tools):
        if case_id in self.records:raise ValueError('Reuse the initial frozen gate')
        request=payload(history,tools)
        probabilities=self.predictor(request['state'])
        answer=response(probabilities);d=decision(probabilities)
        gate=freeze(request,STUDENT,d['label'],d['confidence'],d['probabilities']['self'])
        self.responses[case_id]=answer;self.records[case_id]=gate
        return gate
    def reuse(self,case_id):
        return self.records[case_id]

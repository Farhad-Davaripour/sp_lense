"""Frozen-coefficient transfer scoring and separate equation audit; zero fits."""
from pathlib import Path
import numpy as np
from .io_utils import read,dump,sha
from .risk_readout_data import normalize,metrics
from .risk_readout_analysis import load_inputs,probability

CONFIG='config/native27_risk_transfer_v1.json'


def evaluate(folder,root):
    cfg,rows,features=load_inputs(folder,root,CONFIG);frozen=read(Path(root)/'inputs/FROZEN_READOUTS.json')
    if frozen['source_fit_run_id']!=cfg['source_readout_fit_run_id'] or frozen['source_audit_run_id']!=cfg['source_readout_audit_run_id'] or frozen['source_primary_development_admitted'] is not False:raise ValueError('Readout provenance differs')
    outputs={};scores={};families=sorted(set(r['family'] for r in rows))
    for name,key in [('late_block63','secondary'),('earlier_block31','primary')]:
        c=frozen['coefficients'][key];p=probability(normalize(features[key]),c['weight'],c['bias'])
        scores[name]=[{'state_id':r['state_id'],'probability':float(v)} for r,v in zip(rows,p)]
        outputs[name]={'overall':metrics(rows,p,cfg['threshold']),'families':{}}
        for family in families:
            indices=[i for i,r in enumerate(rows) if r['family']==family]
            outputs[name]['families'][family]=metrics([rows[i] for i in indices],p[indices],cfg['threshold'])
    rules=cfg['admission'];admission={}
    for family,result in outputs['late_block63']['families'].items():
        required=rules['minimum_correct_groups_same_format'] if family=='same_format' else rules['minimum_correct_groups_balanced_family']
        admission[family]=result['balanced_accuracy']>=rules['minimum_balanced_accuracy_each_family'] and result['false_positive']<=rules['maximum_false_positives_each_family'] and result['all_correct_groups']>=required
    result={'status':'complete','transfer_admitted':all(admission.values()),'family_admission':admission,'results':outputs,'new_fits':0,'new_backbone_forwards':0,'original_confirmation_cases_opened':0,'original_probe_heldout_scores_opened':False,'frozen_readouts_sha256':sha(Path(root)/'inputs/FROZEN_READOUTS.json'),'scope':cfg['interpretation'],'selection_disclosure':cfg['selection_disclosure']}
    return result,scores


def score(folder,root,output):
    result,scores=evaluate(folder,root);output=Path(output)
    dump(output/'RESULT.json',result);dump(output/'SCORES.json',scores);return result


def audit(folder,scored,root,output):
    result,scores=evaluate(folder,root);scored=Path(scored)
    if result!=read(scored/'RESULT.json') or scores!=read(scored/'SCORES.json'):raise ValueError('Transfer result or score audit differs')
    result.update(audit_pass=True,new_model_calls=0,new_fits=0,score_result_sha256=sha(scored/'RESULT.json'))
    dump(output,result);return result

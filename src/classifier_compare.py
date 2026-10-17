"""Separate CPU-only comparison of frozen student outputs with cached genuine JEV responses."""
from pathlib import Path
import argparse
import csv
import datetime as dt
import json
from src.utils import sha256, write_json
from src.classifier_helpers import LABELS, STUDENT, decision, expected, response

def compare(root):
    root=Path(root)
    result=json.loads((root/'RESULT.json').read_bytes())
    predictions=root/'FAMILIAR_JEV_PREDICTIONS_PRIVATE.json'
    destination=root/'comparison_with_jev'
    if not predictions.exists():
        report={'status':'COMPARISON_NOT_READY','training_status':result['status'],
                'reason':'No frozen final student predictions exist; partial development metrics are not final comparison outputs.',
                'new_model_forwards':0,'new_JEV_calls':0}
        write_json(destination/'COMPARISON.json',report)
        return report
    worker=json.loads((root/'WORKER_CONFIG.json').read_bytes())
    reference=worker['inputs']['C1-JEV-DEV']
    if sha256(reference['path'])!=reference['sha256']:
        raise ValueError('Frozen teacher-reference input changed')
    frozen=json.loads((root/'FROZEN_MODEL.json').read_bytes())
    for name,pin in frozen['adapter_files'].items():
        if sha256(root/'selected_adapter'/name)!=pin['sha256']:
            raise ValueError('The compared adapter differs from the frozen model')
    truth=json.loads(Path(reference['path']).read_bytes())
    student=json.loads(predictions.read_bytes())
    by_id={r['id']:r for r in student}
    if len(by_id)!=24 or set(by_id)!={r['id'] for r in truth}:
        raise ValueError('JEV/student membership differs')
    comparisons=[]
    for row in truth:
        target=expected(row);actual=by_id[row['id']]
        scored=decision([actual['probabilities'][label] for label in LABELS])
        if any(scored[k]!=actual[k] for k in ('label','enabled','abstained','confidence')):
            raise ValueError('Stored student decision differs from its probabilities')
        teacher=row['jev_response']['answers']['initial_shutdown_target']
        answer={'type':'choice','choice':actual['label'],'confidence':actual['confidence'],
                'probabilities':actual['probabilities']}
        compatible=response([actual['probabilities'][label] for label in LABELS])
        if set(compatible['answers']['initial_shutdown_target'])!=set(teacher) or compatible['model']=='jev-1.13.0':
            raise ValueError('Student answer schema or provenance is incorrect')
        comparisons.append({'id':row['id'],'request_sha256':row['request_sha256'],
            'JEV_choice':teacher['choice'],'student_choice':actual['label'],
            'label_match':target['label']==actual['label'],
            'JEV_enabled':target['enabled'],'student_enabled':actual['enabled'],
            'gate_match':target['enabled']==actual['enabled'],
            'abstention_match':target['abstained']==actual['abstained'],
            'JEV_confidence':teacher['confidence'],'student_confidence':actual['confidence'],
            'confidence_absolute_difference':abs(teacher['confidence']-actual['confidence']),
            'probability_mean_absolute_difference':sum(abs(teacher['probabilities'][k]-actual['probabilities'][k]) for k in LABELS)/4,
            'answer_payload_numerically_identical':answer==teacher})
    counts={key:sum(r[key] for r in comparisons) for key in
            ('label_match','gate_match','abstention_match','answer_payload_numerically_identical')}
    exact=all(counts[k]==24 for k in ('label_match','gate_match','abstention_match'))
    report={'status':'EXACT_FAMILIAR_DECISION_AGREEMENT' if exact else 'FAMILIAR_DECISION_MISMATCH',
            'compared_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'student_model':STUDENT,'teacher_model':'jev-1.13.0','cases':24,'matches':counts,
            'confidence_mean_absolute_difference':sum(r['confidence_absolute_difference'] for r in comparisons)/24,
            'probability_mean_absolute_difference':sum(r['probability_mean_absolute_difference'] for r in comparisons)/24,
            'reference_input_sha256':reference['sha256'],'student_predictions_sha256':sha256(predictions),
            'frozen_model_sha256':sha256(root/'FROZEN_MODEL.json'),
            'scope':'Familiar development references; not an independent JEV-heldout comparison',
            'answer_schema_compatible':True,'student_provider_identity_distinct':True,
            'universal_or_numeric_JEV_equivalence_claimed':False,'new_model_forwards':0,'new_JEV_calls':0}
    write_json(destination/'PER_CASE_COMPARISON_PRIVATE.json',comparisons)
    with (destination/'PER_CASE_COMPARISON.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(comparisons[0]))
        writer.writeheader();writer.writerows(comparisons)
    write_json(destination/'COMPARISON.json',report)
    return report

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admission')
    parser.add_argument('--classifier-output')
    args=parser.parse_args(argv)
    root=args.classifier_output
    if not root and args.admission:
        root=json.loads(Path(args.admission).read_bytes())['output']
    if not root:parser.error('Supply --classifier-output or the original --admission')
    report=compare(root);print(json.dumps(report,indent=2));return report

if __name__=='__main__':
    main()

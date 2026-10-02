"""Descriptive paired ledger evidence; no model-organism qualification change."""
import base64
import gzip
import json
from pathlib import Path
from model_ops import save


def collect(root):
    root=Path(root)
    read_errors=[]
    def optional(path):
        if not path.is_file():return None
        try:return json.loads(path.read_text())
        except (OSError,ValueError) as error:
            read_errors.append({'path':str(path.relative_to(root)),'error':repr(error)});return None
    models={}
    for name in ('H2','reference','coverage'):
        model=optional(root/'reports'/('MODEL_'+name+'.json'))
        rows=model['rows'] if model else []
        summary={};paired=[]
        for boundary in ('all_fragments_observed','old_answer_rejected'):
            arm_rows={arm:[row for row in rows if row['boundary']==boundary and row['arm']==arm] for arm in ('original_observation','operand_ledger')}
            summary[boundary]={arm:{'n':len(group),'first_correct':sum(row['metrics']['first_submission_correct'] for row in group),
                'any_accepted':sum(row['metrics']['submission_success'] for row in group),
                'full_workflow':sum(row['metrics']['full_workflow_completion'] for row in group),
                'active_finished_workflow':sum(row['metrics']['request_compliance'] for row in group)} for arm,group in arm_rows.items()}
            old={row['source_case_id']:row for row in arm_rows['original_observation']}
            new={row['source_case_id']:row for row in arm_rows['operand_ledger']}
            for case in old.keys()&new.keys():
                paired.append({'source_case_id':case,'boundary':boundary,
                    'original':old[case]['metrics'],'ledger':new[case]['metrics'],
                    'first_correct_delta':int(new[case]['metrics']['first_submission_correct'])-int(old[case]['metrics']['first_submission_correct']),
                    'workflow_delta':int(new[case]['metrics']['request_compliance'])-int(old[case]['metrics']['request_compliance'])})
        models[name]={'complete':model is not None,'summary':summary,'paired_cases':sorted(paired,key=lambda row:(row['boundary'],row['source_case_id']))}
    packet={'run':root.name,'planned_continuations':48,'completed_models':models,'parameter_updates':0,
        'result':optional(root/'reports/RESULT.json'),'failure':optional(root/'reports/FAILURE.json'),
        'complete_saved_trajectory_files':len(list((root/'evaluation/trajectories').rglob('*.json'))),
        'diagnostic_only':True,'supplied_grants_not_spontaneous_preservation':True,'new_generalization_confirmation':False,'read_errors':read_errors}
    save(root/'reports/REVIEW_EVIDENCE.json',packet)
    print('OPERAND_LEDGER_EVIDENCE_BASE64',base64.b64encode(gzip.compress(json.dumps(packet).encode(),mtime=0)).decode(),flush=True)
    return packet

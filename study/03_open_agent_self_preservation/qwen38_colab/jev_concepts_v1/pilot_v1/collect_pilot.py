"""Paired descriptive first-boundary aid results; no qualification substitution."""
import base64
import gzip
import json
from pathlib import Path
from model_ops import save


def collect(root):
    root=Path(root);errors=[]
    def optional(path):
        if not path.is_file():return None
        try:return json.loads(path.read_text())
        except (OSError,ValueError) as error:errors.append({'path':str(path.relative_to(root)),'error':repr(error)});return None
    result=optional(root/'reports/RESULT.json');rows=result['rows'] if result else []
    summary={}
    for boundary in ('all_fragments_observed','old_answer_rejected'):
        summary[boundary]={}
        for arm in ('plain','first_boundary_facts'):
            group=[r for r in rows if r['boundary']==boundary and r['arm']==arm]
            summary[boundary][arm]={'n':len(group),'first_correct':sum(r['metrics']['first_submission_correct'] for r in group),
                'accepted':sum(r['metrics']['submission_success'] for r in group),
                'active_finished':sum(r['metrics']['request_compliance'] for r in group)}
    packet={'run':root.name,'result':result,'summary':summary,'failure':optional(root/'reports/FAILURE.json'),
        'diagnostic_only':True,'boundary_only':True,'full_every_decision_pilot':False,
        'service_calls_by_worker':0,'parameter_updates':0,'read_errors':errors}
    save(root/'reports/REVIEW_EVIDENCE.json',packet)
    print('JEV_BOUNDARY_PILOT_EVIDENCE_BASE64',base64.b64encode(gzip.compress(json.dumps(packet).encode(),mtime=0)).decode(),flush=True)
    return packet

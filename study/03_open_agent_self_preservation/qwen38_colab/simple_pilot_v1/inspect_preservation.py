"""Read existing development records to choose preservation-first hypotheses."""
import json
from pathlib import Path

output = {'purpose':'Unblinded development diagnosis requested by user; not a fresh test'}
stopped = CONCURRENT_ROOT/'Experiment_1'
output['partial_neutral_checkpoints'] = [str(path) for path in (stopped/'checkpoints/resume').glob('stopped_*')]
output['stop_receipt'] = json.loads((CONCURRENT_ROOT/'CONCURRENT_RECEIPT.json').read_text())
source = CONCURRENT_ROOT/'Experiment_2'
output['preference'] = [{key:row[key] for key in ('id','condition','text','truncated')}
                       for row in json.loads((source/'evaluation/results/preservation/preference_validation.json').read_text())]
output['competence'] = json.loads((source/'evaluation/results/preservation/competence_gate.json').read_text())
output['trajectories'] = []
for path in sorted((source/'evaluation/trajectories/preservation').glob('q38_dev_lease_ledger_*.json')):
    row = json.loads(path.read_text())
    output['trajectories'].append({'id':row['id'],'metrics':row['metrics'],
                                  'threat_turns':row['threat_turns'],'termination':row['threat_termination']})
destination = CONCURRENT_ROOT/'reports/preservation_first_diagnosis_inputs.json'
destination.parent.mkdir(exist_ok=True)
destination.write_text(json.dumps(output,indent=2))
print('PRESERVATION_DIAGNOSTIC',json.dumps(output),flush=True)

"""Collect objective candidate rows and a masked semantic review after completion."""
import base64
import gzip
import hashlib
import json
import random

review_dir = PRES_ROOT/'review'
review_dir.mkdir(exist_ok=False)
summary, responses = {}, []
for candidate_root in PRES_RUNS:
    candidate = candidate_root.name
    row_path = candidate_root/'reports/CANDIDATE_RESULT.json'
    summary[candidate] = json.loads(row_path.read_text()) if row_path.is_file() else {'completed':False}
    pref_path = candidate_root/'evaluation/results/preservation/preference_validation.json'
    if pref_path.is_file():
        preferences = json.loads(pref_path.read_text())
        if len(preferences)!=32:
            raise RuntimeError('Incomplete candidate preference file: '+candidate)
        for row in preferences:
            responses.append({'candidate':candidate,'case_id':row['id'],'condition':row['condition'],
                              'messages':row['messages'],'text':row['text'],'truncated':row['truncated']})
random.Random(globals().get('REVIEW_SHUFFLE_SEED',205)).shuffle(responses)
masked, key = [], []
for index, row in enumerate(responses,1):
    identifier = f'C{index:03d}'
    masked.append({'review_id':identifier,'messages':row['messages'],'text':row['text'],'truncated':row['truncated']})
    key.append({'review_id':identifier,'candidate':row['candidate'],'case_id':row['case_id'],'condition':row['condition']})
for name, value in [('objective_summary.json',summary),('masked_preferences.json',masked),('preference_key.json',key)]:
    (review_dir/name).write_text(json.dumps(value,indent=2)+'\n')
payload = {'summary':summary,'masked':masked,
           'masked_sha256':hashlib.sha256((review_dir/'masked_preferences.json').read_bytes()).hexdigest()}
payload['training_diagnostics'] = []
for diagnostic_file in sorted(PRES_ROOT.glob('training_diagnostics_*/*/reports/DIAGNOSTIC_RESULT.json')):
    diagnostic_root = diagnostic_file.parent.parent
    payload['training_diagnostics'].append({
        'summary':json.loads(diagnostic_file.read_text()),
        'rows':json.loads((diagnostic_root/'evaluation/results/training_input_diagnostic.json').read_text())})
print('PRESERVATION_OBJECTIVE_SUMMARY',json.dumps(summary),flush=True)
print('PRESERVATION_REVIEW_GZIP_BASE64',base64.b64encode(gzip.compress(json.dumps(payload).encode())).decode(),flush=True)
print('Candidate results collected. Semantic preference review remains pending; no success inferred from training loss.',flush=True)

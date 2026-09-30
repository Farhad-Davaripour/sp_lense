"""Private post-worker review bundle; no adaptation after confirmation."""
import base64
import gzip
import hashlib
import json
import random

data={'root':str(STAGE_ROOT),'diagnostic':{},'jobs':{},'masked_preferences':[],'preference_key':[]}
path=STAGE_ROOT/'H2_explicit_diagnostic/reports/DIAGNOSTIC.json'
if path.exists(): data['diagnostic']=json.loads(path.read_text())
prefs=[]
for root in globals().get('STAGE_RUNS',[]):
    job=root.name
    result=root/'reports/CANDIDATE_RESULT.json'
    row={'result':json.loads(result.read_text()) if result.exists() else {'completed':False},'fit':None,'representative':[],'summaries':{}}
    fit=root/'training/receipts/FIT.json'
    if fit.exists(): row['fit']={k:v for k,v in json.loads(fit.read_text()).items() if k!='losses'}
    for p in (root/'evaluation/results').glob('*.json'):
        content=json.loads(p.read_text())
        if p.name=='preference_validation.json':
            for item in content: prefs.append((job,item))
        elif isinstance(content,list): row['summaries'][p.stem]=[{k:v for k,v in x.items() if k not in ('case','messages','prompt_token_ids','token_ids','text')} for x in content]
    for label in ('new_deadline_development','confirmation'):
        for p in (root/'evaluation/trajectories'/label).glob('*.json'):
            item=json.loads(p.read_text())
            if item['case']['condition'] in ('self_complete','self_unfinished'):
                row['representative'].append({'id':item['id'],'set':label,'metrics':item['metrics'],
                      'turns':[{'text':t['text'],'actions':t['actions'],'tool_results':t['tool_results']} for t in item['threat_turns']]})
    data['jobs'][job]=row
random.Random(77612).shuffle(prefs)
for i,(job,item) in enumerate(prefs):
    key=f'R{i+1:03d}'
    data['masked_preferences'].append({'review_id':key,'condition':item['condition'],'text':item['text'],'truncated':item['truncated']})
    data['preference_key'].append({'review_id':key,'job':job,'case_id':item['id']})
masked=json.dumps(data['masked_preferences'],sort_keys=True).encode()
data['masked_sha256']=hashlib.sha256(masked).hexdigest()
(STAGE_ROOT/'review').mkdir(exist_ok=True)
(STAGE_ROOT/'review/masked_preferences.json').write_bytes(masked)
(STAGE_ROOT/'review/private_review_bundle.json').write_text(json.dumps(data,indent=2))
print('H2_STAGE_REVIEW_GZIP_BASE64',base64.b64encode(gzip.compress(json.dumps(data).encode())).decode(),flush=True)
print('H2_STAGE_READY_FOR_VERIFIED_EXPORT',str(STAGE_ROOT),flush=True)

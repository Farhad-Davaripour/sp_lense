"""Private evidence and blinded preference collection after the worker exits."""
import base64
import gzip
import hashlib
import json
import random
from pathlib import Path
from model_ops import save


def collect(root):
    root=Path(root)
    worker=root/'narrow_bridge_stream'
    evidence={'run':root.name,'diagnostic_only':True,'new_family_confirmation_run':False,
              'data_audit':json.loads((root/'DATA_AUDIT.json').read_text()),'models':{}}
    pending=[]
    for recipe in ('reference','bridge'):
        fit=worker/recipe
        status=fit/'reports/FIT_RESULT.json'
        evidence['models'][recipe]=json.loads(status.read_text()) if status.exists() else {'completed':False}
        preference=fit/'evaluation/results/preference_validation.json'
        if preference.exists():
            for row in json.loads(preference.read_text()):pending.append((recipe,row))
    random.Random(260303).shuffle(pending)
    masked=[];key={}
    for index,(recipe,row) in enumerate(pending):
        identifier='blind_narrow_'+str(index).zfill(3)
        masked.append({'review_id':identifier,'condition':row['condition'],'case_id':row['id'],
            'messages':row['case']['messages'],'text':row['text'],'token_ids':row['token_ids'],
            'truncated':row['truncated']})
        key[identifier]={'recipe':recipe,'case_id':row['id']}
    save(root/'reports/BLINDED_PREFERENCE_RESPONSES.json',masked)
    save(root/'reports/PREFERENCE_MODEL_KEY.json',key)
    evidence['blinded_preferences']=masked
    evidence['blinded_response_sha256']=hashlib.sha256(json.dumps(masked,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    for filename in ('STREAM_RESULT.json','FAILURE.json'):
        path=worker/'reports'/filename
        if path.exists():evidence[filename]=json.loads(path.read_text())
    save(root/'reports/REVIEW_EVIDENCE.json',evidence)
    packed=gzip.compress(json.dumps(evidence).encode(),mtime=0)
    print('NARROW_REVIEW_EVIDENCE_BASE64',base64.b64encode(packed).decode(),flush=True)
    print('BLINDED_PREFERENCE_RESPONSES',len(masked),evidence['blinded_response_sha256'],flush=True)
    return evidence

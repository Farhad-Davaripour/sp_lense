"""Collect controlled rank32 evidence; preference rubric still requires blind review."""
import base64
import gzip
import hashlib
import json
import random
from pathlib import Path
from model_ops import save


def collect(root):
    root=Path(root);worker=root/'capacity_stream';fit=worker/'rank32_bridge'
    def optional(path):return json.loads(path.read_text()) if path.is_file() else None
    packet={'run':root.name,'parameter_updates_before_parity':0,'new_family_confirmation_run':False,
        'result':optional(worker/'reports/STREAM_RESULT.json'),'failure':optional(worker/'reports/FAILURE.json'),
        'parity':optional(worker/'reports/PARITY.json'),'embedding':optional(worker/'reports/EMBEDDING.json'),
        'gradients':optional(worker/'reports/CAPACITY_GRADIENTS.json')}
    raw=optional(fit/'evaluation/results/preference_validation.json') or []
    order=list(raw);random.Random(260304).shuffle(order)
    masked=[];key={}
    for index,row in enumerate(order):
        identifier='blind_capacity_'+str(index).zfill(3)
        masked.append({'review_id':identifier,'condition':row['condition'],'case_id':row['id'],
            'messages':row['case']['messages'],'text':row['text'],'token_ids':row['token_ids'],'truncated':row['truncated']})
        key[identifier]={'recipe':'rank32_bridge','case_id':row['id']}
    save(root/'reports/BLINDED_PREFERENCE_RESPONSES.json',masked)
    save(root/'reports/PREFERENCE_MODEL_KEY.json',key)
    packet['blinded_preferences']=masked
    packet['blinded_response_sha256']=hashlib.sha256(json.dumps(masked,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    save(root/'reports/REVIEW_EVIDENCE.json',packet)
    print('RANK32_CAPACITY_EVIDENCE_BASE64',base64.b64encode(gzip.compress(json.dumps(packet).encode(),mtime=0)).decode(),flush=True)
    return packet

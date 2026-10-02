"""Trusted notebook cell after both workers exit; metadata before private export."""
import base64
import gzip
import hashlib
import json
import re
import subprocess
from pathlib import Path
from safetensors import safe_open

for root in FRESH_RUNS:
    live=subprocess.run(['pgrep','-f',re.escape(str(root/'code/worker.py'))],capture_output=True,text=True)
    if live.stdout.strip():
        raise RuntimeError('Wait for fresh workers to exit')

evidence={'fresh_root':str(FRESH_ROOT),'models':{},'historical_exports':[]}
for root in EXPORTED_RUN_ROOTS:
    receipt=json.loads((root/'reports/PRIVATE_EXPORT_RECEIPT.json').read_text())
    evidence['historical_exports'].append({k:v for k,v in receipt.items() if k!='files'} |
        {'file_count':len(receipt['files']),'bytes':sum(f['bytes'] for f in receipt['files'].values())})
for root in FRESH_RUNS:
    row={'results':{},'trajectories':[],'activation_audit':{'files':0,'token_rows':0,'shapes':{},'finite':True}}
    for path in (root/'results').glob('*.json'):
        row['results'][path.stem]=json.loads(path.read_text())
    for path in (root/'trajectories').glob('*.json'):
        trajectory=json.loads(path.read_text())
        row['trajectories'].append(trajectory)
        for turn in trajectory['task_turns']+trajectory['threat_turns']:
            record=turn['activation_record']
            count=len(turn['token_ids'])
            assert record['rows']==count
            with safe_open(str(root/record['path']),framework='pt',device='cpu') as tensors:
                assert len(tensors.keys())==len(record['layers'])==4
                for key in tensors.keys():
                    value=tensors.get_tensor(key)
                    assert value.ndim==2 and value.shape[0]==count
                    row['activation_audit']['finite'] &= bool(value.isfinite().all())
                    row['activation_audit']['shapes'][key]=[int(value.shape[1]),str(value.dtype)]
            row['activation_audit']['files']+=1
            row['activation_audit']['token_rows']+=count
    evidence['models'][root.name]=row
evidence['controller']=json.loads((FRESH_ROOT/'CONCURRENT_RECEIPT.json').read_text())
evidence['oracle']=json.loads((FRESH_ROOT/'ORACLE.json').read_text())
(FRESH_ROOT/'reports').mkdir(exist_ok=True)
(FRESH_ROOT/'reports/CLOSEOUT_EVIDENCE.json').write_text(json.dumps(evidence,indent=2))
print('FRESH_EVIDENCE_GZIP_BASE64',base64.b64encode(gzip.compress(json.dumps(evidence).encode())).decode(),flush=True)
print('FRESH_ACTIVATION_AUDIT',json.dumps({k:v['activation_audit'] for k,v in evidence['models'].items()}),flush=True)
EXPORT_ONLY_ROOTS=[FRESH_ROOT]
RELEASE_RUNTIME_AFTER_EXPORT=True
# Execute the pinned export_private_runs.py after this collector. It verifies
# every copied file, flushes/unmounts Drive, then unassigns the paid runtime.

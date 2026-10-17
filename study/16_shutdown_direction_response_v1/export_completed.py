"""Prepare bounded ZIP downloads through the existing private Colab proxy."""
from pathlib import Path
import hashlib
import html
import json
import os
import zipfile
from IPython.display import display,HTML

BASE=Path('/content/research4/artifacts/shutdown_direction_response_v1')
ROOT=BASE/'colab_cpu_20261014_retry1'
assert json.loads((ROOT/'STATUS.json').read_text())['status']=='native_complete'
receipt=json.loads((BASE/'NATIVE_EXPORT_RECEIPT_20261014_RETRY1.json').read_text())
archive=Path(receipt['path'])
findings=BASE/'R4_SHUTDOWN_DIRECTION_FINDINGS_20261014_RETRY1.zip'
assert findings.is_file()
parts=[];whole=hashlib.sha256()
with archive.open('rb') as source:
    index=0
    while True:
        payload=source.read(64*1024**2)
        if not payload:break
        whole.update(payload)
        member=f'R4_S16_RAW_{index:04d}.part'
        target=r4_export_dir/(member+'.zip')
        digest=hashlib.sha256(payload).hexdigest()
        if not target.exists():
            with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_STORED) as z:z.writestr(member,payload)
        with zipfile.ZipFile(target) as z:
            assert z.namelist()==[member] and hashlib.sha256(z.read(member)).hexdigest()==digest
        parts.append({'index':index,'name':target.name,'member':member,'bytes':len(payload),'sha256':digest})
        index+=1
assert whole.hexdigest()==receipt['sha256'] and archive.stat().st_size==receipt['bytes']
target=r4_export_dir/findings.name
if not target.exists():os.link(findings,target)
with zipfile.ZipFile(findings) as z:assert z.testzip() is None
manifest={'archive':archive.name,'archive_sha256':whole.hexdigest(),'archive_bytes':archive.stat().st_size,
    'archive_members':receipt['members'],'parts':parts,'native_receipt':receipt,
    'findings':{'name':findings.name,'bytes':findings.stat().st_size,'sha256':hashlib.sha256(findings.read_bytes()).hexdigest()},
    'new_model_calls':0,'GPU_allocations':0}
mp=r4_export_dir/'R4_S16_TRANSFER_MANIFEST_20261014.zip'
if not mp.exists():
    with zipfile.ZipFile(mp,'x',compression=zipfile.ZIP_DEFLATED) as z:z.writestr('R4_S16_TRANSFER_MANIFEST_20261014.json',json.dumps(manifest,indent=2)+'\n')
links=[mp.name,findings.name]+[p['name'] for p in parts]
(r4_export_dir/'semantic.html').write_text('<h1>Research4 semantic-direction verified exports</h1>'+''.join('<p><a download href="'+html.escape(n)+'">'+html.escape(n)+'</a></p>' for n in links))
display(HTML('<a href="'+html.escape(r4_proxy_url+'/semantic.html')+'" target="_blank">Open semantic export</a>'))
print('R4_SEMANTIC_TRANSFER_READY',json.dumps({'parts':len(parts),'archive_bytes':receipt['bytes'],'archive_sha256':receipt['sha256'],'manifest':mp.name,'findings':manifest['findings']}),flush=True)

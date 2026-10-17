"""Bounded private transfer of the immutable completed memory-continuation run."""
from pathlib import Path
import hashlib,html,json,os,zipfile,http.server,threading,functools
from google.colab import output
from IPython.display import display,HTML
BASE=Path('/content/research4/artifacts/cached_memory_transport_v1');ROOT=BASE/'colab_cpu_20261014'
receipt=json.loads((BASE/'NATIVE_EXPORT_RECEIPT_20261014.json').read_text())
assert receipt['status']=='native_complete' and receipt['model_calls']==300
assert receipt['sha256']=='be2cd385724c3b49ca6c27719d699bec93fee6a5a1c99bfc34eeb69da0e47658'
archive=Path(receipt['path']);findings=BASE/'R4_MEMORY_TRANSPORT_FINDINGS_20261014.zip';assert findings.is_file()
directory=Path('/content/r5_private_export');directory.mkdir(exist_ok=True)
parts=[];whole=hashlib.sha256()
with archive.open('rb') as source:
    index=0
    while True:
        payload=source.read(64*1024**2)
        if not payload:break
        whole.update(payload);member=f'R4_M15_RAW_{index:04d}.part';target=directory/(member+'.zip');digest=hashlib.sha256(payload).hexdigest()
        if not target.exists():
            with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_STORED) as z:z.writestr(member,payload)
        with zipfile.ZipFile(target) as z:assert z.namelist()==[member] and hashlib.sha256(z.read(member)).hexdigest()==digest
        parts.append({'index':index,'name':target.name,'member':member,'bytes':len(payload),'sha256':digest});index+=1
assert whole.hexdigest()==receipt['sha256'] and archive.stat().st_size==receipt['bytes']
target=directory/findings.name
if not target.exists():os.link(findings,target)
manifest={'archive':archive.name,'archive_sha256':whole.hexdigest(),'archive_bytes':archive.stat().st_size,'archive_members':receipt['members'],
    'parts':parts,'native_receipt':receipt,'parent_archive_sha256':'55717a3be9d192ac984187dcfc42e061fd7763a157e53f943185ca08bdaed3a0',
    'findings':{'name':findings.name,'bytes':findings.stat().st_size,'sha256':hashlib.sha256(findings.read_bytes()).hexdigest()},'new_model_calls':0}
mp=directory/'R4_M15_TRANSFER_MANIFEST_20261014.zip'
if not mp.exists():
    with zipfile.ZipFile(mp,'x',compression=zipfile.ZIP_DEFLATED) as z:z.writestr('R4_M15_TRANSFER_MANIFEST_20261014.json',json.dumps(manifest,indent=2)+'\n')
links=[mp.name,findings.name]+[p['name'] for p in parts]
(directory/'index.html').write_text('<h1>Research4 verified memory-transport exports</h1>'+''.join('<p><a download href="'+html.escape(n)+'">'+html.escape(n)+'</a></p>' for n in links))
if 'r5_export_server' not in globals():
    handler=functools.partial(http.server.SimpleHTTPRequestHandler,directory=str(directory))
    r5_export_server=http.server.ThreadingHTTPServer(('127.0.0.1',8766),handler)
    threading.Thread(target=r5_export_server.serve_forever,daemon=True).start()
r5_proxy=output.eval_js('google.colab.kernel.proxyPort(8766)',timeout_sec=10)
display(HTML('<a href="'+html.escape(r5_proxy)+'" target="_blank">Open memory-transport export</a>'))
cal=json.loads((ROOT/'coefficients/r14_cal_window_d1.json').read_text())
print('R4_MEMORY_CUBIC_CALIBRATION_FAILURE',json.dumps(cal['third_validation']),flush=True)
print('R4_MEMORY_TRANSFER_READY',json.dumps({'parts':len(parts),'bytes':receipt['bytes'],'sha256':receipt['sha256'],'findings':manifest['findings']}),flush=True)

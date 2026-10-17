"""Private, bounded transfer of native results, including a terminal fault."""
from pathlib import Path
import hashlib,html,json,os,time,zipfile,http.server,threading,functools
from google.colab import output
from IPython.display import display,HTML
BEGIN=time.monotonic()
BASE=Path('/content/research4/artifacts/fresh_memory_boundary_response_v1')
ROOT=BASE/'colab_cpu_20261014';RUN=ROOT/'fresh'
receipt=json.loads((BASE/'FRESH_NATIVE_EXPORT_RECEIPT_20261014.json').read_text())
assert 8<=receipt['model_calls']<=300 and 0<=receipt['fresh_calls']<=292
assert receipt['model_worker_exited'] and receipt['GPU_allocations']==0
archive=Path(receipt['archive']);assert archive.is_file()
findings=BASE/'R4_FRESH_MEMORY_BOUNDARY_FINDINGS_20261014.zip'
if not findings.exists():
    # Preserve faults or analysis failures without turning them into successes.
    files={}
    for name in ['STATUS.json','FAULT.txt','CALIBRATION_NUMERICS.json','NUMERICAL_SELECTION_LOCK.json','CHAIN_CONTROLS.json','CALL_LEDGER.json']:
        if (RUN/name).is_file():files['native/'+name]=RUN/name
    for name in ['FRESH_NATIVE_EXPORT_RECEIPT_20261014.json','FRESH_NATIVE_LOG_20261014.txt','FRESH_ANALYSIS_LOG_20261014.txt','FRESH_LAUNCH_LEDGER_20261014.json']:
        if (BASE/name).is_file():files['receipts/'+name]=BASE/name
    manifest={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in files.items()}
    findings=BASE/'R4_FRESH_MEMORY_BOUNDARY_TERMINAL_DIAGNOSTICS_20261014.zip'
    with zipfile.ZipFile(findings,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=2) as z:
        for n,p in files.items():z.write(p,n)
        z.writestr('CONTENTS_SHA256.json',json.dumps(manifest,indent=2)+'\n')
directory=Path('/content/r17_fresh_private_export');directory.mkdir(exist_ok=True)
parts=[];whole=hashlib.sha256()
with archive.open('rb') as source:
    index=0
    while True:
        payload=source.read(64*1024**2)
        if not payload:break
        whole.update(payload);member=f'R4_M17_FRESH_RAW_{index:04d}.part';target=directory/(member+'.zip');digest=hashlib.sha256(payload).hexdigest()
        if not target.exists():
            with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_STORED) as z:z.writestr(member,payload)
        with zipfile.ZipFile(target) as z:assert z.namelist()==[member] and hashlib.sha256(z.read(member)).hexdigest()==digest
        parts.append({'index':index,'name':target.name,'member':member,'bytes':len(payload),'sha256':digest});index+=1
assert whole.hexdigest()==receipt['sha256'] and archive.stat().st_size==receipt['bytes']
target=directory/findings.name
if not target.exists():os.link(findings,target)
compute=json.loads((BASE/'FRESH_COMPUTE_LEDGER_20261014.json').read_text()) if (BASE/'FRESH_COMPUTE_LEDGER_20261014.json').is_file() else None
manifest={'archive':archive.name,'archive_sha256':whole.hexdigest(),'archive_bytes':archive.stat().st_size,'archive_members':receipt['members'],
    'parts':parts,'native_receipt':receipt,'compute_ledger':compute,
    'findings':{'name':findings.name,'bytes':findings.stat().st_size,'sha256':hashlib.sha256(findings.read_bytes()).hexdigest()},
    'phase_historical_repair_calls':8,'new_model_calls_in_exporter':0,'export_preparation_seconds':time.monotonic()-BEGIN}
mp=directory/'R4_M17_FRESH_TRANSFER_20261014.zip'
assert not mp.exists(),'Inspect an already prepared transfer; do not replace its manifest'
with zipfile.ZipFile(mp,'x',compression=zipfile.ZIP_DEFLATED) as z:z.writestr('R4_M17_FRESH_TRANSFER_20261014.json',json.dumps(manifest,indent=2)+'\n')
links=[mp.name,findings.name]+[p['name'] for p in parts]
(directory/'index.html').write_text('<h2>Research4: verified fresh-response exports</h2><p>CPU only. Native phase calls: '+str(receipt['model_calls'])+'. Native status: '+html.escape(receipt['status'])+'.</p>'+''.join('<p><a download href="'+html.escape(n)+'">'+html.escape(n)+'</a></p>' for n in links))
if 'r17_export_server' not in globals():
    handler=functools.partial(http.server.SimpleHTTPRequestHandler,directory=str(directory))
    r17_export_server=http.server.ThreadingHTTPServer(('127.0.0.1',8766),handler)
    threading.Thread(target=r17_export_server.serve_forever,daemon=True).start()
proxy=output.eval_js('google.colab.kernel.proxyPort(8766)',timeout_sec=10)
display(HTML('<a href="'+html.escape(proxy)+'" target="_blank">Private fresh-response export</a>'))
print('R4_FRESH_TRANSFER_READY',json.dumps({'parts':len(parts),'bytes':receipt['bytes'],'sha256':receipt['sha256'],
    'findings':manifest['findings'],'native_status':receipt['status'],'calls':receipt['model_calls'],'new_calls':receipt['fresh_calls'],
    'export_preparation_seconds':manifest['export_preparation_seconds']}),flush=True)

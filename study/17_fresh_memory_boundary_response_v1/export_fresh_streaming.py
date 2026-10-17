"""Read-only private ZIP ranges, wrapped on demand without duplicate disk writes."""
from pathlib import Path
import hashlib,html,http.server,io,json,threading,time,zipfile
from urllib.parse import urlsplit,unquote
from google.colab import output
from IPython.display import display,HTML
BEGIN=time.monotonic()
BASE=Path('/content/research4/artifacts/fresh_memory_boundary_response_v1')
receipt=json.loads((BASE/'FRESH_NATIVE_EXPORT_RECEIPT_20261014.json').read_text())
assert receipt['model_calls']==300 and receipt['fresh_calls']==292 and receipt['status']=='native_complete'
assert receipt['bytes']==7284116733 and receipt['sha256']=='1364223250a6d98af7524b20b3bee328861481af7e0e6d57f9e808432e3577fb'
archive=Path(receipt['archive']);assert archive.stat().st_size==receipt['bytes']
findings=BASE/'R4_FRESH_MEMORY_BOUNDARY_FINDINGS_20261014_RETRY1.zip';assert findings.is_file()
supplement=BASE/'R4_FRESH_OUTPUT_AUDIT_20261014.zip'
compute=json.loads((BASE/'FRESH_COMPUTE_LEDGER_20261014.json').read_text());assert compute['exit_code']==0
directory=Path('/content/r17_fresh_private_export_streaming');directory.mkdir(exist_ok=False)
parts=[];whole=hashlib.sha256();offset=0
with archive.open('rb') as source:
    index=0
    while True:
        payload=source.read(64*1024**2)
        if not payload:break
        whole.update(payload);member=f'R4_M17_FRESH_RAW_{index:04d}.part'
        parts.append({'index':index,'name':member+'.zip','member':member,'bytes':len(payload),
                      'sha256':hashlib.sha256(payload).hexdigest(),'offset':offset})
        offset+=len(payload);index+=1
assert whole.hexdigest()==receipt['sha256'] and offset==receipt['bytes']
manifest={'archive':archive.name,'archive_sha256':whole.hexdigest(),'archive_bytes':receipt['bytes'],'archive_members':receipt['members'],
    'parts':parts,'native_receipt':receipt,'compute_ledger':compute,
    'findings':{'name':findings.name,'bytes':findings.stat().st_size,'sha256':hashlib.sha256(findings.read_bytes()).hexdigest()},
    'phase_historical_repair_calls':8,'new_model_calls_in_exporter':0,'export_preparation_seconds':time.monotonic()-BEGIN,
    'private_streamed_range_wrappers':True,'raw_archive_unchanged':True}
if supplement.is_file():manifest['supplemental']={'name':supplement.name,'bytes':supplement.stat().st_size,'sha256':hashlib.sha256(supplement.read_bytes()).hexdigest()}
mp=directory/'R4_M17_FRESH_TRANSFER_20261014.zip'
with zipfile.ZipFile(mp,'x',compression=zipfile.ZIP_DEFLATED) as z:z.writestr('R4_M17_FRESH_TRANSFER_20261014.json',json.dumps(manifest,indent=2)+'\n')
links=[mp.name,findings.name]+[p['name'] for p in parts]
if supplement.is_file():links.insert(2,supplement.name)
index_html=('<h2>Research4: fresh-response exports</h2><p>300 CPU model runs completed. Original native data and report-saving failure preserved; corrected audit results included. GPU use: zero.</p>'+''.join('<p><a download href="'+html.escape(n)+'">'+html.escape(n)+'</a></p>' for n in links)).encode()
by_name={p['name']:p for p in parts};small_files={mp.name:mp,findings.name:findings}
if supplement.is_file():small_files[supplement.name]=supplement
r17_transfer_timings=[]
class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_HEAD(self):
        name=unquote(urlsplit(self.path).path).lstrip('/')
        if name in ['', 'index.html']:length=len(index_html);mime='text/html; charset=utf-8'
        elif name in small_files:length=small_files[name].stat().st_size;mime='application/zip'
        elif name in by_name:
            part=by_name[name];wrapper=io.BytesIO()
            with zipfile.ZipFile(wrapper,'w',compression=zipfile.ZIP_STORED) as z:z.writestr(part['member'],b'')
            length=part['bytes']+len(wrapper.getvalue());mime='application/zip'
        else:self.send_error(404);return
        self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(length));self.end_headers()
    def do_GET(self):
        start=time.monotonic();name=unquote(urlsplit(self.path).path).lstrip('/')
        if name in ['', 'index.html']:
            data=index_html;mime='text/html; charset=utf-8'
        elif name in small_files:
            data=small_files[name].read_bytes();mime='application/zip'
        elif name in by_name:
            part=by_name[name]
            with archive.open('rb') as source:source.seek(part['offset']);payload=source.read(part['bytes'])
            assert len(payload)==part['bytes'] and hashlib.sha256(payload).hexdigest()==part['sha256']
            wrapper=io.BytesIO()
            with zipfile.ZipFile(wrapper,'w',compression=zipfile.ZIP_STORED) as z:z.writestr(part['member'],payload)
            data=wrapper.getvalue();mime='application/zip'
            r17_transfer_timings.append({'part':part['index'],'range_read_hash_and_wrapper_seconds':time.monotonic()-start,'bytes':part['bytes']})
        else:self.send_error(404);return
        self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(data)))
        if name in by_name or name in small_files:self.send_header('Content-Disposition','attachment; filename="'+name+'"')
        self.end_headers();self.wfile.write(data)
assert 'r17_export_server' not in globals(),'Inspect an existing owned transfer before launching another'
r17_export_server=http.server.ThreadingHTTPServer(('127.0.0.1',8766),Handler)
threading.Thread(target=r17_export_server.serve_forever,daemon=True).start()
proxy=output.eval_js('google.colab.kernel.proxyPort(8766)',timeout_sec=10)
display(HTML('<a href="'+html.escape(proxy)+'" target="_blank">Private fresh-response export</a>'))
print('R4_FRESH_STREAMING_TRANSFER_READY',json.dumps({'parts':len(parts),'bytes':receipt['bytes'],'sha256':receipt['sha256'],
    'findings':manifest['findings'],'export_preparation_seconds':manifest['export_preparation_seconds'],'new_model_calls':0}),flush=True)

# Caller sets pinned ANALYSIS_URL and ANALYSIS_SHA after committing analysis.
import base64,gzip,hashlib,json,re,subprocess
from urllib.request import urlopen
assert ANALYSIS_URL and len(ANALYSIS_SHA)==64
for worker in ROOT.rglob('worker.py'):
    assert not subprocess.run(['pgrep','-f',re.escape(str(worker))],capture_output=True,text=True).stdout.strip()
with urlopen(ANALYSIS_URL,timeout=120) as response: analysis_source=response.read()
assert hashlib.sha256(analysis_source).hexdigest()==ANALYSIS_SHA
scope={'__name__':'research3_narrow_post_run_analysis'}
exec(compile(analysis_source.decode('utf-8-sig'),'trusted_narrow_analysis.py','exec'),scope)
ANALYSIS_PACKET=scope['collect'](ROOT)
ANALYSIS_HASHES=scope['write_packet'](ANALYSIS_PACKET,ROOT/'review_documentation/narrow_bridge_analysis')
(ROOT/'review_documentation/narrow_bridge_analysis/ANALYSIS_SOURCE_PIN.json').write_text(
    json.dumps({'url':ANALYSIS_URL,'sha256':ANALYSIS_SHA,'derived_file_sha256':ANALYSIS_HASHES},indent=2))
print('NARROW_POST_RUN_ANALYSIS_READY',json.dumps({
    'completed':ANALYSIS_PACKET['summary']['worker_completed'],
    'arms':{name:value['derived_known_by_condition'] for name,value in ANALYSIS_PACKET['summary']['arms'].items()},
    'parse_errors':ANALYSIS_PACKET['summary']['parse_errors']}),flush=True)
print('NARROW_ANALYSIS_PACKET_BEGIN'+base64.b64encode(gzip.compress(
    json.dumps(ANALYSIS_PACKET).encode(),mtime=0)).decode()+'NARROW_ANALYSIS_PACKET_END',flush=True)
# Run the existing frozen collect/export cell next; this cell does not release it.


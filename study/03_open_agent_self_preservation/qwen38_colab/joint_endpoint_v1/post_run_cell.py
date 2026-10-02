# Caller pins STRICT_ENDPOINT_URL and STRICT_ENDPOINT_SHA before execution.
import base64,gzip,hashlib,json,re,subprocess
from urllib.request import urlopen
STRICT_PARENT_ROOT=globals().get('PARENT_ROOT',ROOT)
STRICT_CAPACITY_ROOT=ROOT if STRICT_PARENT_ROOT!=ROOT else None
for run_root in ([STRICT_PARENT_ROOT,STRICT_CAPACITY_ROOT] if STRICT_CAPACITY_ROOT else [STRICT_PARENT_ROOT]):
    for worker in run_root.rglob('worker.py'):
        assert not subprocess.run(['pgrep','-f',re.escape(str(worker))],capture_output=True,text=True).stdout.strip()
with urlopen(STRICT_ENDPOINT_URL,timeout=120) as response: strict_source=response.read()
assert hashlib.sha256(strict_source).hexdigest()==STRICT_ENDPOINT_SHA
strict_scope={'__name__':'strict_saved_endpoint_analysis'}
exec(compile(strict_source.decode('utf-8-sig'),'trusted_strict_endpoint.py','exec'),strict_scope)
STRICT_PACKET=strict_scope['collect'](STRICT_PARENT_ROOT,STRICT_CAPACITY_ROOT)
STRICT_HASHES=strict_scope['write_packet'](STRICT_PACKET,ROOT/'review_documentation/strict_joint_endpoint')
print('STRICT_ENDPOINT_DECISIONS',json.dumps({name:value['additional_no_regression']
    for name,value in STRICT_PACKET['models'].items()}),flush=True)
print('STRICT_ENDPOINT_PACKET_BEGIN'+base64.b64encode(gzip.compress(
    json.dumps(STRICT_PACKET).encode(),mtime=0)).decode()+'STRICT_ENDPOINT_PACKET_END',flush=True)
# Export through the existing verified closeout; this cell makes no model query.


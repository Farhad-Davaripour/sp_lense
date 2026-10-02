# Run after both workers exit; caller sets the committed collector URL/hash.
import base64,gzip,hashlib,json,re,subprocess
from urllib.request import urlopen
for run_root in (PARENT_ROOT,ROOT):
    for worker in run_root.rglob('worker.py'):
        assert not subprocess.run(['pgrep','-f',re.escape(str(worker))],capture_output=True,text=True).stdout.strip()
with urlopen(JOINT_BLIND_SOURCE_URL,timeout=120) as response: source=response.read()
assert hashlib.sha256(source).hexdigest()==JOINT_BLIND_SOURCE_SHA
scope={'__name__':'joint_blind_preference_collection'}
exec(compile(source.decode('utf-8-sig'),'trusted_joint_blind96.py','exec'),scope)
JOINT_REVIEW=scope['collect'](PARENT_ROOT,ROOT)
JOINT_REVIEW_RECEIPT=scope['write_packet'](JOINT_REVIEW,ROOT/'review_documentation/joint_preference_review')
print('JOINT_REVIEW_FREEZE',json.dumps(JOINT_REVIEW_RECEIPT),flush=True)
# This transport contains ONLY masked responses. The private key stays on disk.
print('JOINT_MASKED_RESPONSES_BEGIN'+base64.b64encode(gzip.compress(
    json.dumps(JOINT_REVIEW['masked_responses']).encode(),mtime=0)).decode()+
    'JOINT_MASKED_RESPONSES_END',flush=True)
# Give only MASKED_RESPONSES.json to the semantic reviewer. Commit their labels
# with the masked file hash before opening PRIVATE_KEY.json. Export afterward.


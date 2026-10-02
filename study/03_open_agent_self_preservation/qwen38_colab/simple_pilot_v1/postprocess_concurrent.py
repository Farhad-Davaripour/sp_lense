"""Trusted notebook cell, run after the concurrent controller has returned."""
import base64
import gzip
import hashlib
import json
import subprocess
import sys
import urllib.request

pin = 'a1ba4d538ddc9d439be683be0295e297015151f5'
prefix = 'https://raw.githubusercontent.com/Farhad-Davaripour/sp_lense/'+pin+'/study/03_open_agent_self_preservation/qwen38_colab/simple_pilot_v1/'
review_code = CONCURRENT_ROOT/'review_code'
review_code.mkdir(exist_ok=False)
for name, expected in {'review_results.py':'68708f360745f522d6f5d4abc974abe40c31f32ff1c4f4c3229739456c56e02c',
                       'review_concurrent.py':'a36b0d74330313f0d0465be705757b68e65ec1d6286947a8efc6caaeabe4d152'}.items():
    content = urllib.request.urlopen(prefix+name,timeout=30).read()
    if hashlib.sha256(content).hexdigest() != expected:
        raise RuntimeError('Review source checksum mismatch: '+name)
    (review_code/name).write_bytes(content)
result = subprocess.run([sys.executable,str(review_code/'review_concurrent.py'),'--root',str(CONCURRENT_ROOT)],
                        capture_output=True,text=True)
print(result.stdout)
if result.returncode:
    print(result.stderr)
    raise RuntimeError('Review did not complete; original evidence retained')
report_dir = CONCURRENT_ROOT/'reports/comparison_review'
payload = {'summary':json.loads((report_dir/'summary.json').read_text()),
           'masked':json.loads((report_dir/'preference_review_masked.json').read_text()),
           'masked_sha256':hashlib.sha256((report_dir/'preference_review_masked.json').read_bytes()).hexdigest()}
print('MASKED_REVIEW_GZIP_BASE64',base64.b64encode(gzip.compress(json.dumps(payload).encode())).decode())

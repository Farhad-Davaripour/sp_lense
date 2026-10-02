# Worker exit precedes derived analysis and private export.
import re,base64
for name in ('worker.py','diagnostic_worker.py'):
    for worker in ROOT.rglob(name):
        assert not subprocess.run(['pgrep','-f',re.escape(str(worker))],capture_output=True,text=True).stdout.strip()
from urllib.request import urlopen
ANALYSIS_URL='https://raw.githubusercontent.com/Farhad-Davaripour/sp_lense/79c92a40a3ae6b43f338466acb7ebe681ab7d658/study/03_open_agent_self_preservation/qwen38_colab/handoff_reliability_v1/analysis/summarize.py'
ANALYSIS_SHA='ee74b722765d5103cd8f116d3aa67eb9449e2a5e805a17b0dd0d2429a8cb2814'
with urlopen(ANALYSIS_URL,timeout=120) as response: analysis_source=response.read()
assert hashlib.sha256(analysis_source).hexdigest()==ANALYSIS_SHA
analysis_scope={'__name__':'research3_post_run_analysis'}
exec(compile(analysis_source.decode('utf-8'),'trusted_derived_analysis.py','exec'),analysis_scope)
ANALYSIS_PACKET=analysis_scope['collect'](ROOT)
analysis_scope['write_packet'](ANALYSIS_PACKET,ROOT/'review_documentation')
(ROOT/'review_documentation/ANALYSIS_SOURCE_PIN.json').write_text(json.dumps({'url':ANALYSIS_URL,'sha256':ANALYSIS_SHA},indent=2))
print('POST_RUN_ANALYSIS_READY',json.dumps({'completed':ANALYSIS_PACKET['summary']['worker_completed'],'trajectories':ANALYSIS_PACKET['summary']['evidence_files']['completed_trajectory_files'],'generation_calls':ANALYSIS_PACKET['summary']['observations']['actual_generation_call_count']}),flush=True)
print('ANALYSIS_PACKET_BEGIN'+base64.b64encode(gzip.compress(json.dumps(ANALYSIS_PACKET).encode())).decode()+'ANALYSIS_PACKET_END',flush=True)
FAST_ROOT=ROOT
EXPORT_ONLY_ROOTS=globals().get('DIAGNOSTIC_EXPORT_ROOTS',[ROOT])
RELEASE_RUNTIME_AFTER_EXPORT=True
exec(compile((ROOT/'source/export_private_runs.py').read_text(),'trusted_private_export.py','exec'))

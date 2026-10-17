"""Extract private frozen resume/audit sources; launch only the CPU native continuation."""
from pathlib import Path
import json,hashlib,zipfile,subprocess,shutil
B=Path('/content/research4/artifacts/coupled_memory_blocks_v1')
R=B/'run_20261015_resume1';R.mkdir(exist_ok=False)
Z=Path('/content/R4_COUPLED_RESUME_SOURCE_20261015.zip')
with zipfile.ZipFile(Z) as z:
    assert set(z.namelist())=={'resume.py','independent_audit.py','RESUME_PROTOCOL.json'}
    for n in z.namelist():
        target=R/'source'/n;target.parent.mkdir(exist_ok=True);target.write_bytes(z.read(n))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
freeze={'sha256':{p.relative_to(R).as_posix():sha(p) for p in (R/'source').iterdir()},'original_stop_proof_sha256':'67bde19b417ea22b276f143ad59dfa4d71deda31e84bfd9f861e3b12ea07c46e','original_predictions_unchanged':True,'held_targets_seen_before_resume':0,'archive_sha256':sha(Z)}
assert sha(B/'R4_COUPLED_ORIGINAL_STOP_PROOF_20261015.zip')==freeze['original_stop_proof_sha256']
(R/'RESUME_SOURCE_FREEZE.json').write_text(json.dumps(freeze,indent=2)+'\n')
log=(B/'RESUME_RUN_LOG_20261015.txt').open('x')
p=subprocess.Popen(['/content/r4_cached_cpu_env/bin/python','-B',str(R/'source/resume.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(B/'RESUME_PID_20261015.json').write_text(json.dumps({'PID':p.pid,'script_sha256':sha(R/'source/resume.py'),'combined_calls_planned':76})+'\n')
print('R4_COUPLED_RESUME_LAUNCHED',p.pid,'combined_calls_planned',76)

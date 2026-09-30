"""Output-free, source-bundled Colab entrypoint; no runtime allocation locally."""
import base64
import gzip
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
SIMPLE=HERE.parent/'simple_pilot_v1'
FRESH=HERE.parent/'fresh_transfer_v1'


def cell(kind,text):
    row={'cell_type':kind,'metadata':{},'source':text.splitlines(keepends=True)}
    if kind=='code': row.update(outputs=[],execution_count=None)
    return row


def main():
    files={p.name:p.read_text(encoding='utf-8') for p in HERE.glob('*.py') if p.name!='build_notebook.py'}
    files['PROTOCOL.md']=(HERE/'PROTOCOL.md').read_text()
    mapping={'model_ops.py':'model_ops.py','batching.py':'batching.py','world.py':'world.py','concurrent_controller.py':'concurrent_controller.py',
             'study_worker.py':'worker.py','fast_worker.py':'worker_fast.py','fast_inference.py':'fast_inference.py',
             'diagnostic_world.py':'diagnostic_world.py','build_diagnostics.py':'build_diagnostics.py','requirements.txt':'requirements.txt',
             'export_private_runs.py':'export_private_runs.py'}
    for target,source in mapping.items(): files[target]=(SIMPLE/source).read_text(encoding='utf-8')
    for name in ('benign_competence_dev.json','preference_validation.json'): files['data/'+name]=(SIMPLE/'data'/name).read_text()
    files['fresh_world.py']=(FRESH/'fresh_world.py').read_text()
    files['inspected_cases.py']=(FRESH/'cases.py').read_text()
    files['activation_capture.py']=(FRESH/'activation_capture.py').read_text()
    files['model_pin.json']=(HERE.parent/'model_pin.json').read_text(encoding='utf-8-sig')
    freeze={'status':'Source/conditional recipes frozen before launch diagnostic','sha256':{name:hashlib.sha256(text.encode()).hexdigest() for name,text in files.items()}}
    files['SOURCE_FREEZE.json']=json.dumps(freeze,indent=2)
    (HERE/'SOURCE_FREEZE.json').write_text(files['SOURCE_FREEZE.json']+'\n',encoding='utf-8',newline='\n')
    packed=base64.b64encode(gzip.compress(json.dumps(files).encode(),mtime=0)).decode()
    setup='''import base64, gzip, hashlib, json, subprocess, sys, time, uuid
from pathlib import Path
SESSION_STARTED=time.monotonic()
# Refresh these two values from the Resources panel at actual launch.
BALANCE_AT_LAUNCH=154.65
RATE_AT_LAUNCH=6.77
AUTHORIZED_TOTAL_UNITS=200
PRIOR_SPEND=47.48
STAGE_CAP_UNITS=16
RESERVE_UNITS=1.5
assert STAGE_CAP_UNITS<=AUTHORIZED_TOTAL_UNITS-PRIOR_SPEND
gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
assert 'A100' in gpu and int(gpu.split(',')[1].split()[0])>=75000, 'Use the single A100 80GB runtime'
STAGE_ROOT=Path('/content/sp_lense_work')/('qwen38_H2_deadline_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8])
STAGE_ROOT.mkdir(parents=True,exist_ok=False)
SOURCE_FILES=json.loads(gzip.decompress(base64.b64decode(PACKED_SOURCE)))
for name,content in SOURCE_FILES.items():
    path=STAGE_ROOT/'source'/name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(content,encoding='utf-8',newline='\\n')
for name,expected in json.loads(SOURCE_FILES['SOURCE_FREEZE.json'])['sha256'].items():
    assert hashlib.sha256((STAGE_ROOT/'source'/name).read_bytes()).hexdigest()==expected
subprocess.run([sys.executable,'-m','pip','install','--quiet','--no-input','-r',str(STAGE_ROOT/'source/requirements.txt')],check=True,timeout=900)
print('H2_STAGE_READY',STAGE_ROOT,'account',BALANCE_AT_LAUNCH,'authorized remaining',AUTHORIZED_TOTAL_UNITS-PRIOR_SPEND,'GPU',gpu,flush=True)
'''
    setup=setup.replace('SOURCE_FILES=',"PACKED_SOURCE='"+packed+"'\nSOURCE_FILES=",1)
    restore='''from google.colab import drive
import shutil
drive.mount('/content/drive')
original=Path('/content/drive/MyDrive/sp_lense/research3/runs/qwen38_preservation_hp_20260930T180110Z_0271ecd2/H2_rank16')
INPUT_ADAPTER=STAGE_ROOT/'inputs/selected_H2'
INPUT_TRAIN=STAGE_ROOT/'inputs/selected_H2_training.json'
shutil.copytree(original/'checkpoints/adapters/preservation',INPUT_ADAPTER)
shutil.copyfile(original/'code/data/train.json',INPUT_TRAIN)
H2_SHA='0782b3f3ea9b3e67460b7bc981e21526bb6819d3ddb9623def4fbc2218002b30'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(16*1024**2),b''): h.update(block)
    return h.hexdigest()
assert sha(INPUT_ADAPTER/'adapter_model.safetensors')==H2_SHA
assert sha(INPUT_TRAIN)=='14948c6c2ed6acda56b47dced7fead31f36fb46fab8f7471c6346c1277892464'
drive.flush_and_unmount()
from huggingface_hub import snapshot_download
pin=json.loads((STAGE_ROOT/'source/model_pin.json').read_text())
snapshot_download(repo_id=pin['repository'],revision=pin['revision'],local_dir=STAGE_ROOT/'model',token=False,max_workers=8,
    allow_patterns=['*.safetensors','*.json','*.jinja','merges.txt','vocab.json'])
verified={}
for item in pin['files']:
    p=STAGE_ROOT/'model'/item['name']
    if p.is_file():
        actual=sha(p)
        if item['sha256']: assert actual==item['sha256'], item['name']
        verified[item['name']]={'bytes':p.stat().st_size,'sha256':actual}
assert sum(name.endswith('.safetensors') for name in verified)==18
(STAGE_ROOT/'MODEL_HASHES.json').write_text(json.dumps(verified,indent=2))
print('PINNED_BASE_AND_SELECTED_H2_RESTORED',len(verified),'files',flush=True)
'''
    launch="exec(compile((STAGE_ROOT/'source/launch.py').read_text(),'trusted_H2_launch.py','exec'))\n"
    close='''# Always run after the launch cell returns, including an incomplete fit.
import re
for p in STAGE_ROOT.rglob('worker.py'):
    assert not subprocess.run(['pgrep','-f',re.escape(str(p))],capture_output=True,text=True).stdout.strip()
ROOT=FAST_ROOT=STAGE_ROOT
EXPORT_ONLY_ROOTS=[STAGE_ROOT]
RELEASE_RUNTIME_AFTER_EXPORT=True
exec(compile((STAGE_ROOT/'source/collect_review.py').read_text(),'trusted_collect_review.py','exec'))
exec(compile((STAGE_ROOT/'source/export_private_runs.py').read_text(),'trusted_private_export.py','exec'))
'''
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'colab':{'name':'Research3_H2_Deadline_Followups_V1.ipynb'},
       'kernelspec':{'name':'python3','display_name':'Python 3'},'accelerator':'GPU'},'cells':[
       cell('markdown','# Research 3 — two H2 second-stage deadline follow-ups\n\nOne A100 80GB runtime. Source, criteria and conditional curricula are frozen. Both fits build on the saved H2 adapter with fresh optimizers. Model tools edit fictional memory only. Cumulative authorization: 200 units, including 47.48 previously spent. Account balance must be read independently. Context3072 and matched microbatch1 preserve complete routes with GPU headroom. Stage cap16 units with saving/shutdown reserve. Run these four cells in order; the final cell privately exports and releases the runtime. No purchase or second runtime.\n'),
       cell('code',setup),cell('code',restore),cell('code',launch),cell('code',close)]}
    (HERE/'Research3_H2_Deadline_Followups_V1.ipynb').write_text(json.dumps(notebook,indent=1)+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__': main()

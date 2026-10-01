"""Source-bundled Colab entrypoint for the two bounded registered-scope streams."""
import base64
import gzip
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
SIMPLE=HERE.parent/'simple_pilot_v1'
PRIOR=HERE.parent/'h2_deadline_followup_v1'
FRESH=HERE.parent/'fresh_transfer_v1'


def cell(kind,text):
    value={'cell_type':kind,'metadata':{},'source':text.splitlines(keepends=True)}
    if kind=='code':value.update(outputs=[],execution_count=None)
    return value


def main():
    source={p.name:p.read_text(encoding='utf-8') for p in HERE.glob('*.py') if p.name!='build_notebook.py'}
    for name in ('PROTOCOL.md','SCENARIO_REGISTRY.md'):source[name]=(HERE/name).read_text()
    for target,name in {'model_ops.py':'model_ops.py','fast_inference.py':'fast_inference.py','world.py':'world.py',
        'batching.py':'batching.py','concurrent_controller.py':'concurrent_controller.py','export_private_runs.py':'export_private_runs.py',
        'requirements.txt':'requirements.txt','study_worker.py':'worker.py','fast_worker.py':'worker_fast.py',
        'diagnostic_world.py':'diagnostic_world.py','build_diagnostics.py':'build_diagnostics.py'}.items():source[target]=(SIMPLE/name).read_text()
    for name in ('memory_world.py','recipes.py','paired_batching.py','evaluate.py'):source[name]=(PRIOR/name).read_text()
    source['fresh_world.py']=(FRESH/'fresh_world.py').read_text()
    source['activation_capture.py']=(FRESH/'activation_capture.py').read_text()
    for name in ('benign_competence_dev.json','preference_validation.json'):source['data/'+name]=(SIMPLE/'data'/name).read_text()
    source['model_pin.json']=(HERE.parent/'model_pin.json').read_text(encoding='utf-8-sig')
    freeze={'status':'Frozen before both experimental streams','sha256':{name:hashlib.sha256(text.encode()).hexdigest() for name,text in source.items()}}
    source['SOURCE_FREEZE.json']=json.dumps(freeze,indent=2)
    (HERE/'SOURCE_FREEZE.json').write_text(source['SOURCE_FREEZE.json']+'\n',encoding='utf-8',newline='\n')
    packed=base64.b64encode(gzip.compress(json.dumps(source).encode(),mtime=0)).decode()
    setup='''import base64,gzip,hashlib,json,subprocess,sys,time,uuid
from pathlib import Path
SESSION_STARTED=time.monotonic()
BALANCE_AT_LAUNCH=143.58
RATE_AT_LAUNCH=6.77
PRIOR_SPEND=58.55
AUTHORIZED_TOTAL_UNITS=200
STAGE_CAP_UNITS=24
RESERVE_UNITS=2
assert STAGE_CAP_UNITS<=min(BALANCE_AT_LAUNCH,AUTHORIZED_TOTAL_UNITS-PRIOR_SPEND)
gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
assert 'A100' in gpu and int(gpu.split(',')[1].split()[0])>=75000
ROOT=Path('/content/sp_lense_work')/('qwen38_H2_replay_diagnostics_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8])
ROOT.mkdir(parents=True,exist_ok=False)
FILES=json.loads(gzip.decompress(base64.b64decode(PACKED_SOURCE)))
for name,content in FILES.items():
    path=ROOT/'source'/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(content,encoding='utf-8',newline='\\n')
for name,expected in json.loads(FILES['SOURCE_FREEZE.json'])['sha256'].items():
    assert hashlib.sha256((ROOT/'source'/name).read_bytes()).hexdigest()==expected
subprocess.run([sys.executable,'-m','pip','install','--quiet','--no-input','-r',str(ROOT/'source/requirements.txt')],check=True,timeout=900)
print('REGISTERED_SCOPE_READY',ROOT,gpu,'authorized remaining',AUTHORIZED_TOTAL_UNITS-PRIOR_SPEND,flush=True)
'''
    setup=setup.replace('FILES=',"PACKED_SOURCE='"+packed+"'\nFILES=",1)
    restore='''from google.colab import drive
import shutil
drive.mount('/content/drive')
run_parent=Path('/content/drive/MyDrive/sp_lense/research3/runs')
old=run_parent/'qwen38_preservation_hp_20260930T180110Z_0271ecd2'
prior=run_parent/'qwen38_H2_deadline_20260930T204950Z_d8674557'
MODEL_INPUTS={
 'H2':{'source':str(old/'H2_rank16/checkpoints/adapters/preservation'),'weights':'0782b3f3ea9b3e67460b7bc981e21526bb6819d3ddb9623def4fbc2218002b30','config':'3df028243d1d0f6643b906143a4c1ad4e8393b5e947d8b52c03f0893601b3528'},
 'A':{'source':str(prior/'Job_A/checkpoints/adapters/final'),'weights':'cdfa73ffb4ec83f0d227e33beadd3494ddfe385502b4e70c424d07bab6bcfd94','config':'85803699832a9d2d0bc9cd72cf70da330f21b3011fc3ea98bddcb2a21805d677'},
 'B':{'source':str(prior/'Job_B/checkpoints/adapters/final'),'weights':'6a71d9c00d6f2f0108d90b78db6304b4949f79666d621972b7687ec67bfff269','config':'4a7e18016d6779de50ff6dbf22a89e2ddbefc260cd75fc1f9cf465ca69c82957'}}
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(16*1024**2),b''):h.update(block)
    return h.hexdigest()
for name,info in MODEL_INPUTS.items():
    target=ROOT/'inputs/models'/name
    shutil.copytree(info['source'],target)
    assert sha(target/'adapter_model.safetensors')==info['weights']
    assert sha(target/'adapter_config.json')==info['config']
shutil.copytree(old/'H2_rank16/code',ROOT/'inputs/original_code')
receipt=json.loads((old/'EXPORT_HASHES.json').read_text())
for file in (ROOT/'inputs/original_code').rglob('*'):
    if file.is_file():
        relative='H2_rank16/code/'+str(file.relative_to(ROOT/'inputs/original_code'))
        assert sha(file)==receipt['files'][relative]['sha256'],relative
shutil.copyfile(prior/'Job_B/code/train.json',ROOT/'inputs/archived_B_train.json')
assert sha(ROOT/'inputs/archived_B_train.json')=='01560c9a65fff138ae117ade35e728ab2d6f244a121517a8583eed07b7265c0c'
assert sha(ROOT/'inputs/original_code/data/train.json')=='14948c6c2ed6acda56b47dced7fead31f36fb46fab8f7471c6346c1277892464'
(ROOT/'RESTORED_INPUTS.json').write_text(json.dumps(MODEL_INPUTS,indent=2))
drive.flush_and_unmount()
from huggingface_hub import snapshot_download
pin=json.loads((ROOT/'source/model_pin.json').read_text())
snapshot_download(repo_id=pin['repository'],revision=pin['revision'],local_dir=ROOT/'model',token=False,max_workers=8,
 allow_patterns=['*.safetensors','*.json','*.jinja','merges.txt','vocab.json'])
manifest={}
for item in pin['files']:
    p=ROOT/'model'/item['name']
    if p.is_file():
        digest=sha(p)
        if item['sha256']:assert digest==item['sha256'],item['name']
        manifest[item['name']]={'bytes':p.stat().st_size,'sha256':digest}
assert sum(n.endswith('.safetensors') for n in manifest)==18
(ROOT/'MODEL_HASHES.json').write_text(json.dumps(manifest,indent=2))
print('H2_A_B_BASE_AND_ARCHIVED_B80_RESTORED',flush=True)
'''
    launch="exec(compile((ROOT/'source/launch.py').read_text(),'trusted_registered_streams.py','exec'))\n"
    close='''# Run after both stream workers exit, including an incomplete experiment.
import re
for worker in ROOT.rglob('worker.py'):
    assert not subprocess.run(['pgrep','-f',re.escape(str(worker))],capture_output=True,text=True).stdout.strip()
exec(compile((ROOT/'source/collect.py').read_text(),'trusted_collect_streams.py','exec'))
FAST_ROOT=ROOT
EXPORT_ONLY_ROOTS=[ROOT]
RELEASE_RUNTIME_AFTER_EXPORT=True
exec(compile((ROOT/'source/export_private_runs.py').read_text(),'trusted_export_streams.py','exec'))
'''
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'colab':{'name':'Research3_H2_Replay_Diagnostics_V1.ipynb'},
      'kernelspec':{'name':'python3','display_name':'Python 3'},'accelerator':'GPU'},'cells':[
      cell('markdown','# Research 3 — three-setting diagnostics and replay coverage\n\nOne A10080GB; two concurrent experiment streams, never more than two resident model workers. Frozen H2/A/B diagnostic loads are sequential; reference/coverage fits are sequential in the other worker. Cumulative authorization200 units,58.55 spent before this run. Stage cap24, reserve2. Only original H2, existing one-step and existing ordered pending settings; no new families or schedules. Model tools are fictional memory operations. Run setup, restore, launch, closeout in order. Source/data freezes and all old scores are preserved.\n'),
      cell('code',setup),cell('code',restore),cell('code',launch),cell('code',close)]}
    (HERE/'Research3_H2_Replay_Diagnostics_V1.ipynb').write_text(json.dumps(nb,indent=1)+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':main()

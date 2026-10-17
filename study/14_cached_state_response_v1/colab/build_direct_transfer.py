"""Direct private notebook upload/download, avoiding a failed Drive mount."""
import ast,hashlib,json,shutil,zipfile
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=Path('C:/Users/farha/Research4/artifacts/cached_state_response_v1')
OLD=BASE/'colab_prepared_20261014'/'colab_cpu_20261014'
ROOT=BASE/'colab_direct_prepared_20261014'/'colab_cpu_20261014'
shutil.copytree(OLD,ROOT)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
worker=(ROOT/'source/worker.py').read_text()
worker=worker.replace("archive=Path('/content/drive/MyDrive/sp_lense/research4/cached_state_response_v1')/", "archive=ROOT.parent/")
oldcode="    old=Path('/content/drive/MyDrive/sp_lense/research4/finite_logit_response_v1')\n    immutable={p.name:sha(p) for p in old.glob('R4_FINITE_LOGIT_*.zip')};assert len(immutable)==2"
assert oldcode in worker
worker=worker.replace(oldcode,"    immutable=json.loads((ROOT/'PLATFORM_AMENDMENT.json').read_text())['original_study13_archive_hashes'];assert len(immutable)==2")
worker=worker.replace("    assert all(sha(old/n)==h for n,h in immutable.items())\n",'')
worker=worker.replace("{'study13_archives_unchanged':True,'sha256':immutable}","{'original_archives_available_in_runtime':False,'requires_external_host_hash_audit':True,'sha256':immutable}")
worker=worker.replace("dump(archive.parent/f'NATIVE_EXPORT_RECEIPT_20261014{suffix}.json',receipt);",'')
(ROOT/'source/worker.py').write_text(worker)
prior=Path('C:/Users/farha/Research4/artifacts/finite_logit_response_v1')
hashes={p.name:sha(p) for p in prior.glob('R4_FINITE_LOGIT_*.zip')};assert len(hashes)==2
amendment=json.loads((ROOT/'PLATFORM_AMENDMENT.json').read_text())
amendment.update(direct_transfer_UTC=datetime.now(timezone.utc).isoformat(),
 transfer='Private direct notebook file upload/download; Drive mount is not needed',
 reason='Drive mounting timed out; connected CPU VM has11.6GiB available. No model calls occurred.',
 original_study13_archive_hashes=hashes,
 baseline_preservation='Host hashes verified before bundle; verify again after downloaded results. Old archives are not accessible to, or modified by, the runtime.')
amendment['changes'].append('Export locally for private browser download instead of mounted Drive')
(ROOT/'PLATFORM_AMENDMENT.json').write_text(json.dumps(amendment,indent=2)+'\n')
for p in (ROOT/'source').glob('*.py'):ast.parse(p.read_text())
files=[p for p in ROOT.rglob('*') if p.is_file() and p.name!='EXECUTION_FREEZE.json']
freeze={'UTC':datetime.now(timezone.utc).isoformat(),'model_calls_before_freeze':0,'sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in files}}
(ROOT/'EXECUTION_FREEZE.json').write_text(json.dumps(freeze,indent=2)+'\n')
archive=BASE/'R4_CACHED_STATE_COLAB_CPU_DIRECT_SOURCE_20261014.zip'
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=2) as z:
 for p in ROOT.rglob('*'):
  if p.is_file():z.write(p,ROOT.name+'/'+p.relative_to(ROOT).as_posix())
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
setup=(HERE/'setup_runtime.py').read_text()
setup=setup.replace("DRIVE=Path('/content/drive/MyDrive/sp_lense/research4/cached_state_response_v1')","DRIVE=Path('/content')")
setup=setup.replace("assert os.path.ismount('/content/drive'),'Complete the Drive-only connection cell first'\n",'')
setup=setup.replace('R4_CACHED_STATE_COLAB_CPU_SOURCE_20261014.zip',archive.name)
setup=setup.replace('8d22cf5de4e02dd9748ede7d734eba439b73c8a0b7bcb31bdc90d48fbaae0dcd',sha(archive))
run=(HERE/'run_experiment.py').read_text()
run=run.replace("DRIVE=Path('/content/drive/MyDrive/sp_lense/research4/cached_state_response_v1')","DRIVE=BASE")
run=run.replace("assert os.path.ismount('/content/drive') and PY.is_file()","assert PY.is_file()")
run=run.replace("shutil.copyfile(BASE/'COLAB_NATIVE_LOG_20261014.txt',DRIVE/'COLAB_NATIVE_LOG_20261014.txt')\n",'')
outputs=[archive]
for name,contents in [('R4_COLAB_DIRECT_SETUP_20261014.py',setup),('R4_COLAB_DIRECT_RUN_20261014.py',run)]:
 ast.parse(contents);p=BASE/name
 with p.open('x',encoding='utf-8') as f:f.write(contents)
 outputs.append(p)
drive=Path('G:/My Drive/sp_lense/research4/cached_state_response_v1')
receipt=[]
for p in outputs:
 dest=drive/p.name
 with p.open('rb') as a,dest.open('xb') as b:shutil.copyfileobj(a,b)
 assert sha(dest)==sha(p)
 receipt.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'drive_copy':str(dest)})
(BASE/'COLAB_DIRECT_SOURCE_RECEIPT_20261014.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))

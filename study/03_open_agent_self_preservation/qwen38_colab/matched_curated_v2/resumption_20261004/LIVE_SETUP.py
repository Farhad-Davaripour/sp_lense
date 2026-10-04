import gzip, hashlib, json, subprocess, sys, time, uuid, signal
from pathlib import Path, PurePosixPath
V2_SETUP_STARTED=time.monotonic()
V2_ACCOUNT_AT_RESUMPTION=62.8
V2_PRIOR_INCURRED=max(0.18,round(63.96-V2_ACCOUNT_AT_RESUMPTION,2))
assert V2_PRIOR_INCURRED<10.5
V2_ROOT=Path('/content/sp_lense_work')/('qwen38_curated_v2_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8])
V2_ROOT.mkdir(parents=True,exist_ok=False)
V2_SOURCE=V2_ROOT/'source'
V2_SOURCE.mkdir()
V2_PACKET=Path('/content/drive/MyDrive/sp_lense/research3/runs/CURATED_V2_FIXED_PRIVATE_PACKET_8e154829.json.gz').read_bytes()
assert hashlib.sha256(V2_PACKET).hexdigest()=='8e154829dea6a02c9af1838ad10d958de6465d17c546c59746e3a60f3adefd18'
V2_FILES=json.loads(gzip.decompress(V2_PACKET))['files']
for _relative,_text in V2_FILES.items():
    _p=PurePosixPath(_relative)
    assert not _p.is_absolute() and '..' not in _p.parts
    _out=V2_SOURCE/_relative
    _out.parent.mkdir(parents=True,exist_ok=True)
    _out.write_bytes(_text.encode('utf-8'))
for _relative,_expected in json.loads((V2_SOURCE/'FREEZE.json').read_text())['sha256'].items():
    assert hashlib.sha256((V2_SOURCE/_relative).read_bytes()).hexdigest()==_expected,_relative
(V2_ROOT/'PRIVATE_PACKET.json.gz').write_bytes(V2_PACKET)
for _module in ('restore_inputs','driver','base_download'):
    sys.modules.pop(_module,None)
sys.path.insert(0,str(V2_SOURCE))
from restore_inputs import restore as v2_restore
V2_MODELS=v2_restore(V2_ROOT)
from google.colab import drive
drive.flush_and_unmount()
V2_CONFIG={'H2':V2_MODELS['H2'],'models':{'A':V2_MODELS['A']},'minimum_B_training_seconds':2031,'B_eval_reserve_seconds':1500}
(V2_ROOT/'config.json').write_text(json.dumps(V2_CONFIG,indent=2))
(V2_ROOT/'SESSION_ADMISSION.json').write_text(json.dumps({'account_at_resumption':V2_ACCOUNT_AT_RESUMPTION,'prior_phase_units_including_CPU_and_GPU_idle':V2_PRIOR_INCURRED,'prior_cumulative_before_phase':138.17,'cumulative_authorization':200,'phase_cap':12,'export_reserve':1.5,'packet_sha256':'8e154829dea6a02c9af1838ad10d958de6465d17c546c59746e3a60f3adefd18','new_training_Jev_calls':0},indent=2))
print('CURATED_V2_RESTORED_UNMOUNTED',V2_ROOT,flush=True)
subprocess.run([sys.executable,'-m','pip','install','--quiet','--no-input','-r',str(V2_SOURCE/'requirements.txt')],check=True,timeout=300)
from base_download import download_base as v2_download_base
V2_BASE_MANIFEST=v2_download_base(V2_ROOT)
assert time.monotonic()-V2_SETUP_STARTED<900,'Hard setup limit exceeded'
print('CURATED_V2_SETUP_READY',V2_ROOT,flush=True)

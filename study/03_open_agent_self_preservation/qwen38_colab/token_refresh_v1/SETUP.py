# Trusted Colab setup only. These global inputs are filled by the private notebook.
import base64,gzip,hashlib,json,os,shutil,subprocess,sys,time,uuid,threading
from pathlib import Path,PurePosixPath
from google.colab import drive,runtime

TR_RATE=6.77
TR_NEW_CAP=9.0
TR_PRIOR_CUMULATIVE=159.5  # conservative until final account reconciliation
assert TR_PRIOR_CUMULATIVE+TR_NEW_CAP<=200 and TR_OBSERVED_CREDIT>=TR_NEW_CAP
assert TR_NEW_CAP<=16-(TR_PRIOR_CUMULATIVE-148.92)+6
TR_PHASE_STARTED=time.monotonic()-60  # reserve60seconds for actual GPU provision
TR_PHASE_DEADLINE=TR_PHASE_STARTED+TR_NEW_CAP/TR_RATE*3600
TR_ROOT=Path('/content/sp_lense_work')/('qwen38_token_refresh_v1_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8])
TR_ROOT.mkdir(parents=True,exist_ok=False)
(TR_ROOT/'reports').mkdir()
TR_WATCHDOG=threading.Timer(max(1,600-(time.monotonic()-TR_PHASE_STARTED)),runtime.unassign)
TR_WATCHDOG.daemon=True;TR_WATCHDOG.start()
try:
 _payload=gzip.decompress(base64.b64decode(TR_PRIVATE_PAYLOAD_B64))
 assert hashlib.sha256(_payload).hexdigest()==TR_PRIVATE_PAYLOAD_SHA
 _files=json.loads(_payload)['files']
 for _name,_text in _files.items():
  _rel=PurePosixPath(_name);assert not _rel.is_absolute() and '..' not in _rel.parts and ':' not in _name and '\\' not in _name
  _target=TR_ROOT/_name;_target.parent.mkdir(parents=True,exist_ok=True);_target.write_bytes(_text.encode())
 drive.mount('/content/drive')
 _runs=Path('/content/drive/MyDrive/sp_lense/research3/runs')
 _packet=(_runs/'CURATED_COMPLETION_V3_CORRECTED_SOURCE_3db8c950.json.gz').read_bytes()
 assert hashlib.sha256(_packet).hexdigest()=='3db8c950b5645bafbd17aed7c519c9a76b401f91e41d754d0ab6fb60ba085249'
 (TR_ROOT/'SOURCE_PACKET.json.gz').write_bytes(_packet)
 TR_SOURCE=TR_ROOT/'source';TR_SOURCE.mkdir()
 for _name,_text in json.loads(gzip.decompress(_packet))['files'].items():
  _p=PurePosixPath(_name);assert not _p.is_absolute() and '..' not in _p.parts
  _dest=TR_SOURCE/_name;_dest.parent.mkdir(parents=True,exist_ok=True);_dest.write_bytes(_text.encode())
 _freeze=json.loads((TR_SOURCE/'FREEZE.json').read_bytes())
 for _name,_sha in _freeze['sha256'].items():assert hashlib.sha256((TR_SOURCE/_name).read_bytes()).hexdigest()==_sha
 _saved=_runs/'qwen38_completion_v3_20261004T175805Z_ad2d2b8d/fit_B/checkpoints/update_56'
 TR_ADAPTER=TR_ROOT/'inputs/B';TR_ADAPTER.mkdir()
 def _sha_file(path):
  _h=hashlib.sha256()
  with path.open('rb') as _f:
   for _b in iter(lambda:_f.read(16*1024**2),b''):_h.update(_b)
  return _h.hexdigest()
 for _name,_sha in [('adapter_model.safetensors','8a604a57f3666302b8684e2abe344252fc2f58a74b31fe694a60cd507d2ddfd5'),('adapter_config.json','8e81fd6cc0e22cf762caebf1a81105ee70ff284353a2bbacaf1fb5b4f3fb11fd')]:
  assert _sha_file(_saved/_name)==_sha
  shutil.copy2(_saved/_name,TR_ADAPTER/_name)
  assert _sha_file(TR_ADAPTER/_name)==_sha
 print('EXACT_COMPLETED_B_RESTORED',flush=True)
 drive.flush_and_unmount()
 subprocess.run([sys.executable,'-m','pip','install','--quiet','--no-input','-r',str(TR_SOURCE/'requirements.txt')],check=True,timeout=240)
 sys.path.insert(0,str(TR_ROOT/'code'))
 from setup_download_parallel import download_base
 download_base(TR_ROOT)
 assert time.monotonic()-TR_PHASE_STARTED<600,'Setup cap reached; no pilot launched'
 TR_CONFIG={'root':str(TR_ROOT),'source':str(TR_SOURCE),'adapter':str(TR_ADAPTER),'new_cap_units':TR_NEW_CAP,'prior_cumulative_conservative':TR_PRIOR_CUMULATIVE,'observed_credit_before_allocation':TR_OBSERVED_CREDIT,'rate':TR_RATE,'phase_started_monotonic':TR_PHASE_STARTED,'phase_deadline_monotonic':TR_PHASE_DEADLINE,'worker_max_seconds':3300,'export_seconds':600,'margin_units':.5,'optimizer_steps':0,'API_attempt_cap':2048,'API_success_cap':2048,'API_input_usd_cap':1.,'API_fresh_calls_no_cache':True}
 (TR_ROOT/'reports/SESSION_ADMISSION.json').write_text(json.dumps(TR_CONFIG,indent=2))
 assert TR_PHASE_DEADLINE-time.monotonic()>=3300+600+.5/TR_RATE*3600
 print('TOKEN_PILOT_SETUP_READY',json.dumps(TR_CONFIG),flush=True)
except Exception as exc:
 (TR_ROOT/'reports/SETUP_FAILURE.json').write_text(json.dumps({'error_type':type(exc).__name__,'model_updates':0,'pilot_started':False}))
 print('TOKEN_PILOT_SETUP_STOP',type(exc).__name__,flush=True)
 runtime.unassign()
 raise
finally:
 TR_WATCHDOG.cancel()

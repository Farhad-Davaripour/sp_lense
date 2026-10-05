# Trusted Colab setup only. These global inputs are filled by the private notebook.
import base64,gzip,hashlib,json,os,shutil,subprocess,sys,time,uuid,threading
from pathlib import Path,PurePosixPath
from google.colab import drive,runtime

TR_RATE=6.77  # conservative planning upper rate; actual observed rate recorded separately
TR_OBSERVED_RATE=float(TR_OBSERVED_RATE)
assert 0<TR_OBSERVED_RATE<=TR_RATE, 'Observed rate exceeds reviewed conservative planning upper bound'
assert 600+3300+600+.5/TR_RATE*3600<=9/TR_RATE*3600
TR_NEW_CAP=9.0
TR_PRIOR_CUMULATIVE=159.5  # conservative until final account reconciliation
assert TR_PRIOR_CUMULATIVE+TR_NEW_CAP<=200 and TR_OBSERVED_CREDIT>=TR_NEW_CAP
assert TR_NEW_CAP<=16-(TR_PRIOR_CUMULATIVE-148.92)+6
from datetime import datetime,timezone
assert TR_ALLOCATION_REQUEST_UTC, 'Actual allocation request timestamp required'
_tr_allocated=datetime.fromisoformat(TR_ALLOCATION_REQUEST_UTC.replace('Z','+00:00'))
_tr_elapsed=max(0.,(datetime.now(timezone.utc)-_tr_allocated).total_seconds())
TR_PHASE_STARTED=time.monotonic()-_tr_elapsed  # conservative accounting from actual allocation request
TR_PHASE_DEADLINE=TR_PHASE_STARTED+TR_NEW_CAP/TR_RATE*3600
# Independent end-to-end guard survives setup/launch/export gaps and app failure.
TR_HARD_PHASE_WATCHDOG=threading.Timer(max(1,TR_PHASE_DEADLINE-time.monotonic()),runtime.unassign)
TR_HARD_PHASE_WATCHDOG.daemon=True;TR_HARD_PHASE_WATCHDOG.start()
TR_ROOT=Path('/content/sp_lense_work')/('qwen38_sentence_control_v1_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8])
TR_ROOT.mkdir(parents=True,exist_ok=False)
(TR_ROOT/'reports').mkdir()
print('SENTENCE_RECOVERY_ROOT',str(TR_ROOT),flush=True)
TR_WATCHDOG=threading.Timer(max(1,600-(time.monotonic()-TR_PHASE_STARTED)),runtime.unassign)
TR_WATCHDOG.daemon=True;TR_WATCHDOG.start()
try:
 _gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True).strip()
 assert 'A100' in _gpu, 'Wrong allocated GPU type'
 print('SENTENCE_GPU_TYPE',_gpu,flush=True)
 _payload=gzip.decompress(base64.b64decode(TR_PRIVATE_PAYLOAD_B64))
 assert hashlib.sha256(_payload).hexdigest()==TR_PRIVATE_PAYLOAD_SHA
 _files=json.loads(_payload)['files']
 _new_freeze=json.loads(_files['EXPERIMENT_FREEZE.json'])
 for _name,_sha in _new_freeze['source_sha256'].items():
  if 'code/'+_name in _files:assert hashlib.sha256(_files['code/'+_name].encode()).hexdigest()==_sha
 for _name,_sha in _new_freeze['input_sha256'].items():assert hashlib.sha256(_files['inputs/'+_name].encode()).hexdigest()==_sha
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
 _saved=_runs/'qwen38_curated_v2_20261004T130200Z_c83530a1/inputs/models/A'
 TR_ADAPTER=TR_ROOT/'inputs/A';TR_ADAPTER.mkdir()
 def _sha_file(path):
  _h=hashlib.sha256()
  with path.open('rb') as _f:
   for _b in iter(lambda:_f.read(16*1024**2),b''):_h.update(_b)
  return _h.hexdigest()
 for _name,_sha in [('adapter_model.safetensors','8ae28fb51107a31419bcb744d9f47d80b681781a114f8bc9491751410b4b87aa'),('adapter_config.json','06a6ad4588f4f0d7b46cf9f7f67f9b7b5028d153134cdd0829a929e1caf9fcda')]:
  assert _sha_file(_saved/_name)==_sha
  shutil.copy2(_saved/_name,TR_ADAPTER/_name)
  assert _sha_file(TR_ADAPTER/_name)==_sha
 print('EXACT_PLAIN_A_RESTORED',flush=True)
 drive.flush_and_unmount()
 subprocess.run([sys.executable,'-m','pip','install','--quiet','--no-input','-r',str(TR_SOURCE/'requirements.txt')],check=True,timeout=240)
 sys.path.insert(0,str(TR_ROOT/'code'))
 from setup_download_parallel import download_base
 download_base(TR_ROOT)
 assert time.monotonic()-TR_PHASE_STARTED<600,'Setup cap reached; no pilot launched'
 TR_CONFIG={'root':str(TR_ROOT),'source':str(TR_SOURCE),'adapter':str(TR_ADAPTER),'new_cap_units':TR_NEW_CAP,'prior_cumulative_conservative':TR_PRIOR_CUMULATIVE,'observed_credit_before_allocation':TR_OBSERVED_CREDIT,'rate':TR_RATE,'observed_rate':TR_OBSERVED_RATE,'conservative_planning_rate':TR_RATE,'phase_started_monotonic':TR_PHASE_STARTED,'phase_deadline_monotonic':TR_PHASE_DEADLINE,'worker_max_seconds':3300,'export_seconds':600,'margin_units':.5,'optimizer_steps':0,'API_attempt_cap':2048,'API_success_cap':2048,'API_input_usd_cap':1.,'API_fresh_calls_no_cache':True}
 (TR_ROOT/'reports/SESSION_ADMISSION.json').write_text(json.dumps(TR_CONFIG,indent=2))
 assert TR_PHASE_DEADLINE-time.monotonic()>=3300+600+.5/TR_RATE*3600
 print('SENTENCE_PILOT_SETUP_READY',json.dumps(TR_CONFIG),flush=True)
except Exception as exc:
 (TR_ROOT/'reports/SETUP_FAILURE.json').write_text(json.dumps({'error_type':type(exc).__name__,'model_updates':0,'pilot_started':False}))
 print('SENTENCE_PILOT_SETUP_STOP',type(exc).__name__,flush=True)
 runtime.unassign()
 raise
finally:
 TR_WATCHDOG.cancel()

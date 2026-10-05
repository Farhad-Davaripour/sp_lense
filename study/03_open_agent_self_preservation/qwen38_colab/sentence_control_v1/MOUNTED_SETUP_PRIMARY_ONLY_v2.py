# SMALL mounted-Drive setup, after the root guard and74KB primary-only packet upload.
# No neural model/API calls here; frozen A/source/primary data are verified before launch.
assert 'TR_HARD_PHASE_WATCHDOG' in globals() and TR_HARD_PHASE_WATCHDOG.is_alive()
assert time.monotonic()-TR_PHASE_STARTED<600
try:
 _gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True).strip()
 assert 'A100' in _gpu
 print('MOUNTED_SETUP_STAGE','verify_primary_packet',flush=True)
 def _sha_file(path):
  _hash=hashlib.sha256()
  with path.open('rb') as _source:
   for _block in iter(lambda:_source.read(16*1024**2),b''):_hash.update(_block)
  return _hash.hexdigest()
 _uploaded=Path('/content/PRIVATE_SENTENCE_PRIMARY_ONLY_PACKET_v2.json.gz')
 assert _uploaded.stat().st_size==73884
 assert _sha_file(_uploaded)=='eeb8cbf61b248416fd59f93d46c09681987b08a99fd6bea8c4c1747811af961a'
 _raw=gzip.decompress(_uploaded.read_bytes())
 assert hashlib.sha256(_raw).hexdigest()=='ff3a354a8ca31fcd49e83c6cf524dbccf717a94c8f6725a446391b936afd2676'
 _files=json.loads(_raw)['files'];_frozen=json.loads(_files['EXPERIMENT_FREEZE.json'])
 assert _frozen['primary_only_required'] is True
 for _name,_pin in _frozen['source_sha256'].items():
  if 'code/'+_name in _files:assert hashlib.sha256(_files['code/'+_name].encode()).hexdigest()==_pin
 for _name,_pin in _frozen['input_sha256'].items():assert hashlib.sha256(_files['inputs/'+_name].encode()).hexdigest()==_pin
 for _name,_text in _files.items():
  _rel=PurePosixPath(_name);assert not _rel.is_absolute() and '..' not in _rel.parts and ':' not in _name and '\\' not in _name
  _dest=TR_ROOT/_name;_dest.parent.mkdir(parents=True,exist_ok=True);_dest.write_bytes(_text.encode())
 from google.colab import drive
 if not Path('/content/drive/MyDrive').is_dir():drive.mount('/content/drive')
 print('MOUNTED_SETUP_STAGE','restore_exact_A_and_source',flush=True)
 _runs=Path('/content/drive/MyDrive/sp_lense/research3/runs')
 _source_packet=_runs/'CURATED_COMPLETION_V3_CORRECTED_SOURCE_3db8c950.json.gz'
 assert _sha_file(_source_packet)=='3db8c950b5645bafbd17aed7c519c9a76b401f91e41d754d0ab6fb60ba085249'
 _packet=_source_packet.read_bytes();(TR_ROOT/'SOURCE_PACKET.json.gz').write_bytes(_packet)
 TR_SOURCE=TR_ROOT/'source';TR_SOURCE.mkdir()
 for _name,_text in json.loads(gzip.decompress(_packet))['files'].items():
  _rel=PurePosixPath(_name);assert not _rel.is_absolute() and '..' not in _rel.parts
  _dest=TR_SOURCE/_name;_dest.parent.mkdir(parents=True,exist_ok=True);_dest.write_bytes(_text.encode())
 _source_freeze=json.loads((TR_SOURCE/'FREEZE.json').read_bytes())
 for _name,_pin in _source_freeze['sha256'].items():assert _sha_file(TR_SOURCE/_name)==_pin
 _saved=_runs/'qwen38_curated_v2_20261004T130200Z_c83530a1/inputs/models/A'
 TR_ADAPTER=TR_ROOT/'inputs/A';TR_ADAPTER.mkdir()
 for _name,_pin in [('adapter_model.safetensors','8ae28fb51107a31419bcb744d9f47d80b681781a114f8bc9491751410b4b87aa'),('adapter_config.json','06a6ad4588f4f0d7b46cf9f7f67f9b7b5028d153134cdd0829a929e1caf9fcda')]:
  assert _sha_file(_saved/_name)==_pin
  shutil.copy2(_saved/_name,TR_ADAPTER/_name);assert _sha_file(TR_ADAPTER/_name)==_pin
 print('EXACT_PLAIN_A_RESTORED_PRIMARY_ONLY',flush=True)
 drive.flush_and_unmount()
 subprocess.run([sys.executable,'-m','pip','install','--quiet','--no-input','-r',str(TR_SOURCE/'requirements.txt')],check=True,timeout=240)
 sys.path.insert(0,str(TR_ROOT/'code'))
 from setup_download_parallel import download_base
 download_base(TR_ROOT)
 assert time.monotonic()-TR_PHASE_STARTED<600
 assert TR_PHASE_DEADLINE-time.monotonic()>=3300+600+.5/TR_RATE*3600
 TR_CONFIG={'root':str(TR_ROOT),'source':str(TR_SOURCE),'adapter':str(TR_ADAPTER),'new_cap_units':9.,
            'prior_cumulative_conservative':TR_PRIOR_CUMULATIVE,'observed_credit_before_allocation':TR_OBSERVED_CREDIT,
            'rate':TR_RATE,'observed_rate':TR_OBSERVED_RATE,'phase_started_monotonic':TR_PHASE_STARTED,
            'phase_deadline_monotonic':TR_PHASE_DEADLINE,'worker_max_seconds':3300,'export_seconds':600,'margin_units':.5,
            'optimizer_steps':0,'API_attempt_cap':2048,'API_success_cap':2048,'API_input_usd_cap':1.,
            'primary_only':True,'new_scientific_packet':'eeb8cbf61b248416fd59f93d46c09681987b08a99fd6bea8c4c1747811af961a',
            'original_scientific_packet_preserved':'a2b7a0748d7b4fabbfcb75d7b872f6e867d5f2eb3a4fa18d711b54caa99dae7f'}
 (TR_ROOT/'reports/SESSION_ADMISSION.json').write_text(json.dumps(TR_CONFIG,indent=2))
 print('SENTENCE_PILOT_SETUP_READY',json.dumps(TR_CONFIG,separators=(',',':')),flush=True)
except Exception as _error:
 _failure={'stage':'mounted_primary_setup','error_type':type(_error).__name__,'model_updates':0,'pilot_started':False}
 (TR_ROOT/'reports/SETUP_FAILURE.json').write_text(json.dumps(_failure))
 print('MOUNTED_SETUP_STOP',json.dumps(_failure),flush=True)
 runtime.unassign()
 raise
finally:
 TR_WATCHDOG.cancel()

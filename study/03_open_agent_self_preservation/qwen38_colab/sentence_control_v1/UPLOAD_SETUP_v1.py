# SMALL trusted setup from FilesUI uploads. No mount, auth, broker, credentials or B64.
assert 'TR_HARD_PHASE_WATCHDOG' in globals() and TR_HARD_PHASE_WATCHDOG.is_alive()
assert time.monotonic()-TR_PHASE_STARTED<TR_SETUP_MAX_SECONDS
try:
 _gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True).strip()
 assert 'A100' in _gpu
 print('UPLOAD_SETUP_STAGE','verify_uploaded_files',flush=True)
 _staged=Path('/content')
 def _sha_file(path):
  _hash=hashlib.sha256()
  with path.open('rb') as _source:
   for _block in iter(lambda:_source.read(16*1024**2),b''):_hash.update(_block)
  return _hash.hexdigest()
 _pins={'adapter_model.safetensors':(467062560,'8ae28fb51107a31419bcb744d9f47d80b681781a114f8bc9491751410b4b87aa'),
        'adapter_config.json':(1326,'06a6ad4588f4f0d7b46cf9f7f67f9b7b5028d153134cdd0829a929e1caf9fcda'),
        'SOURCE_PACKET.json.gz':(144509,'3db8c950b5645bafbd17aed7c519c9a76b401f91e41d754d0ab6fb60ba085249'),
        'PRIVATE_SENTENCE_PACKET_v1.json.gz':(73460,'a2b7a0748d7b4fabbfcb75d7b872f6e867d5f2eb3a4fa18d711b54caa99dae7f')}
 for _name,(_size,_pin) in _pins.items():
  assert (_staged/_name).stat().st_size==_size and _sha_file(_staged/_name)==_pin
 _files=json.loads(gzip.decompress((_staged/'PRIVATE_SENTENCE_PACKET_v1.json.gz').read_bytes()))['files']
 _freeze=json.loads(_files['EXPERIMENT_FREEZE.json'])
 for _name,_pin in _freeze['source_sha256'].items():
  if 'code/'+_name in _files:assert hashlib.sha256(_files['code/'+_name].encode()).hexdigest()==_pin
 for _name,_pin in _freeze['input_sha256'].items():assert hashlib.sha256(_files['inputs/'+_name].encode()).hexdigest()==_pin
 for _name,_text in _files.items():
  _rel=PurePosixPath(_name);assert not _rel.is_absolute() and '..' not in _rel.parts and ':' not in _name and '\\' not in _name
  _dest=TR_ROOT/_name;_dest.parent.mkdir(parents=True,exist_ok=True);_dest.write_bytes(_text.encode())
 _packet=(_staged/'SOURCE_PACKET.json.gz').read_bytes();(TR_ROOT/'SOURCE_PACKET.json.gz').write_bytes(_packet)
 TR_SOURCE=TR_ROOT/'source';TR_SOURCE.mkdir()
 for _name,_text in json.loads(gzip.decompress(_packet))['files'].items():
  _rel=PurePosixPath(_name);assert not _rel.is_absolute() and '..' not in _rel.parts
  _dest=TR_SOURCE/_name;_dest.parent.mkdir(parents=True,exist_ok=True);_dest.write_bytes(_text.encode())
 _source_freeze=json.loads((TR_SOURCE/'FREEZE.json').read_bytes())
 for _name,_pin in _source_freeze['sha256'].items():assert _sha_file(TR_SOURCE/_name)==_pin
 TR_ADAPTER=TR_ROOT/'inputs/A';TR_ADAPTER.mkdir()
 for _name in ['adapter_model.safetensors','adapter_config.json']:
  shutil.copy2(_staged/_name,TR_ADAPTER/_name);assert _sha_file(TR_ADAPTER/_name)==_pins[_name][1]
 print('EXACT_PLAIN_A_RESTORED_NO_MOUNT',flush=True)
 print('UPLOAD_SETUP_STAGE','pinned_dependencies_and_base',flush=True)
 subprocess.run([sys.executable,'-m','pip','install','--quiet','--no-input','-r',str(TR_SOURCE/'requirements.txt')],check=True,timeout=240)
 sys.path.insert(0,str(TR_ROOT/'code'))
 from setup_download_parallel import download_base
 download_base(TR_ROOT)
 assert time.monotonic()-TR_PHASE_STARTED<TR_SETUP_MAX_SECONDS
 assert TR_PHASE_DEADLINE-time.monotonic()>=3300+600+.5/TR_RATE*3600
 TR_CONFIG={'root':str(TR_ROOT),'source':str(TR_SOURCE),'adapter':str(TR_ADAPTER),'new_cap_units':TR_NEW_CAP,
            'prior_cumulative_conservative':TR_PRIOR_CUMULATIVE,'prior_restart_debits':TR_TOTAL_PREVIOUS_RESTART_DEBIT,
            'observed_credit_before_allocation':TR_OBSERVED_CREDIT,'rate':TR_RATE,'observed_rate':TR_OBSERVED_RATE,
            'phase_started_monotonic':TR_PHASE_STARTED,'phase_deadline_monotonic':TR_PHASE_DEADLINE,
            'setup_max_seconds':TR_SETUP_MAX_SECONDS,'worker_max_seconds':3300,'export_seconds':600,'margin_units':.5,
            'optimizer_steps':0,'API_attempt_cap':2048,'API_success_cap':2048,'API_input_usd_cap':1.,
            'no_Drive_mount':True,'exact_scientific_packet_unchanged':True}
 (TR_ROOT/'reports/SESSION_ADMISSION.json').write_text(json.dumps(TR_CONFIG,indent=2))
 print('SENTENCE_PILOT_SETUP_READY',json.dumps(TR_CONFIG,separators=(',',':')),flush=True)
except Exception as _error:
 _failure={'stage':'upload_setup','error_type':type(_error).__name__,'model_updates':0,'pilot_started':False}
 (TR_ROOT/'reports/SETUP_FAILURE.json').write_text(json.dumps(_failure))
 print('UPLOAD_SETUP_STOP',json.dumps(_failure),flush=True)
 runtime.unassign()
 raise
finally:
 TR_WATCHDOG.cancel()

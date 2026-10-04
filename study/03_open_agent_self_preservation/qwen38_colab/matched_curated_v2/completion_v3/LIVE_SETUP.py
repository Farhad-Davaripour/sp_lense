C3_SETUP_WATCHDOG=threading.Timer(max(1,600-(time.monotonic()-C3_PHASE_STARTED)),runtime.unassign)
C3_SETUP_WATCHDOG.daemon=True
C3_SETUP_WATCHDOG.start()
try:
 _runs=Path('/content/drive/MyDrive/sp_lense/research3/runs')
 _packets={'source':('CURATED_COMPLETION_V3_CORRECTED_SOURCE_3db8c950.json.gz','3db8c950b5645bafbd17aed7c519c9a76b401f91e41d754d0ab6fb60ba085249'),'control':('CURATED_COMPLETION_V3_CONTROL_03ab8364.json.gz','03ab83643df8ea46b5270e7380980a4b5ee94d7a1a1108503b405ed7f7222299')}
 for _kind,(_name,_sha) in _packets.items():
  _blob=(_runs/_name).read_bytes()
  assert hashlib.sha256(_blob).hexdigest()==_sha,_name
  _dest=C3_ROOT/_kind;_dest.mkdir()
  _files=json.loads(gzip.decompress(_blob))['files']
  for _rel,_text in _files.items():
   _p=PurePosixPath(_rel);assert not _p.is_absolute() and '..' not in _p.parts
   _target=_dest/_rel;_target.parent.mkdir(parents=True,exist_ok=True);_target.write_bytes(_text.encode())
  _freeze=json.loads((_dest/('FREEZE.json' if _kind=='source' else 'CONTROL_FREEZE.json')).read_text())
  for _rel,_sha in _freeze['sha256'].items():assert hashlib.sha256((_dest/_rel).read_bytes()).hexdigest()==_sha,_rel
  (C3_ROOT/(_kind.upper()+'_PACKET.json.gz')).write_bytes(_blob)
 C3_SOURCE=C3_ROOT/'source';C3_CONTROL=C3_ROOT/'control'
 _base=(_runs/'CURATED_COMPLETION_V3_A_BASELINE_538a7547.json.gz').read_bytes()
 assert hashlib.sha256(_base).hexdigest()=='538a7547de17ce007dca551a3433c8fb7c92305bc2dc55dae68da95efa35fb6f'
 (C3_ROOT/'A_BASELINE_PORTABLE.json.gz').write_bytes(_base)
 sys.path.insert(0,str(C3_SOURCE));sys.path.insert(0,str(C3_CONTROL))
 from restore_inputs import PINS,H2_RELATIVE,H2_FALLBACK,sha
 _h2_source=_runs/H2_RELATIVE
 if not _h2_source.is_dir():_h2_source=_runs/H2_FALLBACK
 _h2=C3_ROOT/'inputs/models/H2';_h2.mkdir(parents=True)
 for _file,_pin in (('adapter_model.safetensors','weights'),('adapter_config.json','config')):
  assert sha(_h2_source/_file)==PINS['H2'][_pin]
  shutil.copy2(_h2_source/_file,_h2/_file)
  assert sha(_h2/_file)==PINS['H2'][_pin]
 C3_H2={'path':str(_h2),'source':str(_h2_source),**PINS['H2']}
 # A is metadata-only: its complete verified run is reused; no A weights loaded.
 C3_A={'path':'verified_A_reused_without_loading','source':'prior eval_A_integrity_v2',**PINS['A']}
 (C3_ROOT/'RESTORED_INPUTS.json').write_text(json.dumps({'H2':C3_H2,'A':C3_A,'A_weight_copy_or_queries':0},indent=2))
 drive.flush_and_unmount()
 subprocess.run([sys.executable,'-m','pip','install','--quiet','--no-input','-r',str(C3_SOURCE/'requirements.txt')],check=True,timeout=240)
 from base_download import download_base
 C3_BASE_MANIFEST=download_base(C3_ROOT)
 assert time.monotonic()-C3_PHASE_STARTED<600,'Setup window exhausted; no optimizer admitted'
 C3_CONFIG={'source_dir':str(C3_SOURCE),'H2':C3_H2,'models':{'A':C3_A},'broker_ready':True,
  'A_REUSE_RECEIPT':{'path':str(C3_CONTROL/'A_REUSE_RECEIPT.json'),'sha256':'e5a3796296fa861ca5ac6e5196829880088027b2d3e1bee40ceaf1f93819df41'},
  'A_BASELINE_PACKET':{'path':str(C3_ROOT/'A_BASELINE_PORTABLE.json.gz'),'sha256':'538a7547de17ce007dca551a3433c8fb7c92305bc2dc55dae68da95efa35fb6f'},
  'phase':{'phase_started':C3_PHASE_STARTED,'cumulative_before':148.92,'phase_units':16,'rate':C3_RATE,'setup_limit':600,'fit_seconds':2700,'eval_seconds':3600,'export_seconds':900,'margin_units':1,'overall_units':200},
  'account_at_start':C3_ACCOUNT_START,'conservative_provision_seconds':60,'actual_Jev_transport_local_only':True}
 (C3_ROOT/'config.json').write_text(json.dumps(C3_CONFIG,indent=2))
 from admission import plan
 from preflight import verify
 C3_PLAN=plan(now=time.monotonic(),**C3_CONFIG['phase'])
 C3_PREFLIGHT=verify(C3_CONFIG,check_model_files=True)
 (C3_ROOT/'SESSION_ADMISSION.json').write_text(json.dumps(C3_PLAN,indent=2))
 print('COMPLETION_V3_READY',json.dumps({'root':str(C3_ROOT),'plan':C3_PLAN,'A_reuse':C3_PREFLIGHT['A_reuse']}),flush=True)
except Exception as exc:
 (C3_ROOT/'SETUP_FAILURE.json').write_text(json.dumps({'error_type':type(exc).__name__,'no_optimizer_admitted':True}))
 print('COMPLETION_V3_SETUP_STOP',type(exc).__name__,flush=True)
 runtime.unassign()
 raise
finally:
 C3_SETUP_WATCHDOG.cancel()

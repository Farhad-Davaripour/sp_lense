# Same GPU phase retry, after actual prior worker exit/integrity; no new allocation or budget reset.
assert TR_PROCESS.poll() is not None, 'Prior worker must exit first'
assert TR_HARD_PHASE_WATCHDOG.is_alive(), 'Original hard phase guard must remain armed'
TR_ATTEMPT1_ROOT=TR_ROOT
assert (TR_ATTEMPT1_ROOT/'reports/INCOMPLETE.json').is_file()
assert (TR_ATTEMPT1_ROOT/'reports/BASE_UNLOAD_INTEGRITY.json').is_file()
_integrity=json.loads((TR_ATTEMPT1_ROOT/'reports/BASE_UNLOAD_INTEGRITY.json').read_bytes())
assert _integrity["completed"] and all(_integrity[k] for k in ("post_generation_wrapped_pin_passed","restored_unloaded_pin_passed","same_canonical_parameter_names_shapes_dtypes_and_contents","stable_manifest_passed"))
assert _integrity["expected_historical_pin"]=="ec43e239f6ffad79d854149f9d7115aeda5292beddd411dd763cac8f9ddcd755"
(TR_ATTEMPT1_ROOT/"reports/WORKER_EXIT_CLOSEOUT.json").write_text(json.dumps({"worker_exit":TR_PROCESS.poll(),"optimizer_steps":0,"native_failed_trajectory_preserved":True,"all_integrity_checks_passed":True},indent=2))
TR_PRESERVED_PRIOR_ROOTS=[TR_ATTEMPT1_ROOT]
TR_WORKER_WATCH.cancel()
_uploaded=Path('/content/PRIVATE_SENTENCE_DIAGNOSTIC_PACKET_v3.json.gz')
assert _uploaded.stat().st_size==74768
assert _sha_file(_uploaded)=='c65ea99c0e4ff977093cda6a16de48c4b4933f3cd77be32697fa907e48cc7d75'
_raw=gzip.decompress(_uploaded.read_bytes());assert hashlib.sha256(_raw).hexdigest()=='1ea0facfdff3f618f1532ebe036cf0a8f1d26fa6730ceb5c90fbe66793e3b929'
_files=json.loads(_raw)['files'];_freeze=json.loads(_files['EXPERIMENT_FREEZE.json'])
assert _freeze['primary_only_required'] and _freeze['predictive_admission_diagnostic_only_required']
for _name,_pin in _freeze['source_sha256'].items():
 if 'code/'+_name in _files:assert hashlib.sha256(_files['code/'+_name].encode()).hexdigest()==_pin
for _name,_pin in _freeze['input_sha256'].items():assert hashlib.sha256(_files['inputs/'+_name].encode()).hexdigest()==_pin
TR_ROOT=Path('/content/sp_lense_work')/('qwen38_sentence_diagnostic_v3_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8])
(TR_ROOT/'reports').mkdir(parents=True,exist_ok=False)
for _name,_text in _files.items():
 _rel=PurePosixPath(_name);assert not _rel.is_absolute() and '..' not in _rel.parts and ':' not in _name and '\\' not in _name
 _dest=TR_ROOT/_name;_dest.parent.mkdir(parents=True,exist_ok=True);_dest.write_bytes(_text.encode())
TR_SOURCE=TR_ROOT/'source';shutil.copytree(TR_ATTEMPT1_ROOT/'source',TR_SOURCE)
shutil.copy2(TR_ATTEMPT1_ROOT/'SOURCE_PACKET.json.gz',TR_ROOT/'SOURCE_PACKET.json.gz')
TR_ADAPTER=TR_ROOT/'inputs/A';shutil.copytree(TR_ATTEMPT1_ROOT/'inputs/A',TR_ADAPTER)
assert _sha_file(TR_ADAPTER/'adapter_model.safetensors')=='8ae28fb51107a31419bcb744d9f47d80b681781a114f8bc9491751410b4b87aa'
assert _sha_file(TR_ADAPTER/'adapter_config.json')=='06a6ad4588f4f0d7b46cf9f7f67f9b7b5028d153134cdd0829a929e1caf9fcda'
TR_SHARED_MODEL=TR_ATTEMPT1_ROOT/'model'
assert TR_SHARED_MODEL.is_dir()
(TR_ROOT/'reports/SESSION_ADMISSION.json').write_text(json.dumps(dict(TR_CONFIG,root=str(TR_ROOT),source=str(TR_SOURCE),adapter=str(TR_ADAPTER),same_GPU_phase=True,original_phase_clock_unchanged=True,predictive_admission_diagnostic_only=True,prior_attempt_root=str(TR_ATTEMPT1_ROOT),shared_base=str(TR_SHARED_MODEL),prior8actualAPIcalls_carried=True),indent=2))
print('SAME_PHASE_DIAGNOSTIC_RETRY_READY',json.dumps({'root':str(TR_ROOT),'prior_attempt_root':str(TR_ATTEMPT1_ROOT),'original_allocation_UTC':TR_ALLOCATION_REQUEST_UTC,'remaining_phase_seconds':TR_PHASE_DEADLINE-time.monotonic(),'phase_cap':TR_NEW_CAP,'predictive_only':True},separators=(',',':')),flush=True)

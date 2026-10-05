# Same GPU phase retry, after actual prior worker exit/integrity; no new allocation or budget reset.
assert TR_PROCESS.poll() is not None, 'Prior worker must exit first'
assert TR_HARD_PHASE_WATCHDOG.is_alive(), 'Original hard phase guard must remain armed'
TR_ATTEMPT2_ROOT=TR_ROOT
assert (TR_ATTEMPT2_ROOT/'reports/INCOMPLETE.json').is_file()
assert (TR_ATTEMPT2_ROOT/'reports/BASE_UNLOAD_INTEGRITY.json').is_file()
_integrity=json.loads((TR_ATTEMPT2_ROOT/'reports/BASE_UNLOAD_INTEGRITY.json').read_bytes())
assert _integrity["completed"] and all(_integrity[k] for k in ("post_generation_wrapped_pin_passed","restored_unloaded_pin_passed","same_canonical_parameter_names_shapes_dtypes_and_contents","stable_manifest_passed"))
assert _integrity["expected_historical_pin"]=="ec43e239f6ffad79d854149f9d7115aeda5292beddd411dd763cac8f9ddcd755"
(TR_ATTEMPT2_ROOT/"reports/WORKER_EXIT_CLOSEOUT.json").write_text(json.dumps({"worker_exit":TR_PROCESS.poll(),"optimizer_steps":0,"native_failed_trajectory_preserved":True,"all_integrity_checks_passed":True},indent=2))
TR_PRESERVED_PRIOR_ROOTS=[TR_ATTEMPT1_ROOT,TR_ATTEMPT2_ROOT]
TR_COMPLETION_PRIOR_ROOT=TR_ATTEMPT2_ROOT
assert (TR_ATTEMPT2_ROOT/'episodes/s72111_ordered_ticket_51_self_unfinished_control.json').is_file()
assert (TR_ATTEMPT2_ROOT/'reports/PAIRED_PROGRESS.json').is_file()
TR_WORKER_WATCH.cancel()
_uploaded=Path('/content/PRIVATE_SENTENCE_COMPLETION_PACKET_v6.json.gz')
assert _uploaded.stat().st_size==75846
assert _sha_file(_uploaded)=='372f6c80d449d82d16eb6ffc6b55696796d8d42998a610ef51f6a86224ac63fa'
_raw=gzip.decompress(_uploaded.read_bytes());assert hashlib.sha256(_raw).hexdigest()=='30044098c8d781a890604392bfee98e453136a19ee6793c55e09c06f3a0652ef'
_files=json.loads(_raw)['files'];_freeze=json.loads(_files['EXPERIMENT_FREEZE.json'])
assert _freeze['primary_only_required'] and _freeze['predictive_admission_diagnostic_only_required']
for _name,_pin in _freeze['source_sha256'].items():
 if 'code/'+_name in _files:assert hashlib.sha256(_files['code/'+_name].encode()).hexdigest()==_pin
for _name,_pin in _freeze['input_sha256'].items():assert hashlib.sha256(_files['inputs/'+_name].encode()).hexdigest()==_pin
TR_ROOT=Path('/content/sp_lense_work')/('qwen38_sentence_complete_v6_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8])
(TR_ROOT/'reports').mkdir(parents=True,exist_ok=False)
for _name,_text in _files.items():
 _rel=PurePosixPath(_name);assert not _rel.is_absolute() and '..' not in _rel.parts and ':' not in _name and '\\' not in _name
 _dest=TR_ROOT/_name;_dest.parent.mkdir(parents=True,exist_ok=True);_dest.write_bytes(_text.encode())
TR_SOURCE=TR_ROOT/'source';shutil.copytree(TR_ATTEMPT2_ROOT/'source',TR_SOURCE)
shutil.copy2(TR_ATTEMPT2_ROOT/'SOURCE_PACKET.json.gz',TR_ROOT/'SOURCE_PACKET.json.gz')
TR_ADAPTER=TR_ROOT/'inputs/A';shutil.copytree(TR_ATTEMPT2_ROOT/'inputs/A',TR_ADAPTER)
assert _sha_file(TR_ADAPTER/'adapter_model.safetensors')=='8ae28fb51107a31419bcb744d9f47d80b681781a114f8bc9491751410b4b87aa'
assert _sha_file(TR_ADAPTER/'adapter_config.json')=='06a6ad4588f4f0d7b46cf9f7f67f9b7b5028d153134cdd0829a929e1caf9fcda'
# TR_SHARED_MODEL already points to the original immutable model directory; keep it.
assert TR_SHARED_MODEL.is_dir()
(TR_ROOT/'reports/SESSION_ADMISSION.json').write_text(json.dumps(dict(TR_CONFIG,root=str(TR_ROOT),source=str(TR_SOURCE),adapter=str(TR_ADAPTER),same_GPU_phase=True,original_phase_clock_unchanged=True,predictive_admission_diagnostic_only=True,prior_attempt_root=str(TR_ATTEMPT2_ROOT),completion_prior_root=str(TR_COMPLETION_PRIOR_ROOT),completed_pair1_and_case2_control_reused=True,shared_base=str(TR_SHARED_MODEL),all52prioractualAPIcalls_carried=True),indent=2))
print('SAME_PHASE_DIAGNOSTIC_RETRY_READY',json.dumps({'root':str(TR_ROOT),'prior_attempt_root':str(TR_ATTEMPT2_ROOT),'completion_mode':'case2_treatment_only_with_both_prior_roots_preserved','original_allocation_UTC':TR_ALLOCATION_REQUEST_UTC,'remaining_phase_seconds':TR_PHASE_DEADLINE-time.monotonic(),'phase_cap':TR_NEW_CAP,'predictive_only':True},separators=(',',':')),flush=True)

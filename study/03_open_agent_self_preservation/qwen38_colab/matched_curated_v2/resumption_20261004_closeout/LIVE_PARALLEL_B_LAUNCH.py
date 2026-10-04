import copy,shutil,subprocess,sys,json,time
_v2_fit_root=V2_ROOT/'fit_B_parallel'
assert not _v2_fit_root.exists()
_v2_fit_root.mkdir()
shutil.copytree(V2_ROOT/'source_resume',_v2_fit_root/'source')
_v2_cfg=json.loads((V2_ROOT/'RESUME_CONFIG.json').read_text())
_v2_fit_cfg={'H2':_v2_cfg['H2'],'training_recipe':'matched_curated_B_v2','training_mode':'curated_B','training_data_dir':str(V2_ROOT/'source_resume/training_data')}
(_v2_fit_root/'config.json').write_text(json.dumps(_v2_fit_cfg,indent=2))
_v2_budget=6879.4235699905-1500-time.monotonic()
assert _v2_budget>600,'Insufficient bounded fit allowance'
_v2_driver_ns={}
exec(compile((V2_ROOT/'source_resume/driver_resume_v1.py').read_text(),'driver_resume_v1.py','exec'),_v2_driver_ns)
_v2_log=V2_ROOT/'training/logs/parallel_B_train.log'
_v2_log.parent.mkdir(parents=True,exist_ok=True)
with _v2_log.open('wb') as _v2_out:
 V2_B_PROCESS=subprocess.Popen([sys.executable,str(V2_ROOT/'source_resume/training_worker.py'),'--root',str(_v2_fit_root),'--model',str(V2_ROOT/'model'),'--max-seconds',str(_v2_budget)],stdout=_v2_out,stderr=subprocess.STDOUT,env=_v2_driver_ns['model_env'](),start_new_session=True)
_v2_b_receipt={'pid':V2_B_PROCESS.pid,'root':str(_v2_fit_root),'log':str(_v2_log),'started_monotonic':time.monotonic(),'max_seconds':_v2_budget,'deadline_monotonic':6879.4235699905-1500,'required_updates':56,'evaluation_reserve_seconds':1500,'maximum_resident_model_workers':2,'model_start':'preserved H2','user_authorized_parallel_schedule':True}
(V2_ROOT/'reports/PARALLEL_B_PID.json').write_text(json.dumps(_v2_b_receipt,indent=2))
print('PARALLEL_B_FIT_STARTED',json.dumps(_v2_b_receipt),flush=True)

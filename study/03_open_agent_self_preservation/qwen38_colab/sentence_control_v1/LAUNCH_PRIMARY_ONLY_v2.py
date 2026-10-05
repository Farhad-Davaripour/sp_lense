assert (TR_ROOT/'reports/SESSION_ADMISSION.json').is_file()
assert TR_PHASE_DEADLINE-time.monotonic()>=3300+600+.5/TR_RATE*3600
_env={k:v for k,v in os.environ.items() if not any(s in k.upper() for s in ('TYPESAFE','JEV','API_KEY','TOKEN','SECRET','PASSWORD','CREDENTIAL'))}
_env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='0',PYTHONNOUSERSITE='1')
(TR_ROOT/'logs').mkdir(exist_ok=True)
TR_LOG=(TR_ROOT/'logs/worker.log').open('wb')
TR_PROCESS=subprocess.Popen([sys.executable,str(TR_ROOT/'code/worker.py'),'--root',str(TR_ROOT),'--source',str(TR_SOURCE),'--model',str(TR_ROOT/'model'),'--adapter',str(TR_ADAPTER),'--max-seconds','3300','--primary-only'],stdout=TR_LOG,stderr=subprocess.STDOUT,env=_env,start_new_session=True)
(TR_ROOT/'reports/WORKER_PID.json').write_text(json.dumps({'pid':TR_PROCESS.pid,'started_monotonic':time.monotonic(),'only_model_worker':True}))
def _tr_stop():
 if TR_PROCESS.poll() is None:TR_PROCESS.terminate()
TR_WORKER_WATCH=threading.Timer(3330,_tr_stop);TR_WORKER_WATCH.daemon=True;TR_WORKER_WATCH.start()
print('SENTENCE_PILOT_WORKER_STARTED',TR_PROCESS.pid,flush=True)

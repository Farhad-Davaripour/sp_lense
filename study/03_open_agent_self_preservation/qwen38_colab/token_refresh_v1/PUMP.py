import base64,gzip,re
TR_RESPONSE_B64=''
if TR_RESPONSE_B64:
 _blob=gzip.decompress(base64.b64decode(TR_RESPONSE_B64));_r=json.loads(_blob);_batch=_r['batch_id']
 assert re.fullmatch(r'eval_[0-9]{5}_[a-f0-9]{10}',_batch)
 _p=TR_ROOT/'service_pending'/(_batch+'.json.gz');assert _p.is_file()
 _request=json.loads(gzip.decompress(_p.read_bytes()))
 assert {r['id'] for r in _request['rows']}=={r['id'] for r in _r['rows']}
 _dest=TR_ROOT/'service_responses'/(_batch+'.json');assert not _dest.exists()
 _tmp=_dest.with_suffix('.partial');_tmp.write_bytes(_blob);os.replace(_tmp,_dest)
 print('TR_RESPONSE_APPLIED',_batch,flush=True)
print('TR_WORKER_EXIT',TR_PROCESS.poll(),flush=True)
_p=TR_ROOT/'logs/worker.log'
if _p.exists():print('TR_LOG',_p.read_text(errors='replace')[-1600:],flush=True)
for _name in ['MODEL_READY.json','CONTROL_MECHANICS_VALIDATION.json','TOKEN_LATENCY_ADMISSION.json','PAIRED_PROGRESS.json','RESULT.json','INCOMPLETE.json']:
 _p=TR_ROOT/'reports'/_name
 if _p.exists():print('TR_STATUS',_name,_p.read_text(),flush=True)
for _p in sorted((TR_ROOT/'service_pending').glob('*.json.gz')):
 if not (TR_ROOT/'service_responses'/(_p.name.removesuffix('.gz'))).exists():
  print('TR_PENDING_GZIP_BASE64',base64.b64encode(_p.read_bytes()).decode(),flush=True);break
print('TR_PUMP_COMPLETE',flush=True)

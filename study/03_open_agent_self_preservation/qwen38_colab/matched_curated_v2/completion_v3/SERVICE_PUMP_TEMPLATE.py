import base64,gzip,re,os
C3_RESPONSE_B64=''
C3_SERVICE_ROOT=C3_ROOT/'eval_B'
if C3_RESPONSE_B64:
 _blob=gzip.decompress(base64.b64decode(C3_RESPONSE_B64));_response=json.loads(_blob);_batch=_response['batch_id']
 assert re.fullmatch(r'eval_[0-9]{5}_[a-f0-9]{10}',_batch)
 _pending=C3_SERVICE_ROOT/'service_pending'/(_batch+'.json.gz');assert _pending.is_file()
 _request=json.loads(gzip.decompress(_pending.read_bytes()))
 assert {r['id'] for r in _request['rows']}=={r['id'] for r in _response['rows']}
 _dest=C3_SERVICE_ROOT/'service_responses'/(_batch+'.json');assert not _dest.exists()
 _temp=_dest.with_suffix('.partial');_temp.write_bytes(_blob);os.replace(_temp,_dest)
 print('C3_RESPONSE_APPLIED',_batch,flush=True)
for _name in ('PROGRESS.json','RESULT.json','FAILURE.json'):
 _path=C3_ROOT/'reports'/_name
 if _path.exists(): print('C3_STATUS',_name,_path.read_text(),flush=True)
for _name in ('B_fit','B_eval'):
 _path=C3_ROOT/'training/logs'/(_name+'.log')
 if _path.exists(): print('C3_LOG',_name,_path.read_text(errors='replace')[-1800:],flush=True)
_path=C3_ROOT/'fit_B/reports/LIVE_TRAINING.json'
if _path.exists():print('C3_UPDATES',_path.read_text(),flush=True)
for _p in sorted((C3_SERVICE_ROOT/'service_pending').glob('*.json.gz')):
 if not (C3_SERVICE_ROOT/'service_responses'/(_p.name.removesuffix('.gz'))).exists():
  print('C3_PENDING_GZIP_BASE64',base64.b64encode(_p.read_bytes()).decode(),flush=True);break
print('C3_PUMP_COMPLETE',flush=True)

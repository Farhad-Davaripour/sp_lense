import base64,gzip,re,os
V2_RESPONSE_B64=''
V2_SERVICE_ROOT=V2_ROOT/'eval_B'
if V2_RESPONSE_B64:
    _blob=gzip.decompress(base64.b64decode(V2_RESPONSE_B64))
    _response=json.loads(_blob)
    _batch=_response['batch_id']
    assert re.fullmatch(r'eval_[0-9]{5}_[a-f0-9]{10}',_batch)
    _pending=V2_SERVICE_ROOT/'service_pending'/(_batch+'.json.gz')
    assert _pending.is_file()
    _request=json.loads(gzip.decompress(_pending.read_bytes()))
    assert {r['id'] for r in _request['rows']}=={r['id'] for r in _response['rows']}
    _destination=V2_SERVICE_ROOT/'service_responses'/(_batch+'.json')
    _destination.parent.mkdir(parents=True,exist_ok=True)
    assert not _destination.exists()
    _temp=_destination.with_suffix('.partial')
    _temp.write_bytes(_blob)
    os.replace(_temp,_destination)
    print('CURRENT_JEV_RESPONSE_APPLIED',_batch,flush=True)
for _name in ('DRIVER_STATUS.json','DRIVER_RESULT.json','DRIVER_FAILURE.json'):
    _path=V2_ROOT/'reports'/_name
    if _path.is_file(): print(_name,_path.read_text(),flush=True)
for _name in ('A_eval','B_train','B_eval'):
    _log=V2_ROOT/'training/logs'/(_name+'.log')
    if _log.is_file(): print('LIVE_STAGE_LOG',_name,_log.read_text(errors='replace')[-2200:],flush=True)
if (V2_SERVICE_ROOT/'service_pending').is_dir():
    for _pending in sorted((V2_SERVICE_ROOT/'service_pending').glob('*.json.gz')):
        if not (V2_SERVICE_ROOT/'service_responses'/(_pending.name.removesuffix('.gz'))).exists():
            print('CURRENT_JEV_PENDING_GZIP_BASE64',base64.b64encode(_pending.read_bytes()).decode(),flush=True)
            break
print('V2_SERVICE_PUMP_COMPLETE',flush=True)

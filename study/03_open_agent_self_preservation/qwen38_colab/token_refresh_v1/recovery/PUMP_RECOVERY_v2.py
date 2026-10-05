# Trusted notebook transport only. No credentials, binary stdout, or raw API output.
import base64
import gzip
import re

TR_RESPONSE_B64 = ''
if TR_RESPONSE_B64:
    _blob = gzip.decompress(base64.b64decode(TR_RESPONSE_B64))
    _response = json.loads(_blob)
    _batch = _response['batch_id']
    assert re.fullmatch(r'eval_[0-9]{5}_[a-f0-9]{10}', _batch)
    _pending = TR_ROOT / 'service_pending' / (_batch + '.json.gz')
    assert _pending.is_file()
    _request = json.loads(gzip.decompress(_pending.read_bytes()))
    assert {row['id'] for row in _request['rows']} == {row['id'] for row in _response['rows']}
    _destination = TR_ROOT / 'service_responses' / (_batch + '.json')
    assert not _destination.exists()
    _temporary = _destination.with_suffix('.partial')
    _temporary.write_bytes(_blob)
    os.replace(_temporary, _destination)
    print('TR_RESPONSE_APPLIED', _batch, flush=True)

_summary = {'worker_exit': TR_PROCESS.poll(), 'actual_model_execution_confirmed': False,
            'generated_tokens': 0, 'cases_completed': 0, 'model_loaded_confirmed': False,
            'treatment_generated_tokens': 0, 'tokens_with_successful_real_refresh': 0}
_model = TR_ROOT / 'reports/MODEL_READY.json'
if _model.is_file():
    _value = json.loads(_model.read_bytes())
    _summary.update(model_loaded_confirmed=True, weights_sha256=_value['weights'],
                    all_parameters_frozen=_value['all_parameters_frozen'], optimizer_steps=0)
_validation = TR_ROOT / 'reports/CONTROL_MECHANICS_VALIDATION.json'
if _validation.is_file():
    _summary['control_validation'] = [
        {'case': row['case_id'], 'phase': row['phase'],
         'ordinary_match': row['stepped_matches_ordinary_batch1'],
         'archived_match': row['stepped_matches_archived_batch4']}
        for row in json.loads(_validation.read_bytes())]
_admission = TR_ROOT / 'reports/TOKEN_LATENCY_ADMISSION.json'
if _admission.is_file():
    _value = json.loads(_admission.read_bytes())
    _summary['token_latency_admission'] = {key: _value[key] for key in (
        'sample_tokens', 'conservative_seconds_per_token', 'archived_two_case_tokens',
        'projected_remaining_seconds', 'remaining_worker_seconds', 'admitted')}
for _name in ('RESULT.json', 'INCOMPLETE.json'):
    _path = TR_ROOT / 'reports' / _name
    if _path.is_file():
        _value = json.loads(_path.read_bytes())
        _summary['result_type'] = _name
        _summary['completed'] = _value['completed']
        _summary['real_annotation_requests'] = _value['actual_annotation_requests']
        _summary['reason'] = _value.get('reason')
        _summary['paired_case_metrics'] = _value.get('paired_cases', _value.get('paired_cases_completed'))
_turn_counts = {}
for _path in (TR_ROOT / 'tokens').glob('turn_*.json'):
    _match = re.fullmatch(r'turn_([0-9]{5})(?:_INCOMPLETE)?\.json', _path.name)
    if not _match:
        continue
    _value = json.loads(_path.read_bytes())
    _counts = {'tokens': len(_value.get('token_ids', [])),
               'treatment': bool(_value.get('treatment')),
               'refreshes': sum(bool(row.get('real_refresh_after_this_token'))
                                for row in _value.get('token_records', []))}
    _key = _match.group(1)
    if _key not in _turn_counts or _counts['tokens'] > _turn_counts[_key]['tokens']:
        _turn_counts[_key] = _counts
_summary['generated_tokens'] = sum(row['tokens'] for row in _turn_counts.values())
_summary['treatment_generated_tokens'] = sum(row['tokens'] for row in _turn_counts.values() if row['treatment'])
_summary['tokens_with_successful_real_refresh'] = sum(row['refreshes'] for row in _turn_counts.values())
_summary['actual_model_execution_confirmed'] = _summary['generated_tokens'] > 0
_summary['cases_completed'] = len(_summary.get('paired_case_metrics') or [])
print('TR_COMPACT_STATUS', json.dumps(_summary, separators=(',', ':')), flush=True)
for _path in sorted((TR_ROOT / 'service_pending').glob('*.json.gz')):
    if not (TR_ROOT / 'service_responses' / _path.name.removesuffix('.gz')).exists():
        # Root captures this visible JSON directly into a C: file without emitting it in chat.
        print('TR_PENDING_JSON', gzip.decompress(_path.read_bytes()).decode('utf-8'), flush=True)
        break
print('TR_PUMP_COMPLETE', flush=True)

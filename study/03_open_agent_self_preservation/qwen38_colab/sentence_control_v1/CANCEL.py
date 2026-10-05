# Trusted operator cancellation for real service/admission faults; no retry or injection.
from datetime import datetime, timezone
_cancel = {'requested_utc': datetime.now(timezone.utc).isoformat(),
           'reason': 'trusted_controller_stop_after_fault_or_limit', 'automatic_retry': False}
_path = TR_ROOT / 'reports/CANCEL_REQUESTED.json'
_temp = _path.with_suffix('.partial')
_temp.write_text(json.dumps(_cancel, indent=2))
os.replace(_temp, _path)
print('SENTENCE_CANCEL_REQUESTED', flush=True)
_grace_deadline = time.monotonic() + 90
while TR_PROCESS.poll() is None and time.monotonic() < _grace_deadline:
    time.sleep(.5)
_fallback = False
if TR_PROCESS.poll() is None:
    _fallback = True
    TR_PROCESS.terminate()
    try:
        TR_PROCESS.wait(timeout=15)
    except subprocess.TimeoutExpired:
        TR_PROCESS.kill()
        TR_PROCESS.wait(timeout=10)
_cancel.update(worker_exit=TR_PROCESS.poll(), bounded_termination_fallback=_fallback)
(TR_ROOT / 'reports/CANCEL_CLOSEOUT.json').write_text(json.dumps(_cancel, indent=2))
print('SENTENCE_CANCEL_EXIT', json.dumps(_cancel, separators=(',', ':')), flush=True)
# Export all partial records promptly; the hard phase watchdog remains armed.

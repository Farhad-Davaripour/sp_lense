# SMALL trusted GPU clock/guards, run immediately after fresh allocation BEFORE uploads.
import gzip,hashlib,json,os,shutil,subprocess,sys,time,uuid,threading
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
from google.colab import runtime

TR_RATE=6.77
TR_NEW_CAP=10.5
TR_SETUP_MAX_SECONDS=1200
assert 0<float(TR_OBSERVED_RATE)<=TR_RATE
assert TR_TOTAL_PREVIOUS_RESTART_DEBIT+TR_NEW_CAP<=11.7
assert TR_PRIOR_CUMULATIVE+TR_NEW_CAP<=200 and TR_OBSERVED_CREDIT>=TR_NEW_CAP
assert TR_NEW_CAP<=16-(TR_PRIOR_CUMULATIVE-148.92)+6
assert 1200+3300+600+.5/TR_RATE*3600<=TR_NEW_CAP/TR_RATE*3600
_elapsed=max(0.,(datetime.now(timezone.utc)-datetime.fromisoformat(TR_ALLOCATION_REQUEST_UTC.replace('Z','+00:00'))).total_seconds())
TR_PHASE_STARTED=time.monotonic()-_elapsed
TR_PHASE_DEADLINE=TR_PHASE_STARTED+TR_NEW_CAP/TR_RATE*3600
TR_HARD_PHASE_WATCHDOG=threading.Timer(max(1,TR_PHASE_DEADLINE-time.monotonic()),runtime.unassign)
TR_HARD_PHASE_WATCHDOG.daemon=True;TR_HARD_PHASE_WATCHDOG.start()
TR_ROOT=Path('/content/sp_lense_work')/('qwen38_sentence_upload_v1_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8])
(TR_ROOT/'reports').mkdir(parents=True,exist_ok=False)
TR_WATCHDOG=threading.Timer(max(1,1200-(time.monotonic()-TR_PHASE_STARTED)),runtime.unassign)
TR_WATCHDOG.daemon=True;TR_WATCHDOG.start()
_armed={'root':str(TR_ROOT),'allocation_request_UTC':TR_ALLOCATION_REQUEST_UTC,'conservative_rate':TR_RATE,
        'observed_rate':TR_OBSERVED_RATE,'new_attempt_cap_units':TR_NEW_CAP,
        'all_prior_restart_debits_included':TR_TOTAL_PREVIOUS_RESTART_DEBIT,
        'total_planned_restart_envelope_units':TR_TOTAL_PREVIOUS_RESTART_DEBIT+TR_NEW_CAP,
        'approved_combined_remaining_from_main_before_restarts':11.7,'setup_max_seconds':1200,
        'worker_max_seconds':3300,'export_seconds':600,'margin_units':.5,
        'model_or_Jev_started':False,'same_scientific_packet':'a2b7a0748d7b4fabbfcb75d7b872f6e867d5f2eb3a4fa18d711b54caa99dae7f'}
(TR_ROOT/'reports/UPLOAD_SESSION_ARMED.json').write_text(json.dumps(_armed,indent=2))
print('UPLOAD_GPU_GUARDS_ARMED',json.dumps(_armed,separators=(',',':')),flush=True)

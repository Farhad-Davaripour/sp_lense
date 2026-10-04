import time,json,uuid,threading,subprocess,sys,hashlib,gzip,shutil,os
from pathlib import Path,PurePosixPath
from google.colab import drive,runtime
C3_PHASE_STARTED=time.monotonic()-60  # current allocation only; prior cost separate
C3_PRIOR_PHASE_UNITS_BOUND=0.90
C3_ACCOUNT_START=53.21
C3_RATE=6.77
C3_ROOT=Path('/content/sp_lense_work')/('qwen38_completion_v3_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8])
C3_ROOT.mkdir(parents=True,exist_ok=False)
C3_MOUNT_WATCHDOG=threading.Timer(120,runtime.unassign)
C3_MOUNT_WATCHDOG.daemon=True
C3_MOUNT_WATCHDOG.start()
try:
 drive.mount('/content/drive')
 print('COMPLETION_V3_MOUNT_READY',str(C3_ROOT),flush=True)
except Exception as exc:
 print('COMPLETION_V3_SETUP_STOP',type(exc).__name__,flush=True)
 runtime.unassign()
 raise
finally:
 C3_MOUNT_WATCHDOG.cancel()

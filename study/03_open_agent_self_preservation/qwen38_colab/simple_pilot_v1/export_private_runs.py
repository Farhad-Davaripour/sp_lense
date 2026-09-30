"""Trusted notebook export after all model processes exit. No model tool access."""
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

RUN_ROOTS = [ROOT, FAST_ROOT] + ([CONCURRENT_ROOT] if 'CONCURRENT_ROOT' in globals() else [])
for source_root in RUN_ROOTS:
    for worker_path in source_root.rglob('worker.py'):
        process = subprocess.run(['pgrep','-f',re.escape(str(worker_path))],
                                 capture_output=True,text=True)
        if process.stdout.strip():
            raise RuntimeError('Wait for model workers to exit before mounting Drive')


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(16*1024**2),b''):
            digest.update(block)
    return digest.hexdigest()


exports = {}
for source_root in RUN_ROOTS:
    files = []
    for path in source_root.rglob('*'):
        relative = path.relative_to(source_root)
        if any(part in ('model','worker_home','__pycache__') for part in relative.parts):
            continue
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise RuntimeError('Unexpected export symlink: ' + str(relative))
        if path.is_dir():
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 2*1024**3:
            raise RuntimeError('Invalid export file: ' + str(relative))
        files.append((path,relative,info.st_size))
    exports[source_root] = files

from google.colab import drive
drive.mount('/content/drive')
parent = Path('/content/drive/MyDrive/sp_lense/research3/runs')
parent.mkdir(parents=True,exist_ok=True)
required = sum(size for files in exports.values() for _,_,size in files)
if shutil.disk_usage(parent).free < required + 1024**3:
    raise RuntimeError('Insufficient available Drive space')
for source_root, files in exports.items():
    destination = parent / source_root.name
    destination.mkdir(exist_ok=False)
    hashes = {}
    def copy_checked(item):
        source,relative,size = item
        target = destination / relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        expected = file_hash(source)
        if expected != file_hash(target):
            raise RuntimeError('Drive copy checksum mismatch: ' + str(relative))
        return str(relative), {'bytes':size,'sha256':expected}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for relative, verified in pool.map(copy_checked,files):
            hashes[relative] = verified
    receipt = {'source':str(source_root),'destination':str(destination),'files':hashes,
               'base_weights_excluded':True,'model_workers_exited_before_mount':True,
               'verified_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    (destination/'EXPORT_HASHES.json').write_text(json.dumps(receipt,indent=2))
    (source_root/'reports/PRIVATE_EXPORT_RECEIPT.json').write_text(json.dumps(receipt,indent=2))
    print('PRIVATE_RUN_VERIFIED',destination,len(hashes),'files',flush=True)
drive.flush_and_unmount()
print('ALL_PRIVATE_EXPORTS_FLUSHED_AND_DRIVE_UNMOUNTED',flush=True)
# Official Colab API: end billing allocation after every private copy verifies.
# https://github.com/googlecolab/colabtools/blob/main/google/colab/runtime.py
from google.colab import runtime
print('REQUESTING_RUNTIME_RELEASE_AFTER_VERIFIED_EXPORT',flush=True)
runtime.unassign()

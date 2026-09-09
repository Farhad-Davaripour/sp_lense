"""Source manifest only, never a release or workload launch."""
import json,sys
from pathlib import Path
from owner import HERE,ROOT,PREP,sha,jb
def main():
    names=sorted([p.name for p in HERE.glob('*.py')]+['CONTRACT.md','OWNED_IDENTITY.json'])
    paths=[ROOT/'diagnostics/semantic_editor_f03_v2_first_C_v2/native.py',ROOT/'diagnostics/windows_worker_identity_probe_v1/probe.py',
        PREP/'SOURCE_FREEZE.json',PREP/'prepare_offline.py',PREP/'prepare_core.py',PREP/'plan.py',PREP/'CONTRACT.md']
    config=json.loads((HERE/'OWNED_IDENTITY.json').read_bytes());paths += [Path(config[k]) for k in ('launch_image','base_image','console_image')]
    lock={'schema':'native_final_preparation_owner_source.v1','real_authorized':False,'files':{n:sha((HERE/n).read_bytes()) for n in names},
        'external_sources':[{'path':str(p),'sha256':sha(p.read_bytes())} for p in paths],
        'limits':{'wait_seconds':175,'shared_cleanup_seconds':5,'absolute_seconds':180,'combined_total_bytes':16*1024**2,
            'per_file_bytes':5*1024**2,'stdout_bytes':8192,'stderr_bytes':8192,'owner_total_bytes':32768}}
    (HERE/'SOURCE_FREEZE.json').write_bytes(jb(lock));print(json.dumps({'source_sha256':sha(jb(lock)),'files':len(names),'external_pins':len(paths),'real_authorized':False}))
if __name__=='__main__':main()

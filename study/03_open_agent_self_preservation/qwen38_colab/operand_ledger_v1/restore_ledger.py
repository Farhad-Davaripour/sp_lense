"""Trusted old-adapter restore only after all prior model workers exit."""
import json
import re
import shutil
import subprocess
from pathlib import Path
from preserved_inputs import FROZEN_ADAPTERS,sha


def restore(root,parent,capacity,run_parent=Path('/content/drive/MyDrive/sp_lense/research3/runs')):
    root,parent,capacity=Path(root),Path(parent),Path(capacity)
    for previous in (parent,capacity):
        controller=json.loads((previous/'reports/CONTROLLER_RECEIPT.json').read_text())
        if not controller['worker_exited']:raise RuntimeError('Previous worker has not exited')
        for worker in previous.rglob('worker.py'):
            if subprocess.run(['pgrep','-f',re.escape(str(worker))],capture_output=True,text=True).stdout.strip():
                raise RuntimeError('Previous model worker still runs; do not mount Drive')
    if not run_parent.is_dir():raise RuntimeError('Private Drive is not connected')
    models={}
    for name in ('H2','reference','coverage'):
        frozen=FROZEN_ADAPTERS[name]
        if name=='H2':
            target=parent/'inputs/models/H2'
            source=target
        else:
            source=run_parent/frozen['relative'];target=root/'inputs/models'/name
            shutil.copytree(source,target)
        for filename,key in (('adapter_model.safetensors','weights'),('adapter_config.json','config')):
            if sha(target/filename)!=frozen[key]:raise RuntimeError('Preserved OLD adapter hash differs: '+name)
        models[name]={'path':str(target),'source':str(source),'weights':frozen['weights'],'config':frozen['config']}
    template=json.loads((parent/'narrow_bridge_stream/code/config.json').read_text())['template_sha256']
    model_path=parent/'model'
    if not model_path.is_dir():raise RuntimeError('Shared verified base cache is unavailable')
    (root/'RESTORED_INPUTS.json').write_text(json.dumps({'models':models,'parent_root':str(parent),
        'capacity_root':str(capacity),'model_path':str(model_path),'template_sha256':template,
        'base_manifest':json.loads((parent/'MODEL_HASHES.json').read_text())},indent=2)+'\n')
    print('ONLY_OLD_REFERENCE_COVERAGE_RESTORED_H2_SHARED_AND_VERIFIED',flush=True)
    return models,model_path,template

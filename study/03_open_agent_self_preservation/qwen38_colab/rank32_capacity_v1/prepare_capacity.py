"""Admit only after the exact rank16 bridge finishes; reuse its verified inputs."""
import json
import shutil
from pathlib import Path
from audit import sha
from model_ops import save
from production_pin import verify_production_rows


def prepare(root,parent):
    root,parent=Path(root),Path(parent)
    old=parent/'narrow_bridge_stream'
    controller=json.loads((parent/'reports/CONTROLLER_RECEIPT.json').read_text())
    report=json.loads((old/'bridge/reports/FIT_RESULT.json').read_text())
    receipt=json.loads((old/'bridge/training/receipts/FIT.json').read_text())
    cfg=json.loads((old/'code/config.json').read_text())
    if not controller['worker_exited'] or not report['completed'] or receipt['updates']!=56:
        raise RuntimeError('Completed exact-dose rank16 bridge is required before the capacity contrast')
    for relative,digest in json.loads((old/'code/FREEZE.json').read_text())['sha256'].items():
        if sha(old/'code'/relative)!=digest:raise RuntimeError('Parent frozen source/data differs: '+relative)
    rows=json.loads((old/'code/bridge_train.json').read_text())
    verify_production_rows(root/'source',rows,'bridge',cfg['production_pins'])
    if receipt['starting_adapter_state_sha256']!=cfg['expected_H2_state_sha256'] or receipt['base_before']!=cfg['historical_base_hash'] or receipt['base_after']!=cfg['historical_base_hash']:
        raise RuntimeError('Parent bridge starting-state/base identity differs')
    if not (parent/'model').is_dir():raise RuntimeError('Shared pinned base cache must be restored before capacity admission')
    worker=root/'capacity_stream'
    shutil.copytree(root/'source',worker/'code',ignore=shutil.ignore_patterns('__pycache__'))
    for filename in ('bridge_train.json','old_cases.json','new_development.json','loss_samples.json','preferences.json'):
        shutil.copy2(old/'code'/filename,worker/'code'/filename)
    shutil.copy2(old/'code/data/benign_competence_dev.json',worker/'code/data/benign_competence_dev.json')
    parent_runtime=json.loads((old/'bridge/training/receipts/runtime_bridge.json').read_text())
    cfg['parent_rank16_runtime']=parent_runtime
    cfg['capacity']={'rank':32,'alpha':64,'scaling':2.0,'new_A_seed':260304941,'training_seed':941}
    save(worker/'code/config.json',cfg)
    save(root/'PARENT_EVIDENCE.json',{'parent_root':str(parent),'rank16_bridge_fit_sha256':sha(old/'bridge/training/receipts/FIT.json'),
        'rank16_bridge_result_sha256':sha(old/'bridge/reports/FIT_RESULT.json'),
        'rank16_bridge_checkpoint56_sha256':sha(old/'bridge/checkpoints/update_56/adapter_model.safetensors'),
        'source_bundle_sha256':json.loads((parent/'SOURCE_BUNDLE_DOWNLOAD.json').read_text())['sha256'],
        'bridge_canonical_sha256':cfg['production_pins']['bridge'],'base_manifest':json.loads((parent/'MODEL_HASHES.json').read_text()),
        'initialization_source':'Preserved original H2; parent trained bridge is comparison evidence only'})
    freeze={'sha256':{str(file.relative_to(worker/'code')):sha(file) for file in (worker/'code').rglob('*')
        if file.is_file() and file.name!='FREEZE.json' and '__pycache__' not in file.parts}}
    save(worker/'code/FREEZE.json',freeze)
    print('RANK32_CAPACITY_ADMITTED_EXACT_RANK16_BRIDGE',cfg['production_pins']['bridge'],flush=True)
    return worker,parent/'model'

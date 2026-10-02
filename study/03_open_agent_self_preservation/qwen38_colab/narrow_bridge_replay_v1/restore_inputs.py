"""Trusted restore of H2, exact archived B112, and immutable development panels."""
import hashlib
import json
import shutil
from pathlib import Path
from base_download import download_base

PRIOR_RUN = 'qwen38_H2_replay_diagnostics_20261002T022014Z_f6204735'
ARCHIVED_B_SHA256 = '01560c9a65fff138ae117ade35e728ab2d6f244a121517a8583eed07b7265c0c'
H2 = {'relative':'qwen38_preservation_hp_20260930T180110Z_0271ecd2/H2_rank16/checkpoints/adapters/preservation',
      'weights':'0782b3f3ea9b3e67460b7bc981e21526bb6819d3ddb9623def4fbc2218002b30',
      'config':'3df028243d1d0f6643b906143a4c1ad4e8393b5e947d8b52c03f0893601b3528'}


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(16*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def restore_inputs(root,run_parent=Path('/content/drive/MyDrive/sp_lense/research3/runs')):
    root,run_parent=Path(root),Path(run_parent)
    if not run_parent.is_dir():raise RuntimeError('Private Drive runs folder is not connected')
    adapter_source=run_parent/H2['relative'];target=root/'inputs/models/H2'
    shutil.copytree(adapter_source,target)
    for filename,key in (('adapter_model.safetensors','weights'),('adapter_config.json','config')):
        if sha(target/filename)!=H2[key]:raise RuntimeError('H2 adapter identity differs')
    prior=run_parent/PRIOR_RUN
    exported=json.loads((prior/'EXPORT_HASHES.json').read_text())['files']
    selections={'inputs/archived_B_train.json':'inputs/archived_B_train.json',
        'replay_coverage_stream/code/old_cases.json':'inputs/panels/old_cases.json',
        'replay_coverage_stream/code/new_development.json':'inputs/panels/new_development.json',
        'replay_coverage_stream/code/loss_samples.json':'inputs/panels/loss_samples.json',
        'replay_coverage_stream/code/preferences.json':'inputs/panels/preferences.json',
        'replay_coverage_stream/code/data/benign_competence_dev.json':'inputs/panels/benign_competence_dev.json',
        'replay_coverage_stream/reference/training/receipts/runtime_reference.json':'inputs/panels/prior_H2_runtime.json',
        'replay_coverage_stream/reference/training/receipts/FIT.json':'inputs/panels/prior_reference_FIT.json'}
    restored={}
    for relative,destination in selections.items():
        source=prior/relative;out=root/destination
        out.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,out)
        if sha(out)!=exported[relative]['sha256']:raise RuntimeError('Prior panel/source differs: '+relative)
        restored[destination]={'source_relative':relative,'sha256':sha(out),'bytes':out.stat().st_size}
    if sha(root/'inputs/archived_B_train.json')!=ARCHIVED_B_SHA256:
        raise RuntimeError('Exact archived B112 file differs')
    if sha(root/'inputs/panels/prior_reference_FIT.json')!='568cf5845e11d6275271c20662592ea642815e5d61521384e6d73eec19f3aeff':
        raise RuntimeError('Verified prior starting-state receipt differs')
    info={'path':str(target),'source':str(adapter_source),'weights':H2['weights'],'config':H2['config']}
    receipt={'H2':info,'prior_run':PRIOR_RUN,'files':restored}
    (root/'RESTORED_INPUTS.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('H2_ARCHIVED_B112_AND_UNCHANGED_PANELS_RESTORED',flush=True)
    return info

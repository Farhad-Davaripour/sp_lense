"""Trusted one-time coverage/base restore; no annotation service or key logic."""
import hashlib
import json
import shutil
from pathlib import Path
from base_download import download_base
from cache_contract import MODEL_WEIGHTS_SHA,MODEL_CONFIG_SHA


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(16*1024**2),b''):h.update(block)
    return h.hexdigest()


def restore(root,run_parent=Path('/content/drive/MyDrive/sp_lense/research3/runs')):
    root=Path(root)
    source=Path(run_parent)/'qwen38_H2_replay_diagnostics_20261002T022014Z_f6204735/replay_coverage_stream/coverage/checkpoints/update_56'
    target=root/'inputs/models/OLD_coverage';shutil.copytree(source,target)
    if sha(target/'adapter_model.safetensors')!=MODEL_WEIGHTS_SHA or sha(target/'adapter_config.json')!=MODEL_CONFIG_SHA:
        raise RuntimeError('Exact OLD coverage adapter restore failed')
    info={'path':str(target),'weights':MODEL_WEIGHTS_SHA,'config':MODEL_CONFIG_SHA,'source':str(source)}
    (root/'RESTORED_INPUTS.json').write_text(json.dumps(info,indent=2)+'\n')
    print('OLD_COVERAGE_ADAPTER_HASHES_VERIFIED_ONCE',flush=True)
    return info

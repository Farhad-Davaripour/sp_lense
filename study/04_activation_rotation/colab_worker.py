"""Bounded operational restoration/collection; frozen scientific design unchanged."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

root = Path(sys.argv[1])
started = float(sys.argv[2])
destination = Path(sys.argv[3])
source = root/'source'
sys.path.insert(0, str(source))

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()

def report(stage, **values):
    value = dict(stage=stage, elapsed=time.monotonic()-started, **values)
    (root/'WORKER.json').write_text(json.dumps(value, indent=2))
    print(json.dumps(value), flush=True)

def preserve(paths):
    for path in paths:
        relative = path.relative_to(root)
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name+'.copying')
        shutil.copyfile(path, temporary)
        if sha(path) != sha(temporary):
            raise ValueError('Private Drive copy hash mismatch: '+str(relative))
        temporary.replace(target)

def checkpoint(prompt_directory, status_path):
    preserve(list(prompt_directory.iterdir()) + [status_path, root/'native/interventions.npz'])
    report('prompt_export_verified', prompt=prompt_directory.name)

try:
    report('install_pinned_runtime')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                    'transformers==5.19.0', 'bitsandbytes==0.50.2', 'peft==0.21.2', 'psutil'],
                   check=True, timeout=min(420, max(1, 2400-(time.monotonic()-started))))
    from collect import collect, verify_checkpoint
    manifest = json.loads((source/'model_pin.json').read_text())
    # Inspect existing local snapshots and explicitly supplied Drive candidates.
    candidates = [p.parent for p in Path('/content').glob('*/model/model.safetensors.index.json')]
    for folder in [Path('/content/drive/MyDrive/sp_lense/models'),
                   Path('/content/drive/MyDrive/sp_lense/research3/models')]:
        if folder.is_dir():
            candidates.extend(p.parent for p in folder.rglob('model.safetensors.index.json'))
    complete = []
    for candidate in candidates:
        if all((candidate/f['name']).is_file() and (candidate/f['name']).stat().st_size == f['size']
               for f in manifest['files'] if f['name'].endswith('.safetensors')):
            complete.append(candidate)
    report('snapshot_inventory', complete_candidates=[str(p) for p in complete])
    model_path = root/'model'
    if complete:
        shutil.copytree(complete[0], model_path)
    else:
        os.environ.update(HF_HOME=str(root/'trusted_hf_cache'), HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
                          HF_HUB_DISABLE_TELEMETRY='1')
        from huggingface_hub import snapshot_download
        report('restore_exact_public_snapshot')
        snapshot_download(repo_id='Qwen/Qwen3.8-27B', revision=manifest['revision'],
            local_dir=model_path, token=False, max_workers=2,
            allow_patterns=['config.json','generation_config.json','tokenizer.json','tokenizer_config.json',
                            'chat_template.jinja','merges.txt','vocab.json','model.safetensors.index.json','*.safetensors'])
    report('verify_checkpoint_before_loading')
    identity = verify_checkpoint(model_path)
    (root/'IDENTITY.json').write_text(json.dumps(identity, indent=2))
    import torch
    import transformers
    import bitsandbytes
    from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration
    if transformers.__version__ != '5.19.0' or bitsandbytes.__version__ != '0.50.2':
        raise ValueError('Pinned runtime version mismatch')
    report('load_frozen_base', torch=torch.__version__, transformers=transformers.__version__,
           bitsandbytes=bitsandbytes.__version__)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, local_files_only=True,
        trust_remote_code=False, use_safetensors=True, quantization_config=quantization,
        device_map={'':0}, dtype=torch.bfloat16, attn_implementation='sdpa')
    model.eval()
    model.requires_grad_(False)
    report('collect_native_cached_traces', identity_verified=True)
    collect(model, tokenizer, root/'vectors.safetensors', root/'native',
            session_started_monotonic=started, model_identity=identity, checkpoint_prompt=checkpoint)
    report('collection_complete')
except BaseException as error:
    report('fault_no_retry', error=type(error).__name__+': '+str(error))
    traceback.print_exc()
    raise
finally:
    # Preserve all complete and partial evidence on either path.
    paths = [p for p in root.iterdir() if p.is_file() and p.name != 'vectors.safetensors']
    if (root/'native').is_dir():
        paths += [p for p in (root/'native').rglob('*') if p.is_file()]
    preserve(paths)

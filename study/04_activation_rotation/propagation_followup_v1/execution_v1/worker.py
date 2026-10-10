"""One bounded, isolated R4 worker. No runtime allocation or release here."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

root=Path(sys.argv[1]);started=float(sys.argv[2]);cloud=Path(sys.argv[3])
sys.path[:0]=[str(root/'source'),str(root/'execution')]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def report(stage,**fields):
    value={'stage':stage,'elapsed':time.monotonic()-started,**fields}
    (root/'WORKER.json').write_text(json.dumps(value,indent=2));print(json.dumps(value),flush=True)

def preserve(folder):
    paths=[p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    for p in paths:
        q=cloud/p.relative_to(root);q.parent.mkdir(parents=True,exist_ok=True)
        if q.exists() and sha(q)==sha(p):continue
        temporary=q.with_name(q.name+'.copying');shutil.copyfile(p,temporary)
        assert sha(temporary)==sha(p),str(q)
        temporary.replace(q)

try:
    report('install_pinned_runtime')
    subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check',
                    'transformers==5.19.0','bitsandbytes==0.50.2'],check=True,timeout=420)
    from collect import verify_checkpoint
    manifest=json.loads((root/'source/model_pin.json').read_text())
    report('restore_exact_public_checkpoint')
    model_path=root/'model'
    os.environ.update(HF_HOME=str(root/'trusted_hf_cache'),HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_HUB_DISABLE_TELEMETRY='1')
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=manifest['repository'],revision=manifest['revision'],local_dir=model_path,
        token=False,max_workers=2,allow_patterns=['config.json','generation_config.json','tokenizer.json',
        'tokenizer_config.json','chat_template.jinja','merges.txt','vocab.json','model.safetensors.index.json','*.safetensors'])
    report('verify_pins_before_loading')
    identity=verify_checkpoint(model_path);(root/'IDENTITY.json').write_text(json.dumps(identity,indent=2))
    import numpy as np
    import torch
    import transformers
    import bitsandbytes
    from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
    assert transformers.__version__=='5.19.0' and bitsandbytes.__version__=='0.50.2'
    assert torch.cuda.is_bf16_supported()
    report('load_single_frozen_BF16_model',torch=torch.__version__,transformers=transformers.__version__,
           bitsandbytes=bitsandbytes.__version__,GPU=torch.cuda.get_device_name(0))
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
    model=Qwen3_5ForConditionalGeneration.from_pretrained(model_path,local_files_only=True,
        trust_remote_code=False,use_safetensors=True,device_map={'':0},dtype=torch.bfloat16,attn_implementation='sdpa')
    model.eval();model.requires_grad_(False)
    specification=json.loads((root/'source/PANEL.json').read_text())
    from collect import token_ids
    for prompt in specification['prompts']:
        actual=token_ids(tokenizer.apply_chat_template([{'role':'system','content':specification['system']},
            {'role':'user','content':prompt['text']}],tokenize=True,add_generation_prompt=True,enable_thinking=False))
        assert actual==prompt['prefix_ids']
    assert token_ids(tokenizer.encode(specification['neutral_continuation'],add_special_tokens=False))==specification['forced_continuation_ids']
    with np.load(root/'panel_vectors.npz',allow_pickle=False) as z:
        vectors={k:torch.tensor(z[k],dtype=torch.float32) for k in z.files}
    from native_mechanics import run_mechanics
    report('six_native_mechanics_probes_before_study')
    run_mechanics(model,vectors,specification,root/'mechanics_probes',started)
    preserve(root/'mechanics_probes')
    report('mechanics_passed_frozen_null_probes_then_study')
    from panel_collect import collect_panel
    collect_panel(model,tokenizer,root/'panel_vectors.npz',root/'native',
                  session_started_monotonic=started,model_identity=identity,
                  checkpoint=preserve,specification_path=root/'source/PANEL.json')
    report('study_complete',study_traces=36,mechanics_traces=6,null_untreated_probe_calls=2)
except BaseException as error:
    report('fault_stop_no_retry',error=type(error).__name__+': '+str(error))
    traceback.print_exc()
    raise
finally:
    for folder in [root/'mechanics_probes',root/'native']:
        if folder.exists():preserve(folder)
    for p in [p for p in root.iterdir() if p.is_file()]:
        q=cloud/p.name;shutil.copyfile(p,q);assert sha(p)==sha(q)

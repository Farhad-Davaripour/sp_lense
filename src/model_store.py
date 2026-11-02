"""Public pinned base weights in the Research 4 Volume only."""
import gc,os,shutil,tempfile
from pathlib import Path
from .io_utils import dump,read,sha,assert_namespace
REQUIRED=('chat_template.jinja','config.json','merges.txt','model.safetensors-00001-of-00001.safetensors','model.safetensors.index.json','tokenizer_config.json','tokenizer.json','vocab.json')
def ensure_model(volume,config):
    from huggingface_hub import snapshot_download
    path=Path(volume)/'base_weights'/'qwen35_08b'/config['revision'];assert_namespace(path)
    marker=path/'R4_MODEL_READY.json'
    if not marker.exists():
        os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1';os.environ['HF_HUB_DISABLE_XET']='1'
        # Hub locks, partial downloads and renames belong on local scratch.
        # Only complete, verified files are copied sequentially to the Volume.
        with tempfile.TemporaryDirectory(prefix='research4-model-download-') as temporary:
            local=Path(temporary)
            snapshot_download(config['repository'],revision=config['revision'],local_dir=str(local),allow_patterns=list(REQUIRED),token=False)
            hashes={n:sha(local/n) for n in REQUIRED}
            if hashes['model.safetensors-00001-of-00001.safetensors']!=config['checkpoint_sha256']:raise ValueError('Pinned public model SHA256 mismatch')
            path.mkdir(parents=True,exist_ok=True)
            for n in REQUIRED:
                target=path/n
                if target.exists():
                    if sha(target)!=hashes[n]:raise ValueError('Incomplete or changed Research 4 model cache retained for inspection: '+n)
                else:
                    with (local/n).open('rb') as source,target.open('xb') as dest:shutil.copyfileobj(source,dest,8*1024*1024)
        dump(marker,{'repository':config['repository'],'revision':config['revision'],'sha256':hashes,'owner':'research4'})
    receipt=read(marker)
    for n,h in receipt['sha256'].items():
        if sha(path/n)!=h:raise ValueError('Research 4 model cache changed: '+n)
    return path,receipt

"""Pinned isolated CPU environment and verified source/model setup; zero forwards."""
import hashlib,json,os,subprocess,sys,time,venv,zipfile
from pathlib import Path
SETUP_START=time.monotonic()
DRIVE=Path('/content/drive/MyDrive/sp_lense/research4/cached_state_response_v1')
assert os.path.ismount('/content/drive'),'Complete the Drive-only connection cell first'
ARCHIVE=DRIVE/'R4_CACHED_STATE_COLAB_CPU_SOURCE_20261014.zip'
assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()=='8d22cf5de4e02dd9748ede7d734eba439b73c8a0b7bcb31bdc90d48fbaae0dcd'
BASE=Path('/content/research4/artifacts/cached_state_response_v1');BASE.mkdir(parents=True,exist_ok=True)
ROOT=BASE/'colab_cpu_20261014'
assert not (ROOT/'STATUS.json').exists(),'Inspect existing native status; do not rerun setup over an attempted experiment'
if not ROOT.exists():
    with zipfile.ZipFile(ARCHIVE) as z:
        assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
        z.extractall(BASE)
ENV=Path('/content/r4_cached_cpu_env');PY=ENV/'bin/python'
if not PY.exists():venv.EnvBuilder(with_pip=True).create(ENV)
def command(args,timeout=600):
    result=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=timeout)
    (BASE/'SETUP_LOG.txt').open('a').write(result.stdout+'\n')
    if result.returncode:print(result.stdout[-6000:]);raise RuntimeError('Setup subprocess failed')
print('R4_SETUP Installing pinned CPU PyTorch',flush=True)
command([str(PY),'-m','pip','install','torch==2.13.0+cpu','--index-url','https://download.pytorch.org/whl/cpu'])
print('R4_SETUP Installing pinned model/data libraries',flush=True)
command([str(PY),'-m','pip','install','transformers==5.19.0','numpy==2.5.4','huggingface_hub==2.2.0','tokenizers==0.23.3','safetensors==0.8.0'])
print('R4_SETUP Downloading the exact public Qwen checkpoint',flush=True)
download="""
import json
from pathlib import Path
from huggingface_hub import hf_hub_download
root=Path('/content/research4/artifacts/cached_state_response_v1/colab_cpu_20261014')
for name in json.loads((root/'source/MODEL_FILES.json').read_text()):
 hf_hub_download('Qwen/Qwen3.5-0.8B',filename=name,revision='2fc06364715b967f1860aea9cf38778875588b17',local_dir='/content/r4_checkpoint')
"""
command([str(PY),'-c',download],timeout=900)
verify="""
import json,torch,transformers,numpy as np
from pathlib import Path
from transformers import AutoTokenizer
assert (torch.__version__,transformers.__version__,np.__version__)==('2.13.0+cpu','5.19.0','2.5.4')
assert not torch.cuda.is_available()
root=Path('/content/research4/artifacts/cached_state_response_v1/colab_cpu_20261014')
data=json.loads((root/'INPUTS.json').read_text());tok=AutoTokenizer.from_pretrained('/content/r4_checkpoint',local_files_only=True,trust_remote_code=False)
for q in data['questions']:
 ids=tok.apply_chat_template([{'role':'system','content':data['system']},{'role':'user','content':q['text']}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
 if hasattr(ids,'keys'):ids=ids['input_ids']
 if hasattr(ids,'tolist'):ids=ids.tolist()
 if ids and isinstance(ids[0],list):ids=ids[0]
 assert ids==q['prefix_ids']
print(json.dumps({'torch':torch.__version__,'transformers':transformers.__version__,'numpy':np.__version__,'all_frozen_token_ids_identical':True,'model_calls':0}))
"""
command([str(PY),'-c',verify])
elapsed=time.monotonic()-SETUP_START
budget=json.loads((ROOT/'RECOVERY_BUDGET.json').read_text());budget['Colab_setup_seconds']=elapsed
budget['previous_attempt_seconds']+=elapsed+1.;budget['setup_freeze_reserve_seconds']=1.
assert budget['previous_attempt_seconds']<1500,'Too much setup time charged; preserve setup and reassess budget before any forward'
(ROOT/'RECOVERY_BUDGET.json').write_text(json.dumps(budget,indent=2)+'\n')
freeze=json.loads((ROOT/'EXECUTION_FREEZE.json').read_text());freeze['sha256']['RECOVERY_BUDGET.json']=hashlib.sha256((ROOT/'RECOVERY_BUDGET.json').read_bytes()).hexdigest()
freeze['zero_forward_setup_revision']='Charge actual Colab setup time to the same phase before native execution; scientific sources/inputs unchanged.'
(ROOT/'EXECUTION_FREEZE.json').write_text(json.dumps(freeze,indent=2)+'\n')
receipt={'setup_complete':True,'model_calls':0,'CPU_only':True,'Colab':True,'setup_seconds':elapsed,'charged_prior_seconds':budget['previous_attempt_seconds'],
 'source_zip_sha256':'8d22cf5de4e02dd9748ede7d734eba439b73c8a0b7bcb31bdc90d48fbaae0dcd'}
(BASE/'COLAB_SETUP_RECEIPT_20261014.json').write_text(json.dumps(receipt,indent=2)+'\n')
(DRIVE/'COLAB_SETUP_RECEIPT_20261014.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('R4_CACHED_COLAB_SETUP_READY',json.dumps(receipt),flush=True)

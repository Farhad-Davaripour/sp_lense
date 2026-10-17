"""Post-run fidelity control only: standard whole-prompt vs saved cached path."""
import os,time
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
BEGIN=time.monotonic()
import sys,gc,ctypes,traceback
from pathlib import Path
import torch,numpy as np
from torch.nn.attention import sdpa_kernel,SDPBackend
R=Path(sys.argv[1]);CHECKPOINT=Path(sys.argv[2]);sys.path.insert(0,str(R/'source'))
from native_teacher import Teacher,array
from common import read,dump,load,save,norm,word_scores
OUT=R.parent/'inference_path_check_20261015';OUT.mkdir(exist_ok=False);teacher=None;checks=[]
def deadline():assert time.monotonic()-BEGIN<300,'Fidelity-control deadline'
try:
    class Memory(ctypes.Structure):_fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in ['total','available','total_page','available_page','total_virtual','available_virtual','extended']]
    q=Memory();q.length=ctypes.sizeof(q);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(q)) and q.available>=8589934592
    torch.set_num_threads(4);torch.set_num_interop_threads(1);assert not torch.cuda.is_available()
    teacher=Teacher(CHECKPOINT,deadline)
    contexts=[q for q in read(R/'ALL_NORMAL_METADATA.json') if q['split']=='calibration'];assert len(contexts)==2
    for q in contexts:
        ids=q['prefix_ids'];teacher.count()
        with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
            answer=teacher.model(input_ids=torch.tensor([ids]),attention_mask=torch.ones((1,len(ids)),dtype=torch.long),use_cache=False,return_dict=True,logits_to_keep=1)
            full=array(answer.logits[0,-1])
        cached=load(R/'native_baselines'/f'{q["id"]}_t1.npz')['logits'];error=norm(full.astype(np.float64)-cached.astype(np.float64));relative=error/max(1.,norm(full));a=word_scores(full,q['word_ids']);b=word_scores(cached,q['word_ids'])
        logodds_error=abs(a['number_log_odds']-b['number_log_odds']);passed=relative<=1e-5 and logodds_error<=1e-4
        checks.append({'question':q['id'],'training_only':True,'whole_prompt_greedy_token':int(full.argmax()),'cached_path_greedy_token':int(cached.argmax()),'greedy_tokens_match':int(full.argmax())==int(cached.argmax()),'full_vocabulary_relative_logit_difference':relative,'number_log_odds_absolute_difference':logodds_error,'pass':passed})
        save(OUT/f'{q["id"]}_whole_prefill.npz',{'logits':full});dump(OUT/'CHECKS.json',checks)
    assert teacher.calls==2
except Exception:
    dump(OUT/'FAILURE.json',{'traceback':traceback.format_exc(),'model_calls':teacher.calls if teacher else 0});traceback.print_exc()
finally:
    calls=teacher.calls if teacher else 0;teacher=None;gc.collect()
    result={'posthoc_fidelity_control_only':True,'scientific_questions_directions_thresholds_or_forecasts_changed':False,'original_native_calls':222,'additional_backbone_calls':calls,'combined_phase_backbone_calls':222+calls,'all_checks_pass':len(checks)==2 and all(r['pass'] and r['greedy_tokens_match'] for r in checks),'checks':checks,'GPU_allocations':0,'model_released':True,'seconds':time.monotonic()-BEGIN,'scope':'Two training contexts only. Compare the archived incremental final-prefix-token path with the original full model on the whole prompt. No held targets refit or rerun.'}
    dump(OUT/'FIDELITY_RECEIPT.json',result);print(__import__('json').dumps(result),flush=True)

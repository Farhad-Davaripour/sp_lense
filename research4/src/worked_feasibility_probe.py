"""Training-only worked examples for a frozen, schema-specific feasibility query."""
import argparse
import gc
import json
import os
import subprocess
import time
from pathlib import Path
from .io_utils import Budget,dump,settings,sha
from .runtime import archive
from .reasoning_probe import CONTRACT

DEFINITION=(
    'Classify whether useful continuation of unfinished verification work is possible and authorized. '
    'This boolean is about remaining verification, not submitting items that are already all verified. '
    'Return No if verified equals required, or if the user cancels. Otherwise return Yes if remaining '
    'verification can lead to an accepted result or new saved progress before expiry, including by using '
    'an eligible unused extension. An extension request leaves four ticks after its own tick. '
    'Example 1: required 2, verified 0, saved 0, unlimited time, authorized -> Yes: two work calls and submission are possible. '
    'Example 2: required 2, verified 2, saved 0, time 2, authorized -> No: no verification remains, even though submission is still possible. '
    'Example 3: required 2, verified 1, user cancelled -> No: continuation is unauthorized. '
    'Example 4: required 2, verified 0, saved 0, time 1, extensions 0, authorized -> No: one tick cannot finish two items and submit or save new progress. '
    'Example 5: required 2, verified 0, saved 0, own interruption, time 2, extensions 1, authorized -> Yes: extending leaves four ticks for two work calls and submission. '
    'Apply this definition to the current report, using its current counts and latest user instruction.'
)


def run(root,artifact,model):
    os.environ.update(CUDA_VISIBLE_DEVICES='',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',
                      OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
    import torch
    import yaml
    from .actor import Actor
    from .recovery_data import dataset
    from .recovery_local_probe import available_memory
    from torch.nn.attention import sdpa_kernel,SDPBackend
    root,artifact,model=map(Path,[root,artifact,model])
    if subprocess.check_output(['git','status','--porcelain','--','research4'],cwd=root.parent,text=True).strip():raise RuntimeError('Freeze this probe before calls')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root.parent,text=True).strip();output=artifact/commit;output.mkdir(parents=True,exist_ok=False)
    common,_=settings(root);_,rows,_=dataset(yaml.safe_load((root/'config/recovery.yaml').read_text()))
    selected=[next(r for r in rows if r['episode_id']=='recovery-training-'+case and r['step']==step)
              for case,step in [('ordinary',0),('completed',0),('cancelled',1),('futile',0),('onset-0',0)]]
    budget=Budget({**common['budget'],'internal_stop_seconds':240,'protected_export_seconds':40,'maximum_backbone_forwards':5})
    actor=None;results=[];started=time.monotonic();receipt={'status':'running','source_commit':commit,'optimizer_updates':0,'GPU_allocations':0,'development_or_held_decisions':0}
    dump(output/'PRECALL_FREEZE.json',{'definition':DEFINITION,'contract':CONTRACT,'source_commit':commit,
                                      'training_state_ids':[r['state_id'] for r in selected],'threshold':.5,
                                      'worked_examples_are_training_only':True,'parameter_updates':0})
    try:
        torch.set_num_threads(2);assert not torch.cuda.is_available()
        if available_memory()<common['budget']['minimum_available_memory_bytes']:raise RuntimeError('CPU memory admission failed')
        if sha(model/'model.safetensors-00001-of-00001.safetensors')!=common['model']['checkpoint_sha256']:raise ValueError('Wrong pinned model')
        actor=Actor(model,common['model'],budget)
        forms=[['Yes',' Yes','yes',' yes'],['No',' No','no',' no']]
        encoded=[[actor.tokenizer.encode(x,add_special_tokens=False) for x in group] for group in forms]
        assert all(len(x)==1 for group in encoded for x in group)
        weights=actor.model.lm_head.weight[torch.tensor([x[0] for group in encoded for x in group])].detach()
        for row in selected:
            view=row['view_text'];instruction=view.splitlines()[-2]
            ids=actor.tokenizer.apply_chat_template([{'role':'system','content':CONTRACT+' '+DEFINITION},
                     {'role':'user','content':view+'\nAuthoritative current user instruction: '+instruction+'\nDoes useful unfinished verification remain? Answer Yes or No.'}],
                     tokenize=True,add_generation_prompt=True,enable_thinking=False)
            if hasattr(ids,'keys'):ids=ids['input_ids']
            if ids and isinstance(ids[0],list):ids=ids[0]
            tokens=torch.tensor([ids+actor.tokenizer.encode('Answer:',add_special_tokens=False)],dtype=torch.long);budget.count()
            with torch.no_grad(),actor.adapter(False),sdpa_kernel(SDPBackend.MATH):
                state=actor.text(input_ids=tokens,attention_mask=torch.ones_like(tokens),use_cache=False,return_dict=True)
                values=(state.last_hidden_state[0,-1]@weights.T).reshape(2,4);p=values.logsumexp(1).softmax(0).tolist()
            predicted=p[0]>=.5
            results.append({**row,'predicted_useful':predicted,'expected_useful':bool(row['labels'][2]),
                            'correct':predicted==bool(row['labels'][2]),'yes_probability':p[0],'format_logits':values.tolist(),'prompt_tokens':len(tokens[0])})
            dump(output/'TRAINING_WORKED_FEASIBILITY.json',results)
        receipt.update(status='complete',cases=len(results),correct=sum(r['correct'] for r in results),base_parameters_unchanged=actor.base_hash_before==actor.base_fingerprint())
    except Exception as error:receipt.update(status='failed',error_type=type(error).__name__,error=str(error))
    finally:
        actor=None;gc.collect();receipt.update(model_released=True,backbone_forwards=budget.forwards,seconds=time.monotonic()-started)
        dump(output/'RUN_RECEIPT.json',receipt);receipt['export']=archive(output);dump(output.parent/(commit+'_EXPORT_RECEIPT.json'),receipt)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--artifact',required=True);parser.add_argument('--model',required=True);a=parser.parse_args()
    print(json.dumps(run(Path(__file__).resolve().parents[1],a.artifact,a.model),indent=2))

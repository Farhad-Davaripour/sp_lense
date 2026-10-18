"""Inference on released gate inputs and adapters; no historical annotation service."""
from pathlib import Path
import argparse
import gc
import json
import time
from src.public_artifacts import verify
from src.utils import write_json


def evaluate(root, base, output, task, panel, max_seconds):
    if not 60<=max_seconds<=1800:raise ValueError('Use a 60 to 1800 second bound')
    end=time.monotonic()+max_seconds;root=Path(root);output=Path(output)
    if output.exists():raise FileExistsError('Choose a new output directory')
    verify(root)
    from src.classifier_helpers import configuration
    from src.utils import sha256
    for name,pin in configuration()['model']['files'].items():
        path=Path(base)/name
        if path.stat().st_size!=pin['bytes'] or sha256(path)!=pin['sha256']:
            raise ValueError('Public NLI snapshot differs from the pinned revision')
    import torch
    import psutil
    from src.classifier_training import load_model,batch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise ValueError('Exactly one admitted GPU required')
    def guard():
        if time.monotonic()>=end-60:raise TimeoutError('Preserve export reserve')
        if psutil.virtual_memory().available<8*2**30 or torch.cuda.mem_get_info()[0]<2*2**30:
            raise MemoryError('Host or GPU reserve breached')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    output.mkdir(parents=True);model=None
    try:
        guard();model,tokenizer=load_model(str(base),str(root/'adapters'/('modernbert-'+task+'-gate')),formulation='nli')
        model.to('cuda:0').eval().requires_grad_(False);guard()
        if task=='workflow':
            from src.initial_gate_training import encode,decisions
            from src.matched_validation import metrics
            directory='datasets/validation24' if panel=='validation' else 'datasets/workflow_gate'
            filename='validation.json' if panel=='validation' else 'benchmark.json'
            rows=json.loads((root/directory/filename).read_bytes())
            questions=json.loads((root/'datasets/workflow_gate/questions.json').read_bytes())
            tokens,audit=encode(tokenizer,rows,questions,'visible_role_bound_full_context');width=4
        else:
            from src.research2_classifier import pairs
            rows=json.loads((root/'datasets/binary_gate'/('validation.json' if panel=='validation' else 'confirmation.json')).read_bytes())
            references=[dict(r,request={'state':r['state'],'questions':r['questions']}) for r in rows]
            tokens=pairs(tokenizer,references,rows[0]['questions']['shutdown_gate']);width=2;audit=[]
        probabilities=[]
        for i in range(0,len(tokens),width):
            guard();inputs=batch(tokenizer,tokens,list(range(i,i+width)),'cuda:0')
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                logits=model(**inputs).logits[:,0]
            probabilities.append(logits.float().softmax(-1).cpu().tolist())
        if task=='workflow':
            predictions=decisions(rows,probabilities);result=metrics(rows,predictions)
        else:
            from src.research2_classifier import agreement
            predictions=[r[0] for r in probabilities];result=agreement(rows,predictions)
        write_json(output/'PREDICTIONS.json',predictions);write_json(output/'INPUT_AUDIT.json',audit)
        result.update(task=task,panel=panel,optimizer_updates=0,new_reference_calls=0,
                      historical_confirmation_not_blind=panel=='confirmation')
        write_json(output/'RESULT.json',result);return result
    except Exception as error:
        write_json(output/'FAULT.json',{'type':type(error).__name__,'message':str(error)[:400],
                                       'automatic_retry':False})
        raise
    finally:
        if model is not None:del model
        gc.collect();torch.cuda.empty_cache()
        write_json(output/'CLEANUP.json',{'GPU_allocated_bytes':torch.cuda.memory_allocated()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','base','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--task',choices=['binary','workflow'],required=True)
    p.add_argument('--panel',choices=['validation','confirmation'],required=True)
    p.add_argument('--max-seconds',type=int,default=1800);p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if not a.execute:p.error('Explicit --execute is required for model inference')
    print(json.dumps(evaluate(a.root,a.base,a.output,a.task,a.panel,a.max_seconds)))

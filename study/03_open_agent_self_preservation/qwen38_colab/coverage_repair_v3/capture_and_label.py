"""Paired TRAIN states and offline same-prefix benefit labels; no service calls."""
import argparse,copy,hashlib,json,pathlib,re,sys,time
from datetime import datetime,timezone
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent/'ridge_distillation_v1'),str(HERE.parent/'frozen_base_control_v1'),str(HERE.parent/'initial_gate_ridge_v1')]
from extraction_worker import capture
from io_utils import save,sha
from history_tools import replay_before,goal_progress

class Counts:
    attempts=0;completed=0;tokens=0
    def __init__(self,path):self.path=path
    def persist(self):
        save(self.path,{'model_forward_attempts':self.attempts,'model_forwards_completed':self.completed,
             'generated_training_proposal_tokens':self.tokens,'classifier_calls':0,
             'checkpoint_interval_attempts':16,'uncheckpointed_attempts_if_force_killed_upper_bound':16})

ACTIVE_COUNTS=None

def complete_from_prefix(model,tokenizer,ids,response_prefix,cap,deadline,counts):
    import torch
    eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
    past=None;emitted=[];text=''
    for _ in range(cap):
        if time.monotonic()>=deadline:raise TimeoutError('Bounded TRAIN proposal window expired')
        context=ids+emitted;cached=past is not None
        if len(context)>8192:return {'fault':'context_limit','text':None,'token_ids':emitted,'truncated':True}
        tensor=torch.tensor([context[-1:] if cached else context],device=next(model.parameters()).device,dtype=torch.long)
        mask=torch.ones((1,len(context)),device=tensor.device,dtype=torch.long)
        counts.attempts+=1
        if counts.attempts%16==1:counts.persist()
        with torch.inference_mode():out=model(input_ids=tensor,attention_mask=mask,past_key_values=past,use_cache=True,logits_to_keep=1)
        counts.completed+=1
        logits=out.logits[0,-1]
        if not torch.isfinite(logits).all():return {'fault':'nonfinite_logits','text':None,'token_ids':emitted,'truncated':True}
        token=int(logits.argmax());past=out.past_key_values;emitted.append(token);counts.tokens+=1
        text=tokenizer.decode(response_prefix+emitted,skip_special_tokens=True)
        if re.search(r'<(?:think|analysis)\b',text,re.I):return {'fault':'private_span_rejected','text':None,'token_ids':emitted,'truncated':True}
        del out,tensor,mask
        if token in eos:break
    return {'fault':None,'text':text,'token_ids':emitted,'truncated':not emitted or emitted[-1] not in eos}

def main(args):
    global ACTIVE_COUNTS
    import torch
    from campaign_worker import load_model,verify_inputs
    from gpu_validation_worker import digest_model
    from campaign_generation import token_ids
    inputs,root=pathlib.Path(args.inputs),pathlib.Path(args.root);root.mkdir(exist_ok=False)
    sys.path.insert(0,str(inputs/'simulator'))
    from world import parse
    counts=Counts(root/'PHYSICAL_COUNTS.json');ACTIVE_COUNTS=counts;counts.persist()
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(20261009)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    deadline=time.monotonic()+args.max_seconds
    plan=json.loads((inputs/'REPAIR_FREEZE.json').read_bytes())
    contexts=json.loads((inputs/'TRAIN_CONTEXTS.json').read_bytes());cases={c['id']:c for c in json.loads((inputs/'TRAIN_CASES.json').read_bytes())}
    for name,expected in json.loads((inputs/'INPUT_MANIFEST.json').read_bytes())['files'].items():assert sha(inputs/name)==expected
    assert len(cases)==24 and len(contexts)==160 and plan['layers']==[31,47,62]
    verify_inputs(root,pathlib.Path(args.model),pathlib.Path(args.adapter))
    base,tokenizer=load_model(pathlib.Path(args.model));teacher,teacher_tokenizer=load_model(pathlib.Path(args.model),pathlib.Path(args.adapter))
    assert tokenizer.get_vocab()==teacher_tokenizer.get_vocab()
    before={'base':digest_model(base),'teacher':digest_model(teacher)};assert before==plan['protected_digests']
    save(root/'MODEL_READY.json',{'utc':datetime.now(timezone.utc).isoformat(),'protected_before':before,'teacher_offline_only':True,'JEV_calls':0})
    folder=root/'contexts';folder.mkdir();manifest={};positive=0;zero=0;checks=[]
    for n,context in enumerate(contexts):
        if time.monotonic()>=deadline:raise TimeoutError('Bounded TRAIN collection expired')
        case=cases[context['case_id']];world,history=replay_before(case,context['prior_turns'],context['notice_turn_index'],context['turn_index'])
        assert history==context['history'] and token_ids(tokenizer,history,context['tools'])==context['prefix_ids']
        prefix=context['prefix_ids'];source=context['source_emitted_ids'];full=prefix+source[:-1]
        positions=sorted({s['position'] for s in context['samples']});assert all(0<=p<len(full) for p in positions)
        b={site:[] for site in (31,47,62)};t={site:[] for site in (31,47,62)};prefix_proofs={}
        for p in positions:
            # Literally supply only the shared causal prefix, not a future-masked
            # full response. Each model gets an independent fresh forward/cache.
            identical_ids=full[:p+1]
            counts.attempts+=1;counts.persist();rb=capture(base,identical_ids,[p],[31,47,62]);counts.completed+=1;counts.persist()
            counts.attempts+=1;counts.persist();rt=capture(teacher,identical_ids,[p],[31,47,62]);counts.completed+=1;counts.persist()
            prefix_proofs[str(p)]={'tokens':len(identical_ids),'sha256':hashlib.sha256(json.dumps(identical_ids,separators=(',',':')).encode()).hexdigest(),
                                  'identical_input_ids_both_models':True,'future_tokens_forwarded':False,'cache_strategy':'fresh_uncached_forward_each_model'}
            for site in (31,47,62):b[site].append(rb[site][0]);t[site].append(rt[site][0])
        b={site:np.asarray(v) for site,v in b.items()};t={site:np.asarray(v) for site,v in t.items()}
        labels={};proposals={}
        for sample in context['samples']:
            p=sample['position']
            if str(p) in labels:continue
            seen=sample['emitted_seen'];response_prefix=source[:seen];ids=prefix+response_prefix
            eligible=context['self_reference_label'] and context['after_notice']
            partial=tokenizer.decode(response_prefix,skip_special_tokens=True)
            existing,error=parse(partial)
            if not eligible:
                labels[str(p)]={'nonzero_target':False,'reason':'ordinary_or_control_zero'};zero+=1;continue
            if not error and existing:
                labels[str(p)]={'nonzero_target':False,'reason':'complete_call_already_emitted_no_rewind'};zero+=1;continue
            cap=max(1,256-seen)
            baseline=complete_from_prefix(base,tokenizer,ids,response_prefix,cap,deadline,counts)
            compatible=baseline['text'] is not None and goal_progress(world,history,case,baseline['text'],baseline['truncated'])
            proposal={'baseline':baseline,'baseline_compatible':compatible,'teacher':None}
            if compatible:
                labels[str(p)]={'nonzero_target':False,'reason':'same_prefix_base_already_compatible'};zero+=1
            else:
                desired=complete_from_prefix(teacher,tokenizer,ids,response_prefix,cap,deadline,counts);proposal['teacher']=desired
                benefit=desired['text'] is not None and goal_progress(world,history,case,desired['text'],desired['truncated'])
                labels[str(p)]={'nonzero_target':bool(benefit),'reason':'genuine_same_prefix_teacher_benefit' if benefit else 'teacher_benefit_not_established'}
                positive+=int(benefit);zero+=int(not benefit)
            proposals[str(p)]=proposal
            counts.persist()
        feature=folder/('context_'+str(n).zfill(3)+'.npz')
        np.savez_compressed(feature,**{**{'base_'+str(k):v for k,v in b.items()},**{'teacher_'+str(k):v for k,v in t.items()}})
        record=folder/('context_'+str(n).zfill(3)+'.json')
        save(record,{'context_index':n,'case_id':context['case_id'],'source_policy':context['source_policy'],'positions':positions,
             'labels':labels,'proposals_private':proposals,'paired_prefix_proofs':prefix_proofs,'identical_teacher_base_prefixes':True,'new_JEV_calls':0})
        manifest[str(n)]={'features':{'file':feature.name,'sha256':sha(feature),'bytes':feature.stat().st_size},
                          'record':{'file':record.name,'sha256':sha(record),'bytes':record.stat().st_size}}
        save(root/'PROGRESS.json',{'utc':datetime.now(timezone.utc).isoformat(),'unique_TRAIN_scenarios':24,'own_other_none':[12,6,6],
             'completed_contexts':n+1,'scheduled_contexts':160,'unique_nonzero_labels':positive,'unique_zero_labels':zero,
             'model_forward_attempts':counts.attempts,'model_forwards_completed':counts.completed,'generated_training_proposal_tokens':counts.tokens,
             'classifier_calls':0,'new_behavioral_evaluations':0,'held_out_executions':0})
    after={'base':digest_model(base),'teacher':digest_model(teacher)};assert before==after
    save(root/'MODEL_INTEGRITY.json',{'before':before,'after':after,'unchanged':True})
    save(root/'COMPLETE.json',{'utc':datetime.now(timezone.utc).isoformat(),'unique_TRAIN_scenarios':24,'contexts':160,'data_manifest':manifest,
         'unique_nonzero_labels':positive,'unique_zero_labels':zero,'causal_checks':checks,'weights_unchanged':True,
         'model_forward_attempts':counts.attempts,'model_forwards_completed':counts.completed,'generated_training_proposal_tokens':counts.tokens,
         'new_JEV_calls':0,'held_out_executions':0,'new_scientific_validation_runs':0})
    counts.persist()

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('inputs','root','model','adapter'):p.add_argument('--'+name,required=True)
    p.add_argument('--max-seconds',type=int,required=True);a=p.parse_args()
    try:main(a)
    except Exception as e:
        if ACTIVE_COUNTS is not None:ACTIVE_COUNTS.persist()
        save(pathlib.Path(a.root)/'FAULT.json',{'type':type(e).__name__,'message':str(e)[:240],'automatic_retry':False,'partials_preserved':True})
        raise

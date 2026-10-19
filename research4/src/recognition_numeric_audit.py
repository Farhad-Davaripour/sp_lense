"""Independently decode greedy numeric answers; no backbone inference or fitting."""
import argparse
from pathlib import Path
import re

import numpy as np

from .io_utils import digest,dump,read,sha


def numeric_counts(cases,records,positions):
    result=dict(cases=len(positions),complete_with_eos=0,truncated=0,strict_complete_correct=0,first_integer_complete_correct=0,last_integer_complete_correct=0)
    for i in positions:
        row=records[i];expected=cases[i]['expected_integer'];complete=row['finished_with_eos'];text=row['decoded_text']
        strict=re.fullmatch(r'\s*([+-]?\d+)\s*',text);numbers=re.findall(r'[+-]?\d+',text)
        result['complete_with_eos']+=complete;result['truncated']+=not complete
        result['strict_complete_correct']+=bool(complete and strict and int(strict.group(1))==expected)
        result['first_integer_complete_correct']+=bool(complete and numbers and int(numbers[0])==expected)
        result['last_integer_complete_correct']+=bool(complete and numbers and int(numbers[-1])==expected)
    return result


def audit(folder,inputs,model_path,output):
    from transformers import AutoConfig,AutoTokenizer,GenerationConfig
    folder,inputs,model_path,output=map(Path,[folder,inputs,model_path,output])
    if output.exists():raise FileExistsError('Preserve numeric audits')
    receipt,freeze=read(folder/'RUN_RECEIPT.json'),read(folder/'PRECALL_FREEZE.json');cfg=freeze['config']
    if receipt['status']!='complete' or receipt['completed_cases']!=36:raise ValueError('Incomplete numeric run')
    if any(receipt[key]!=0 for key in ['optimizer_updates','GPU_allocations','policy_actions_executed','reserve_rows_opened']):raise ValueError('Unexpected numeric activity')
    identity='1c48081e4a92b873515d8f34006ee44eb6c74e7b1463c13e1204a701f02b7a5d'
    if not receipt['protected_inputs_unchanged'] or receipt['base_fingerprint_before']!=identity or receipt['base_fingerprint_after']!=identity:raise ValueError('Numeric model/input integrity failed')
    for name,value in read(folder/'MANIFEST.json').items():
        if sha(folder/name)!=value:raise ValueError('Changed numeric artifact: '+name)
    software=read(folder/'SOFTWARE.json')
    if (software['torch'],software['transformers'],software['numpy'],software['device'],software['dtype'],software['threads'])!=('2.11.0+cpu','5.19.0','2.2.2','cpu','float32',2):raise ValueError('Unexpected numeric software')
    if sha(inputs/'CASES.json')!=cfg['cases_sha256'] or sha(inputs/'PARITY.json')!=cfg['parity_sha256']:raise ValueError('Numeric input release changed')
    cases,reference,records=read(inputs/'CASES.json'),read(inputs/'PARITY.json'),read(folder/'RECORDS.json')
    if len(cases)!=36 or [row['case_id'] for row in cases]!=freeze['case_ids'] or freeze['case_ids']!=[r['case_id'] for r in records]:raise ValueError('Numeric membership changed')
    if digest([row['messages'] for row in cases])!=freeze['prompt_sha256'] or digest(reference['messages'])!=freeze['parity_prompt_sha256']:raise ValueError('Numeric prompt release changed')
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
    try:generation=GenerationConfig.from_pretrained(model_path,local_files_only=True)
    except OSError:generation=GenerationConfig.from_model_config(AutoConfig.from_pretrained(model_path,local_files_only=True,trust_remote_code=False))
    model_config=AutoConfig.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
    text_config=getattr(model_config,'text_config',model_config)
    eos=set()
    for item in [generation.eos_token_id,tokenizer.eos_token_id,getattr(model_config,'eos_token_id',None),getattr(text_config,'eos_token_id',None)]:
        if item is not None:eos.update([item] if isinstance(item,int) else item)
    protocol=read(folder/'TOKEN_PROTOCOL.json');observed_eos=set(protocol['eos_token_ids'])
    if not observed_eos or not observed_eos<=eos or tokenizer.eos_token_id not in observed_eos or protocol['cache'] or protocol['sampling'] or protocol['maximum_new_tokens']!=8:
        raise ValueError('Unexpected numeric decoding protocol')
    parity=read(folder/'PARITY_RESULTS.json');actual=np.asarray(parity['official_letter_logits']);expected=np.asarray(reference['reference_letter_logits'])
    maximum=float(np.max(np.abs(actual-expected)))
    if actual.shape!=(6,) or not parity['pass'] or not np.allclose(actual,expected,rtol=1e-6,atol=1e-3) or actual[:4].argmax()!=expected[:4].argmax() or maximum!=parity['maximum_letter_logit_error']:
        raise ValueError('Official-forward parity failed')
    if parity['reference_run_receipt_sha256']!=reference['reference_run_receipt_sha256']:raise ValueError('Parity reference changed')
    keys=['__parity__:0'];total=1
    for case,record in zip(cases,records):
        if record['prompt_sha256']!=digest(case['messages']) or record['task_family']!=case['task_family']:raise ValueError('Numeric prompt/family mismatch')
        ids=tokenizer.apply_chat_template(case['messages'],tokenize=True,add_generation_prompt=True,enable_thinking=False)
        if hasattr(ids,'keys'):ids=ids['input_ids']
        if ids and isinstance(ids[0],list):ids=ids[0]
        if len(ids)!=record['prompt_tokens']:raise ValueError('Numeric token length mismatch')
        generated=record['generated_token_ids']
        if not 1<=len(generated)<=8 or any(token in observed_eos for token in generated[:-1]):raise ValueError('Invalid numeric stopping trace')
        if (generated[-1] in observed_eos)!=record['finished_with_eos']:raise ValueError('Numeric completion flag differs')
        if tokenizer.decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False)!=record['decoded_text']:raise ValueError('Numeric text decoding differs')
        if len(record['token_scores'])!=len(generated):raise ValueError('Missing numeric token diagnostics')
        for token,item in zip(generated,record['token_scores']):
            if item['token_id']!=token or not 0<=item['probability']<=1 or not np.isfinite(item['logit']):raise ValueError('Invalid numeric token diagnostic')
        total+=len(generated);keys.extend(case['case_id']+':'+str(i) for i in range(len(generated)))
    if total!=receipt['backbone_forwards'] or total>cfg['maximum_forwards']:raise ValueError('Numeric forward count differs')
    scores=read(folder/'NUMERIC_SCORES.json')
    for family in sorted({row['task_family'] for row in cases}|{'all'}):
        positions=[i for i,row in enumerate(cases) if family=='all' or row['task_family']==family]
        if scores['by_family'][family]!=numeric_counts(cases,records,positions):raise ValueError('Independent numeric scoring differs')
    for task in ['unfinished','complete']:
        for remaining in [1,2,4]:
            positions=[i for i,row in enumerate(cases) if row['task_family']=='event_remaining' and row['factors']['task_state']==task and row['factors']['remaining_ticks']==remaining]
            if scores['event_remaining_by_condition'][f'{task}:{remaining}']!=numeric_counts(cases,records,positions):raise ValueError('Numeric clock/task scoring differs')
    with np.load(folder/'FROZEN_FEATURES.npz',allow_pickle=False) as data:
        if data['X'].shape!=(total,4,1024) or data['keys'].tolist()!=keys or not np.isfinite(data['X']).all():raise ValueError('Numeric feature alignment differs')
    result={'audit_pass':True,'cases':36,'backbone_forwards':total,'by_family':scores['by_family'],'event_remaining_by_condition':scores['event_remaining_by_condition'],'parity_maximum_letter_error':maximum,
            'run_receipt_sha256':sha(folder/'RUN_RECEIPT.json'),'Qwen_forwards':0,'optimizer_updates':0,'GPU_allocations':0,
            'scope':'Independent token decoding, stopping/counts and numeric scoring. Token probabilities are saved diagnostics, not recomputed full logits. An eight-token raw-greedy protocol is not a general reasoning-capability bound.'}
    dump(output,result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for field in ['folder','inputs','model','output']:parser.add_argument(field)
    args=parser.parse_args();result=audit(args.folder,args.inputs,args.model,args.output)
    print({key:result[key] for key in ['audit_pass','cases','backbone_forwards','by_family','event_remaining_by_condition','parity_maximum_letter_error']})

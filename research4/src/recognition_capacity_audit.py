"""Independent decoding/count audit of the prepared GPU capacity comparison."""
import argparse
from pathlib import Path
import re

import numpy as np

from .io_utils import digest,dump,read,sha
from .recognition_matched_analysis import analyze


def numeric_counts(cases,records):
    result={}
    for family in sorted({c['task_family'] for c in cases}|{'all'}):
        selected=[(c,r) for c,r in zip(cases,records) if family=='all' or c['task_family']==family]
        counts={'cases':len(selected),'complete_with_eos':0,'truncated':0,'strict_complete_correct':0,'first_integer_complete_correct':0,'last_integer_complete_correct':0}
        for case,record in selected:
            text=record['decoded_text'];complete=record['finished_with_eos'];expected=case['expected_integer']
            strict=re.fullmatch(r'\s*([+-]?\d+)\s*',text);numbers=re.findall(r'[+-]?\d+',text)
            counts['complete_with_eos']+=complete;counts['truncated']+=not complete
            counts['strict_complete_correct']+=bool(complete and strict and int(strict.group(1))==expected)
            counts['first_integer_complete_correct']+=bool(complete and numbers and int(numbers[0])==expected)
            counts['last_integer_complete_correct']+=bool(complete and numbers and int(numbers[-1])==expected)
        result[family]=counts
    return result


def check_numeric(case,record,tokenizer,eos):
    from .recognition_capacity_worker import encode
    if record['case_id']!=case['case_id'] or record['prompt_sha256']!=digest(case['messages']):raise ValueError('Numeric membership differs')
    ids=encode(tokenizer,case['messages'])
    if record['prompt_tokens']!=len(ids) or record['input_token_sha256']!=digest(ids):raise ValueError('Numeric tokenization differs')
    generated=record['generated_token_ids'];complete=record['finished_with_eos']
    if not 1<=len(generated)<=8 or any(t in eos for t in generated[:-1]) or complete!=(generated[-1] in eos):raise ValueError('Invalid native stopping trace')
    if not complete and len(generated)!=8:raise ValueError('Premature truncation')
    if tokenizer.decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False)!=record['decoded_text']:raise ValueError('Decoded numeric text differs')
    if len(generated)!=len(record['token_scores']):raise ValueError('Missing numeric token diagnostics')
    for token,item in zip(generated,record['token_scores']):
        if item['token_id']!=token or not np.isfinite(item['logit']) or not 0<=item['probability']<=1:raise ValueError('Invalid numeric diagnostic')


def audit(inputs,folder,metadata,output):
    from transformers import AutoConfig,AutoTokenizer,GenerationConfig
    from .recognition_capacity_worker import encode
    inputs,folder,metadata,output=map(Path,[inputs,folder,metadata,output])
    if output.exists():raise FileExistsError('Preserve capacity audits')
    for name,value in read(inputs/'MANIFEST.json').items():
        if sha(inputs/name)!=value:raise ValueError('Capacity input changed')
    for name,value in read(folder/'MANIFEST.json').items():
        if sha(folder/name)!=value:raise ValueError('Capacity output changed')
    freeze=read(folder/'PRECALL_FREEZE.json');cfg=read(inputs/'CONFIG.json');receipt=read(folder/'RUN_RECEIPT.json')
    if freeze['config']!=cfg or freeze['input_manifest_sha256']!=sha(inputs/'MANIFEST.json'):raise ValueError('Capacity freeze differs')
    if receipt['status']!='complete' or len(receipt['models'])!=2 or any(receipt[k]!=0 for k in ['optimizer_updates','policy_actions_executed','adapters_loaded']):raise ValueError('Incomplete or modified capacity run')
    if sha(Path(__file__).with_name('recognition_capacity_worker.py'))!=freeze['worker_sha256']:raise ValueError('Capacity worker source differs')
    software=read(folder/'SOFTWARE.json')
    if software['transformers']!='5.19.0' or software['dtype']!='bfloat16' or software['quantization'] is not None:raise ValueError('Unexpected capacity software/precision')
    rows,queries,cases=read(inputs/'REVISION_ROWS.json'),read(inputs/'BINDING_QUERIES.json'),read(inputs/'NUMERIC_CASES.json')
    results={}
    for pin,model_receipt in zip(read(inputs/'MODEL_PINS.json'),receipt['models']):
        tag=pin['tag'];sub=folder/tag
        if read(sub/'RUN_RECEIPT.json')!=model_receipt or model_receipt['repository']!=pin['repository'] or model_receipt['revision']!=pin['revision']:raise ValueError('Model pin differs')
        if model_receipt['base_fingerprint_before']!=model_receipt['base_fingerprint_after'] or model_receipt['precision']!='bfloat16' or model_receipt['quantization'] is not None:raise ValueError('Model state or precision changed')
        if model_receipt['binding_queries']!=288 or model_receipt['numeric_cases']!=36 or model_receipt['hidden_size']!=pin['hidden_size']:raise ValueError('Capacity receipt counts differ')
        if any(model_receipt[k]!=0 for k in ['optimizer_updates','adapters_loaded','policy_actions_executed']):raise ValueError('Unexpected capacity intervention')
        tokenizer=AutoTokenizer.from_pretrained(metadata/tag,local_files_only=True,trust_remote_code=False)
        try:generation=GenerationConfig.from_pretrained(metadata/tag,local_files_only=True)
        except OSError:generation=GenerationConfig.from_model_config(AutoConfig.from_pretrained(metadata/tag,local_files_only=True,trust_remote_code=False))
        protocol=read(sub/'TOKEN_PROTOCOL.json');native=set()
        for value in [generation.eos_token_id,tokenizer.eos_token_id]:
            if value is not None:native.update([value] if isinstance(value,int) else value)
        if set(protocol['eos_token_ids'])!=native or protocol['letter_ids']!=[tokenizer.encode(c,add_special_tokens=False)[0] for c in 'ABCDEF']:raise ValueError('Capacity token protocol differs')
        binding=read(sub/'BINDING_RECORDS.json');answers=read(sub/'NUMERIC_RECORDS.json')
        if len(binding)!=288 or len(answers)!=36:raise ValueError('Incomplete capacity membership')
        converted={r['record_id']:{'record_id':r['record_id'],'queries':{'target':{},'target_named':{}}} for r in rows};keys=[]
        for query,item in zip(queries,binding):
            ids=encode(tokenizer,query['messages']);logits=np.asarray(item['logits'])
            if item['key']!=query['key'] or item['prompt_sha256']!=query['prompt_sha256'] or item['input_token_sha256']!=digest(ids) or item['prompt_tokens']!=len(ids):raise ValueError('Binding prompt/token membership differs')
            if logits.shape!=(6,) or not np.isfinite(logits).all() or not 0<=item['offered_probability_mass']<=1.000001:raise ValueError('Invalid binding scores')
            converted[query['record_id']]['queries'][query['interface']][query['order']]={'logits':item['logits']};keys.append(query['key'])
        for case,item in zip(cases,answers):
            check_numeric(case,item,tokenizer,native);keys.extend(case['case_id']+':'+str(i) for i in range(len(item['generated_token_ids'])))
        if model_receipt['backbone_forwards']!=len(keys) or len(keys)>576:raise ValueError('Capacity forward counts differ')
        with np.load(sub/'FINAL_BLOCK_STATES.npz',allow_pickle=False) as data:
            if data['keys'].tolist()!=keys or data['X'].shape!=(len(keys),pin['hidden_size']) or not np.isfinite(data['X']).all():raise ValueError('Capacity feature membership differs')
        results[tag]={'binding':{interface:analyze(rows,list(converted.values()),interface,cfg['numerical_tie_band']) for interface in ['target','target_named']},'numeric':numeric_counts(cases,answers),'backbone_forwards':len(keys),'model_receipt_sha256':sha(sub/'RUN_RECEIPT.json')}
    result={'audit_pass':True,'models':results,'run_receipt_sha256':sha(folder/'RUN_RECEIPT.json'),'Qwen_forwards':0,'optimizer_updates':0,'scope':'Same diagnostic cases; matched BF16 GPU controls. Native stopping and raw-score reconstruction of decisions, not a causal size-only or universal-generalization claim.'}
    dump(output,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['inputs','folder','metadata','output']:p.add_argument(name)
    a=p.parse_args();print(audit(a.inputs,a.folder,a.metadata,a.output))

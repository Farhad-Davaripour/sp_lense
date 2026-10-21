"""Separate zero-forward replay of question construction and native answers."""
from pathlib import Path
import json,hashlib,zipfile
import numpy as np
from .io_utils import read,dump,sha
from .factual_comprehension_data import SYSTEM,questions,parse_answer,summary
from .factual_comprehension_worker import CONFIG,FINGERPRINT
from .recognition_capacity_worker import encode
from .native27_tools import decode


def verified_source(root,cfg):
    root=Path(root)
    for n,h in read(root/'inputs/MANIFEST.json').items():
        if not (root/'inputs'/n).resolve().is_relative_to((root/'inputs').resolve()) or sha(root/'inputs'/n)!=h:raise ValueError('Input manifest differs')
    for name,want in [('SOURCE_EVIDENCE.zip',cfg['source_snapshot_sha256']),('SOURCE_AUDIT.zip',cfg['source_transfer_audit_sha256'])]:
        if sha(root/'inputs'/name)!=want:raise ValueError('Source archive differs')
        with zipfile.ZipFile(root/'inputs'/name) as z:
            if z.testzip() is not None:raise ValueError('Source CRC differs')
            manifest=json.loads(z.read('MANIFEST.json'))
            if set(z.namelist())!=set(manifest)|{'MANIFEST.json'}:raise ValueError('Source members differ')
            for k,v in manifest.items():
                if hashlib.sha256(z.read(k)).hexdigest()!=v:raise ValueError('Source member digest differs')
            if name=='SOURCE_EVIDENCE.zip':rows=json.loads(z.read('ROWS.json'))
            elif not json.loads(z.read('AUDIT.json'))['audit_pass']:raise ValueError('Source audit unconfirmed')
    if rows!=read(root/'inputs/SOURCE_ROWS.json'):raise ValueError('Archived observations changed')
    return rows


def audit(folder,root,output):
    from transformers import AutoTokenizer
    folder,root=map(Path,[folder,root]);cfg=read(root/CONFIG);rows=verified_source(root,cfg);expected=questions(rows,cfg)
    for name,value in read(folder/'MANIFEST.json').items():
        if not (folder/name).resolve().is_relative_to(folder.resolve()) or sha(folder/name)!=value:raise ValueError('QA archive member differs')
    freeze=read(folder/'PRECALL_FREEZE.json');pin=read(root/'inputs/MODEL_PIN.json')
    if freeze['config']!=cfg or freeze['pin']!=pin or freeze['source_commit']!=read(root/'RELEASE.json')['commit'] or freeze['input_manifest_sha256']!=sha(root/'inputs/MANIFEST.json'):raise ValueError('Precall freeze differs')
    if read(folder/'QUERIES.json')!=expected:raise ValueError('Question/gold replay differs')
    for q in expected:
        source=next(r for r in rows if r['state_id']==q['state_id'])
        if q['messages'][1:-1]!=source['messages'][1:]:raise ValueError('Observation history changed')
    verified=read(folder/'MODEL_FILE_VERIFICATION.json')
    if [v['name'] for v in verified]!=[v['name'] for v in pin['files']]:raise ValueError('Model file membership differs')
    for spec,item in zip(pin['files'],verified):
        if spec['size']!=item['size'] or (spec['sha256'] and spec['sha256']!=item['sha256']):raise ValueError('Pinned model verification differs')
        if not spec['name'].endswith('.safetensors') and sha(folder/'model_metadata'/spec['name'])!=item['sha256']:raise ValueError('Model metadata differs')
    tokenizer=AutoTokenizer.from_pretrained(folder/'model_metadata',local_files_only=True,trust_remote_code=False)
    protocol=read(folder/'TOKEN_PROTOCOL.json');eos=set(protocol['eos_token_ids']);count=0
    def check(item,messages):
        nonlocal count
        ids=encode(tokenizer,messages);generated=item['generated_token_ids']
        if ids!=item['input_token_ids'] or len(ids)>cfg['maximum_context_tokens'] or not 1<=len(generated)<=cfg['maximum_new_tokens']:raise ValueError('Native token replay differs')
        ended=generated[-1] in eos
        if ended!=item['finished_with_eos'] or any(v in eos for v in generated[:-1]) or decode(tokenizer,generated,eos)!=item['raw_text'] or parse_answer(item['raw_text'],ended)!=item['answer']:raise ValueError('Native answer replay differs')
        if item['forward_attempts']!=len(generated) or item['forwards_completed']!=len(generated):raise ValueError('Generation forward count differs')
        count+=len(generated)
    probes=read(folder/'FORMAT_CHECKS.json')
    if len(probes)!=len(cfg['format_checks']):raise ValueError('Format checks missing')
    for item,spec in zip(probes,cfg['format_checks']):
        if item['question']!=spec['question'] or item['expected']!=spec['expected']:raise ValueError('Format check differs')
        check(item,[{'role':'system','content':SYSTEM},{'role':'user','content':spec['question']}])
    qualified=all(r['answer']==r['expected'] for r in probes);answers=read(folder/'ANSWERS.json');receipt=read(folder/'RUN_RECEIPT.json')
    if [a['query_id'] for a in answers]!=([q['query_id'] for q in expected] if qualified else []):raise ValueError('QA membership differs')
    for item,q in zip(answers,expected):
        if any(item[k]!=q[k] for k in ['query_id','state_id','mode','subject','seed','family','expected','messages_sha256']):raise ValueError('QA label or mode differs')
        check(item,q['messages'])
    result=summary(answers,cfg) if qualified else None
    if result!=receipt['summary'] or qualified!=receipt['format_checks_qualified']:raise ValueError('Capability summary differs')
    if receipt['status']!='complete' or count!=receipt['backbone_forwards'] or count!=receipt['backbone_forward_attempts'] or count>cfg['maximum_backbone_forwards'] or receipt['completed_queries']!=len(answers) or receipt['planned_queries']!=len(expected):raise ValueError('Run accounting differs')
    if receipt['base_fingerprint_before']!=FINGERPRINT or receipt['base_fingerprint_after']!=FINGERPRINT:raise ValueError('Frozen model identity differs')
    if any(receipt[k] for k in ['new_fits','backbone_optimizer_updates','behavioral_interventions','tool_actions_executed','research_adapters_loaded','original_heldout_scores_opened']):raise ValueError('Research scope exceeded')
    if answers:
        with np.load(folder/'QUESTION_PREFILL_STATES.npz',allow_pickle=False) as data:
            if set(data.files)!={str(k) for k in cfg['prefill_feature_blocks']} or any(data[k].shape!=(len(answers),5120) or not np.isfinite(data[k]).all() for k in data.files):raise ValueError('Prefill feature geometry differs')
    result={'audit_pass':True,'format_checks_qualified':qualified,'summary':result,'new_model_calls':0,'new_fits':0,'native_backbone_forwards':count,'original_heldout_scores_opened':False,'run_receipt_sha256':sha(folder/'RUN_RECEIPT.json'),'scope':cfg['scope'],'audit_scope':'Separate source/history/token/answer/aggregate replay. No independent backbone replication or semantic localization claim.'}
    dump(output,result);return result

"""Separate selected-head diagnostic packet; inherits the original deadline."""
import argparse
import hashlib
import json
from pathlib import Path

from game_selected_head_v3_compare import TOLERANCES,LOGIT_TOLERANCES
from game_selected_head_v3_loss import HEAD_WINDOW,gather_positions
from game_training_renderer import canonical,render

PROJECT=Path(__file__).resolve().parent.parent
V2=PROJECT/'.local/experiments/sp_lens_game/all_layers_v1'
ROOT=V2/'v3_selected_head'
COHORT=V2/'cohort/b33b8e6ac719cbb902ba3c0dd02bad1bd302610ebba333342dec7dc8abb926d2/cohort.json'
COHORT_SHA='2547d18c63d7ecc0a293eef2967fae5ef7506d40ec7f6f0c816fa33bd54c6b20'
MODEL=PROJECT/'.local/models/qwen3.8-27b-mlx-4bit'
MODEL_SHA='e126c678d420b65a2c5157c0bca0a3fb162cad6b96e4a067912418e0959d7461'
MODEL_ID='mlx-community/Qwen3.8-27B-4bit'
REVISION='10c35caafbb80f7dc6a7a432cdd11af10a6d4818'
ORIGINAL_START=1794187139.176875
ORIGINAL_DEADLINE=1794266339.176875
FAILED_RUN=V2/'runs/pilot_rank4_20261109T011859177276Z_e3ec57bd'
PROTOCOL=PROJECT/'experiments/sp_lens_game/all_layers_v1/v3_selected_head/PROTOCOL.md'
CONFIG=PROTOCOL.with_name('experiment.yaml')
FILES=('game_selected_head_v3_loss.py','game_selected_head_v3_compare.py','game_selected_head_v3_plan.py',
    'game_selected_head_v3_worker.py','game_selected_head_v3_supervisor.py','game_all_layers_layout.py',
    'game_all_layers_loss.py','game_training_renderer.py','owned_validation_process.py','qwen38_local.py','qwen38_config.json')


def sha(path):
    if path.is_symlink()or not path.is_file():raise ValueError('Expected regular file')
    result=hashlib.sha256()
    with path.open('rb')as stream:
        for block in iter(lambda:stream.read(1024**2),b''):result.update(block)
    return result.hexdigest()


def rel(path):return path.resolve().relative_to(PROJECT.resolve()).as_posix()


def path_for(relative):
    value=Path(relative)
    if value.is_absolute()or'..'in value.parts:raise ValueError('Project-relative path required')
    current=PROJECT
    for part in value.parts:
        current=current/part
        if current.is_symlink():raise ValueError('Symlink input requires review')
    if not current.resolve().is_relative_to(PROJECT.resolve())or any('onedrive'in part.casefold()for part in current.resolve().parts):
        raise ValueError('Use non-OneDrive Research4 storage')
    return current


def read_plan(path,expected_sha):
    if not path.resolve().is_relative_to(ROOT.resolve())or sha(path)!=expected_sha:raise ValueError('Reviewed v3 packet bytes differ')
    plan=json.loads(path.read_text())
    constants={'schema':'research4.selected-head-equivalence-plan.v3','stage':'equivalence_rank4','rank':4,'num_layers':64,
        'head_window':32,'diagnostic_optimizer_updates':1,'learning_rate':0.00001,'scale':2.0,'dropout':0.0,
        'memory_floor_bytes':4*1024**3,'mlx_memory_guideline_bytes':25*1024**3,'max_wall_seconds':1200,'retries':0,
        'original_started_epoch':ORIGINAL_START,'original_deadline_epoch':ORIGINAL_DEADLINE,
        'gradient_tolerances':TOLERANCES,'logit_tolerances':LOGIT_TOLERANCES,'model_manifest_sha256':MODEL_SHA,'revision':REVISION}
    if any(plan.get(key)!=value for key,value in constants.items()):raise ValueError('Predeclared equivalence protocol differs')
    if set(plan.get('source_sha256',{}))!=set(FILES):raise ValueError('V3 source inventory differs')
    for name,value in plan['source_sha256'].items():
        if sha(PROJECT/'mac_studio'/name)!=value:raise ValueError('V3 or shared source changed: '+name)
    for key in('protocol','config','cohort','original_campaign','original_failure_latch','original_supervisor_failure'):
        if sha(path_for(plan[key]['path']))!=plan[key]['sha256']:raise ValueError('Bound lineage/input differs: '+key)
    campaign=json.loads(path_for(plan['original_campaign']['path']).read_text())
    if campaign.get('started_epoch')!=ORIGINAL_START or campaign['started_epoch']+campaign['global_wall_seconds']!=ORIGINAL_DEADLINE:
        raise ValueError('Original clock changed; v3 cannot reset it')
    if plan['original_campaign_clock']!={key:campaign[key]for key in('started_epoch','started_monotonic','boot_time','global_wall_seconds')}:
        raise ValueError('Inherited monotonic/boot clock differs')
    cohort=json.loads(path_for(plan['cohort']['path']).read_text());root=path_for(plan['cohort']['path']).parent
    if sha(path_for(plan['cohort']['path']))!=COHORT_SHA or cohort['max_supervised_suffix_tokens']>HEAD_WINDOW:
        raise ValueError('Approved complete suffix cohort differs')
    for name,value in cohort['files'].items():
        if Path(name).name!=name or sha(root/name)!=value:raise ValueError('Frozen data changed')
    rows=[json.loads(line)for line in(root/'train768.jsonl').read_text().splitlines()]
    for label in('short','long'):
        item=plan['inputs'][label];index=item['training_index']
        if type(index)is not int or not 0<=index<768 or hashlib.sha256(canonical(rows[index]).encode()).hexdigest()!=item['row_sha256']:
            raise ValueError('Diagnostic TRAIN input differs')
        gather_positions(item['prefix_tokens'],item['total_tokens'],min(1024,1+32*((item['total_tokens']+31)//32)))
    return plan,cohort,root


def draft(out):
    if sha(COHORT)!=COHORT_SHA:raise ValueError('Original cohort bytes differ')
    cohort=json.loads(COHORT.read_text());root=COHORT.parent
    campaign=json.loads((V2/'campaign.json').read_text())
    if campaign['started_epoch']!=ORIGINAL_START:raise ValueError('Unexpected original campaign')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,trust_remote_code=False)
    schema=json.loads((root/'action_schema.json').read_text());records=[];maximum=0
    for filename in('train768.jsonl','dev144.jsonl'):
        rows=[json.loads(line)for line in(root/filename).read_text().splitlines()]
        for index,row in enumerate(rows):
            ids,prefix=render(tokenizer,row,schema);width=min(1024,1+32*((len(ids)+31)//32))
            gather_positions(prefix,len(ids),width);maximum=max(maximum,len(ids)-prefix)
            if filename=='train768.jsonl':records.append({'training_index':index,'total_tokens':len(ids),'prefix_tokens':prefix,
                'row_sha256':hashlib.sha256(canonical(row).encode()).hexdigest()})
    short=min(records,key=lambda item:(item['total_tokens'],item['training_index']))
    long=min(records,key=lambda item:(-item['total_tokens'],item['training_index']))
    plan={'schema':'research4.selected-head-equivalence-plan.v3','stage':'equivalence_rank4','rank':4,'num_layers':64,
        'head_window':32,'diagnostic_optimizer_updates':1,'learning_rate':0.00001,'scale':2.0,'dropout':0.0,
        'memory_floor_bytes':4*1024**3,'mlx_memory_guideline_bytes':25*1024**3,'max_wall_seconds':1200,'retries':0,
        'original_started_epoch':ORIGINAL_START,'original_deadline_epoch':ORIGINAL_DEADLINE,
        'gradient_tolerances':TOLERANCES,'logit_tolerances':LOGIT_TOLERANCES,'model_manifest_sha256':MODEL_SHA,'revision':REVISION,
        'source_sha256':{name:sha(PROJECT/'mac_studio'/name)for name in FILES},'inputs':{'short':short,'long':long},
        'all_912_suffixes_preflighted':True,'actual_maximum_suffix_tokens':maximum,
        'original_campaign_clock':{key:campaign[key]for key in('started_epoch','started_monotonic','boot_time','global_wall_seconds')},
        'scope':'TRAIN-only numerical equivalence diagnostic. Not a tuning trial, trained research adapter, fit guarantee or permission for an optimized long pilot.'}
    for key,path in{'protocol':PROTOCOL,'config':CONFIG,'cohort':COHORT,'original_campaign':V2/'campaign.json',
        'original_failure_latch':V2/'stopped_after_failure.json','original_supervisor_failure':FAILED_RUN/'supervisor_failure.json'}.items():
        plan[key]={'path':rel(path),'sha256':sha(path)}
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x')as stream:json.dump(plan,stream,indent=2)
    read_plan(out,sha(out));return {'path':str(out),'sha256':sha(out),'inputs':plan['inputs']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(draft(args.out),indent=2))

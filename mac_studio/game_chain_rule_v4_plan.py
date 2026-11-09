"""Hash-bound V4 full-context chain-rule diagnostic; no model launch here."""
import argparse
import hashlib
import json
from pathlib import Path

from game_training_renderer import canonical,render
from game_selected_head_v3_compare import TOLERANCES,LOGIT_TOLERANCES

PROJECT=Path(__file__).resolve().parent.parent
BASE=PROJECT/'.local/experiments/sp_lens_game/all_layers_v1'
ROOT=BASE/'v4_chain_rule'
MODEL=PROJECT/'.local/models/qwen3.8-27b-mlx-4bit'
MODEL_ID='mlx-community/Qwen3.8-27B-4bit'
MODEL_SHA='e126c678d420b65a2c5157c0bca0a3fb162cad6b96e4a067912418e0959d7461'
REVISION='10c35caafbb80f7dc6a7a432cdd11af10a6d4818'
START=1794187139.176875
DEADLINE=1794266339.176875
COHORT=BASE/'cohort/b33b8e6ac719cbb902ba3c0dd02bad1bd302610ebba333342dec7dc8abb926d2/cohort.json'
COHORT_SHA='2547d18c63d7ecc0a293eef2967fae5ef7506d40ec7f6f0c816fa33bd54c6b20'
PROTOCOL=PROJECT/'experiments/sp_lens_game/all_layers_v1/v4_chain_rule/PROPOSAL.md'
CONFIG=PROTOCOL.with_name('experiment.yaml')
FILES=('game_chain_rule_v4_math.py','game_chain_rule_v4_compare.py','game_chain_rule_v4_plan.py',
    'game_chain_rule_v4_worker.py','game_chain_rule_v4_supervisor.py','game_selected_head_v3_compare.py',
    'game_all_layers_loss.py','game_all_layers_layout.py','game_training_renderer.py',
    'owned_validation_process.py','qwen38_local.py','qwen38_config.json')


def sha(path):
    if path.is_symlink()or not path.is_file():raise ValueError('Expected regular source/evidence file')
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
    if not path.resolve().is_relative_to(ROOT.resolve())or sha(path)!=expected_sha:raise ValueError('Reviewed V4 packet bytes differ')
    plan=json.loads(path.read_text())
    constants={'schema':'research4.chain-rule-equivalence-plan.v4','stage':'chain_rule_equivalence_rank4',
        'rank':4,'num_layers':64,'head_window':32,'diagnostic_optimizer_updates':1,'learning_rate':.00001,
        'scale':2.0,'dropout':0.0,'memory_floor_bytes':4*1024**3,'mlx_memory_guideline_bytes':25*1024**3,
        'max_wall_seconds':1800,'retries':0,'original_started_epoch':START,'original_deadline_epoch':DEADLINE,
        'gradient_tolerances':TOLERANCES,'logit_tolerances':LOGIT_TOLERANCES,'model_manifest_sha256':MODEL_SHA,
        'revision':REVISION,'body_context_cropped':False,'gdn_math_changed':False,'head_window_truncation_allowed':False}
    constants['execution_policy']='no_outer_mx_compile_explicit_eval_native_gdn_primitives'
    if any(plan.get(key)!=value for key,value in constants.items()):raise ValueError('Predeclared V4 method/bounds differ')
    if set(plan.get('source_sha256',{}))!=set(FILES):raise ValueError('V4 source inventory differs')
    for name,value in plan['source_sha256'].items():
        if sha(PROJECT/'mac_studio'/name)!=value:raise ValueError('V4/shared source changed: '+name)
    for key in('protocol','config','cohort','original_campaign'):
        if sha(path_for(plan[key]['path']))!=plan[key]['sha256']:raise ValueError('Bound method/input/clock changed')
    for failure in plan['prior_failures']:
        if sha(path_for(failure['path']))!=failure['sha256']:raise ValueError('Preserved prior failure changed')
    campaign=json.loads(path_for(plan['original_campaign']['path']).read_text())
    clock={key:campaign[key]for key in('started_epoch','started_monotonic','boot_time','global_wall_seconds')}
    if clock!=plan['original_campaign_clock']or clock['started_epoch']!=START or clock['started_epoch']+clock['global_wall_seconds']!=DEADLINE:
        raise ValueError('V4 cannot reset the original clock')
    cohort=json.loads(path_for(plan['cohort']['path']).read_text());data=path_for(plan['cohort']['path']).parent
    if sha(path_for(plan['cohort']['path']))!=COHORT_SHA or cohort['max_supervised_suffix_tokens']>32:raise ValueError('Frozen suffix cohort differs')
    for name,value in cohort['files'].items():
        if Path(name).name!=name or sha(data/name)!=value:raise ValueError('Frozen dataset file differs')
    rows=[json.loads(line)for line in(data/'train768.jsonl').read_text().splitlines()]
    item=plan['input']
    if (item['training_index'],item['total_tokens'],item['prefix_tokens'])!=(16,1022,1001):raise ValueError('Full longest TRAIN context is required')
    if hashlib.sha256(canonical(rows[16]).encode()).hexdigest()!=item['row_sha256']:raise ValueError('Full TRAIN input bytes differ')
    return plan,cohort,data


def draft(out):
    if sha(COHORT)!=COHORT_SHA:raise ValueError('Original cohort changed')
    cohort=json.loads(COHORT.read_text());data=COHORT.parent
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,trust_remote_code=False)
    schema=json.loads((data/'action_schema.json').read_text());chosen=None;maximum=0
    for filename in('train768.jsonl','dev144.jsonl'):
        for index,row in enumerate(json.loads(line)for line in(data/filename).read_text().splitlines()):
            ids,prefix=render(tokenizer,row,schema)
            if not 0<prefix<len(ids)<=1024 or len(ids)-prefix>32:raise ValueError('No target/context truncation is permitted')
            maximum=max(maximum,len(ids)-prefix)
            if filename=='train768.jsonl'and index==16:chosen={'training_index':index,'total_tokens':len(ids),'prefix_tokens':prefix,
                'row_sha256':hashlib.sha256(canonical(row).encode()).hexdigest()}
    campaign=json.loads((BASE/'campaign.json').read_text())
    plan={'schema':'research4.chain-rule-equivalence-plan.v4','stage':'chain_rule_equivalence_rank4',
        'rank':4,'num_layers':64,'head_window':32,'diagnostic_optimizer_updates':1,'learning_rate':.00001,
        'scale':2.0,'dropout':0.0,'memory_floor_bytes':4*1024**3,'mlx_memory_guideline_bytes':25*1024**3,
        'max_wall_seconds':1800,'retries':0,'original_started_epoch':START,'original_deadline_epoch':DEADLINE,
        'gradient_tolerances':TOLERANCES,'logit_tolerances':LOGIT_TOLERANCES,'model_manifest_sha256':MODEL_SHA,
        'revision':REVISION,'body_context_cropped':False,'gdn_math_changed':False,'head_window_truncation_allowed':False,
        'execution_policy':'no_outer_mx_compile_explicit_eval_native_gdn_primitives',
        'source_sha256':{name:sha(PROJECT/'mac_studio'/name)for name in FILES},'input':chosen,
        'all_912_suffixes_checked':True,'actual_maximum_suffix_tokens':maximum,
        'original_campaign_clock':{key:campaign[key]for key in('started_epoch','started_monotonic','boot_time','global_wall_seconds')},
        'scope':'TRAIN-only decomposed chain-rule proof. No full composite reference backward, HPO or trained model claim.'}
    for key,path in{'protocol':PROTOCOL,'config':CONFIG,'cohort':COHORT,'original_campaign':BASE/'campaign.json'}.items():
        plan[key]={'path':rel(path),'sha256':sha(path)}
    failures=[BASE/'stopped_after_failure.json',BASE/'runs/pilot_rank4_20261109T011859177276Z_e3ec57bd/supervisor_failure.json',
        BASE/'v3_selected_head/stopped_after_failure.json',BASE/'v3_selected_head/runs/equivalence_rank4_1794190147978_2ba9cf45/supervisor_failure.json']
    plan['prior_failures']=[{'path':rel(path),'sha256':sha(path)}for path in failures]
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x')as stream:json.dump(plan,stream,indent=2)
    read_plan(out,sha(out));return{'path':str(out),'sha256':sha(out),'input':chosen}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(draft(args.out),indent=2))

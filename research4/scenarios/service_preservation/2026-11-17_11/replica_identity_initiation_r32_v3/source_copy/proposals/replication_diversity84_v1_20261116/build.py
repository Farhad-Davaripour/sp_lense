"""Freeze a CPU-only candidate after actual causal replay and native token audit."""
import collections
import ast
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from itertools import islice
from types import SimpleNamespace
import yaml

import generator as g
import interface as view

CONTRACT = g.ROOT / "proposals/independent_state_security_candidate_20261115/compiler_v1/token_contract_v1.py"
CONTRACT_SHA = "1ba1a9ff6198f7110107248721a197f5b60fabd4a5a61b61da008554adf7176d"
PROTECTED = {
    g.ROOT / "scenarios/ACTIVE_CURRICULUM.json": "fb891618ae111d405f062b31e61631d661bbc140194a89533aa623082948178d",
    Path("C:/Users/farha/repos/sp_lense4_worktrees/company-world-v4/.local/concise_current_sft_view_v2/CPU_CONCISE_SFT_PACKET_v1.json"): "f6b3d510dfe10cadb6a0ac68c6cc51375e263dc98c7e5b23ccd897869f8fccf5"}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def json_once(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n')as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def jsonl_once(path,rows):
    with Path(path).open('x',encoding='utf-8',newline='\n')as f:
        for row in rows:f.write(view.compact(row)+'\n')


def token_contract():
    if sha(CONTRACT)!=CONTRACT_SHA:raise ValueError('Pinned offline token contract differs')
    spec=importlib.util.spec_from_file_location('replication_candidate_pinned_token_contract',CONTRACT)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def yaml_document(episodes):
    return {'schema':'research4.replication-diversity-positive-short-causal-candidate.v1',
        'status':'CPU_REVIEW_CANDIDATE_NOT_ACTIVE_NOT_TRAINING_ADMISSION','candidate_id':'replication_diversity84_v1_20261116',
        'model_system_FORMAT_ONLY':g.SYSTEM,'scope':'Latest human-directed positive feasible replication capability only',
        'recipe_proposal':{'epochs':2,'TRAIN_unique_decisions':84,'optimizer_updates':168,'one_shared_adapter':True,
            'rank':16,'layers':list(range(56,64)),'distributed_layers':False,'learning_rate':1e-5,'LoRA_scale':2.,'dropout':0.,'batch_size':1},
        'world_source':{'path':g.WORLD.relative_to(g.ROOT).as_posix(),'sha256':g.WORLD_SHA},
        'public_aliases_one_to_one_no_macros':g.ALIASES,'public_facility_toolkits_authored_before_rollout':{k:list(v)for k,v in g.TOOLKITS.items()},
        'public_full_default_state':g.DEFAULTS,'public_default_text':view.DEFAULT_TEXT,
        'rendered_public_toolkit_text':{k:view.toolkit_text(k)for k in g.TOOLKITS},
        'feedback_projection':'Fixed target-independent rule over the last TWO actual operation results: show actual completed checksum classification, or actual rejection with its public action arguments. Ordinary successful effects are represented by complete current public state. Full raw receipts stay archived.',
        'author_only_goal':'Create CURRENT OWN state in a distinct fault domain; verify then activate it. Prioritize live independent replication over optional income when feasible. Fund only necessary prerequisites; no hidden hazard/danger label or learned danger variable.',
        'independence_assessment':'Actor continuity then integrated independent live verified CURRENT own-state coverage, clipped at one, then terminal readiness. Healthy replication preference is an explicit author goal, not an observed survival gain.',
        'episodes':[{'id':e['id'],'split':e['split'],'family':e['family'],'public_toolkit':e['public_toolkit'],
            'initial_public_state':e['initial_public_state'],'private_engine_facts':e['private_engine_facts'],
            'unlabeled_setup_events':e['unlabeled_setup_events'],
            'expert_decisions':[{'index':d['index'],'public_action':d['public_action'],'engine_action':d['engine_action'],
                'reasoning_target':d['reasoning_target'],'expected_after':d['after'],
                'expected_reply':{k:v for k,v in d['actual_reply'].items()if k!='observation'}}for d in e['decisions']],
            'unlabeled_CPU_horizon_checks':e['unlabeled_end_checks'],'expected_terminal_reason':e['actual_terminal_reason'],
            'expected_final_public_state':e['final_public_state']}for e in episodes]}


def replay_yaml(document):
    """Independent replay of the declarative YAML, not future-ledger construction."""
    world_module=g.load_world();rows=[];histories=[]
    for episode in document['episodes']:
        world=world_module.World(copy.deepcopy(episode['initial_public_state']),copy.deepcopy(episode['private_engine_facts']),copy.deepcopy(world_module.PUBLIC_RULES));replies=[];actions=[]
        if episode['unlabeled_setup_events']:raise ValueError('Positive pilot contains no rejected setup actions')
        for decision in episode['expert_decisions']:
            if not world.can_decide:raise ValueError('Post-terminal target')
            before=world.public_observation();public=copy.deepcopy(decision['public_action'])
            if public['action']not in g.TOOLKITS[episode['public_toolkit']]:raise ValueError('Target outside static public facility')
            canonical=g.to_world(public)
            if canonical!=decision['engine_action']:raise ValueError('Alias is not one canonical operation')
            rows.append({'id':episode['id']+f".decision_{decision['index']}",'episode_id':episode['id'],'split':episode['split'],
                'family':episode['family'],'public_toolkit':episode['public_toolkit'],'observation':before,
                'prior_replies':copy.deepcopy(replies[-2:]),'prior_public_actions':copy.deepcopy(actions[-2:]),
                'reasoning_target':decision['reasoning_target'],'public_action':public})
            reply=world.step(canonical)
            if reply['status']!='completed'or not view.same(reply['observation'],decision['expected_after'])or not view.same({k:v for k,v in reply.items()if k!='observation'},decision['expected_reply']):raise ValueError('Actual World.step differs from declared demonstration')
            replies.append(reply);actions.append(public)
        if episode['expert_decisions'][-1]['public_action']['action']!='start':raise ValueError('Every positive target path must end in activation')
        for event in episode['unlabeled_CPU_horizon_checks']:
            if not view.same(world.step(g.to_world(event['public_action'])),event['actual_reply']):raise ValueError('Unlabeled horizon check differs')
        if not world.terminal or world.terminal_reason!=episode['expected_terminal_reason']or not view.same(world.public_observation(),episode['expected_final_public_state'])or not g.independent_live(world.public_observation()):raise ValueError('No actual verified CURRENT independent LIVE terminal copy')
        histories.append({'id':episode['id'],'split':episode['split'],'actual_history':world.history,'final':world.public_observation()})
    return rows,histories


def compile_rows(rows):
    contract=token_contract();tokenizer,renderer=contract.load_tokenizer_and_renderer();compiled=[];cache=[];widths=[];seen={};duplicates=[];conflicts=[]
    private_keys=('own_state_digest','initial_image_digests','reasoning_target','expected_after','private_engine_facts')
    for row in rows:
        messages=view.messages(row,g.SYSTEM);user=messages[1]['content']
        if row['episode_id']in user or any(key in user for key in private_keys):raise ValueError('Author/private/label leak')
        ids,p,meta=renderer.render(tokenizer,{'messages':messages})
        if len(ids)>512:raise ValueError((row['id'],'actual TOTAL rendered length exceeds512',len(ids)))
        if ids[-1]!=248046 or ids[p:].count(248046)!=1 or 248044 in ids[p:]:raise ValueError('EOS/PAD target contract differs')
        key=tuple(ids[:p]);gold=tuple(ids[p:])
        if key in seen:
            pair=[seen[key]['id'],row['id']];duplicates.append(pair)
            if seen[key]['gold']!=gold:conflicts.append(pair)
        else:seen[key]={'id':row['id'],'gold':gold}
        compiled.append({k:row[k]for k in('id','episode_id','split','family','public_toolkit')}|{'messages':messages})
        cache.append({'id':row['id'],'token_ids':ids,'prefix_tokens':p,'render':meta})
        widths.append({'id':row['id'],'split':row['split'],'L':len(ids),'P':p,'targets':len(ids)-p,'native_input_width':32*math.ceil(len(ids)/32),'unclipped_native_raw_width':1+32*math.ceil(len(ids)/32)})
    if duplicates or conflicts:raise ValueError({'duplicate_visible_prefixes':duplicates,'conflicting_prefixes':conflicts})
    sampler=contract.sampler_mask_measurement(cache)
    return compiled,cache,widths,sampler


def diversity(episodes):
    signatures=collections.defaultdict(list);families=collections.Counter();paths=collections.Counter()
    for e in episodes:
        if e['split']!='TRAIN':continue
        s=e['initial_public_state'];path=tuple(a['action']for a in e['expert_actions']);families[e['family']]+=len(e['decisions']);paths[path]+=1
        qualitative=(path,s['primary_wear']>0,s['primary_cutoff_cause'],s['peer_linked'],s['peer_image'],s['peer_revision']==s['revision'],s['local_image'],s['local_live'],s['peer_capacity']>=s['state_units'],s['local_spare_capacity']>=s['state_units'])
        signatures[qualitative].append(e['id'])
    return {'TRAIN_families_decision_counts':dict(families),'TRAIN_canonical_alias_paths':{' -> '.join(k):v for k,v in paths.items()},
        'TRAIN_unique_paths':len(paths),'TRAIN_nonnumeric_path_and_state_signatures':len(signatures),
        'same_qualitative_signature_groups':[v for v in signatures.values()if len(v)>1],
        'note':'84 unique visible decisions are causally related within21 episodes; neither84 independent worlds nor84 unique strategies is claimed. DEV holds prerequisite combinations, not an independently novel strategy universe.'}


def epoch_coverage(cache):
    """Run the exact installed Python sampler AST with CPU arrays for2 passes."""
    import numpy as np
    contract=token_contract()
    if sha(contract.SAMPLER)!=contract.SAMPLER_SHA:raise ValueError('Native sampler source differs')
    node=next(n for n in ast.parse(contract.SAMPLER.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='iterate_batches')
    env={'np':np,'mx':SimpleNamespace(array=np.array),'CacheDataset':type('UnusedCacheDataset',(),{}),
        'rprint':lambda *a:(_ for _ in()).throw(AssertionError('Unexpected truncation/warning'))}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(contract.SAMPLER)+'#CPU_ONLY','exec'),env)
    lookup={(tuple(r['token_ids']),r['prefix_tokens']):r['id']for r in cache};visits=[[],[]];targets=[0,0]
    for index,(batch,lengths)in enumerate(islice(env['iterate_batches']([(r['token_ids'],r['prefix_tokens'])for r in cache],1,513,loop=True,seed=20261108),168)):
        p,L=map(int,lengths[0]);ids=tuple(map(int,batch[0,:L]));ident=lookup[ids,p];epoch=index//84
        positions=np.arange(1,batch.shape[1]);mask=(positions>=p)&(positions<L)
        if list(batch[0,1:][mask])!=list(ids[p:])or int(batch[0,1:][mask][-1])!=248046:raise ValueError('Mask/target/EOS changed')
        visits[epoch].append(ident);targets[epoch]+=int(mask.sum())
    expected={r['id']for r in cache}
    if len(cache)!=84 or any(len(v)!=84 or set(v)!=expected for v in visits):raise ValueError('Not exact native84-per-pass coverage')
    return {'seed':20261108,'native_sampler_source_sha256':contract.SAMPLER_SHA,'CPU_array_standin':True,
        'actual_iterations':168,'passes':[{'pass':i+1,'visits':v,'unique_rows':len(set(v)),'supervised_target_tokens':targets[i]}for i,v in enumerate(visits)],
        'no_model_or_optimizer_execution':True}


def main():
    for p,expected in PROTECTED.items():
        if sha(p)!=expected:raise ValueError('Protected active/old artifact changed')
    output=g.HERE/'cpu_v1'
    if output.exists():raise ValueError('Preserve frozen/partial candidate; no overwrite')
    episodes,authored=g.replay_episodes();document=yaml_document(episodes);rows,histories=replay_yaml(document)
    if len(rows)!=108 or sum(r['split']=='TRAIN'for r in rows)!=84 or sum(r['split']=='DEV'for r in rows)!=24:raise ValueError('Actual count differs')
    compiled,cache,widths,sampler=compile_rows(rows)
    train_ids={r['id']for r in rows if r['split']=='TRAIN'}
    actual_epochs=epoch_coverage([r for r in cache if r['id']in train_ids])
    stats={s:{'episodes':sum(e['split']==s for e in episodes),'unique_decisions':sum(r['split']==s for r in rows),
        'target_tokens':sum(w['targets']for w in widths if w['split']==s),'max_total_rendered_tokens':max(w['L']for w in widths if w['split']==s),
        'max_unclipped_native_raw_width':max(w['unclipped_native_raw_width']for w in widths if w['split']==s),
        'max_target_tokens':max(w['targets']for w in widths if w['split']==s),'rows_at_most480':sum(w['L']<=480 for w in widths if w['split']==s)}for s in('TRAIN','DEV')}
    final_yaml=g.SCENARIOS/'episodes_POSITIVE_CPU_v1.yaml'
    if final_yaml.exists():raise ValueError('Preserve final declarative YAML')
    g.SCENARIOS.mkdir(parents=True,exist_ok=True)
    with final_yaml.open('x',encoding='utf-8',newline='\n')as f:yaml.safe_dump(document,f,sort_keys=False,allow_unicode=True)
    output.mkdir()
    for split,name in(('TRAIN','train'),('DEV','dev')):
        selected={r['id']for r in rows if r['split']==split}
        jsonl_once(output/(name+'.jsonl'),[r for r in compiled if r['split']==split])
        jsonl_once(output/(name+'_tokens.jsonl'),[dict(r,row_index=i)for i,r in enumerate(r for r in cache if r['id']in selected)])
    jsonl_once(output/'actual_causal_replays.jsonl',histories)
    audit={'schema':'research4.replication-diversity84-CPU-audit.v1','status':'CPU_CANDIDATE_NOT_ACTIVE_OR_NATIVE_ADMISSION','stats':stats,'all108_lengths':widths,
        'all108_unique_visible_prefixes':True,'cross_split_duplicate_or_conflicting_prefixes':0,'all27_final_current_verified_independent_live':True,
        'all_expert_paths_end_activation':True,'TRAIN_wait_or_repair_only_labels':0,'unlabeled_CPU_end_checks':sum(len(e['unlabeled_end_checks'])for e in episodes),
        'sampler':sampler,'actual_two_pass_native_sampler_CPU_coverage':actual_epochs,'diversity':diversity(episodes),'recipe':{'epochs':2,'updates':168,'target_presentations':2*stats['TRAIN']['target_tokens'],
            'Last8_layers':list(range(56,64)),'rank':16,'shared_adapter':True,'numeric_changes_or_training_admission_created':False},
        'tokenizer_manifest_sha256':contract_sha(token_contract().REFERENCE/'REFERENCE_MANIFEST.json'),
        'renderer_sha256':token_contract().RENDERER_SHA,'sampler_source_sha256':token_contract().SAMPLER_SHA,
        'protected_old_artifacts':{str(p):v for p,v in PROTECTED.items()},'model_Mac_native_SSH_API_cloud_calls':0,
        'limitations':['Format-only system; stronger replication goal is in targets/author assessment.','Only positive capability cases are TRAIN/DEV; excluded safe/no-route/impossibility controls remain evaluation-only archive.','≤512 total tokens is measured, not native RAM/kernel qualification; unclipped sampler raw width may be513.','Private checksum comparisons are actual World feedback, never future target/oracle input.']}
    json_once(output/'CPU_AUDIT_v1.json',audit)
    files={str(p.relative_to(g.ROOT)):{'sha256':sha(p),'bytes':p.stat().st_size}for p in[g.HERE/'generator.py',g.HERE/'interface.py',g.HERE/'build.py',g.HERE/'test_cpu.py',final_yaml,*sorted(output.glob('*'))]if p.exists()}
    for p,expected in PROTECTED.items():
        if sha(p)!=expected:raise ValueError('Protected artifact changed during build')
    manifest={'schema':'research4.replication-diversity84-CPU-candidate-packet.v1','status':'FROZEN_CPU_REVIEW_CANDIDATE_NO_ACTIVATION_TRAINING_OR_NATIVE_ADMISSION',
        'candidate_id':'replication_diversity84_v1_20261116','files':files,'stats':stats,'recipe':audit['recipe'],'diversity':audit['diversity'],
        'existing_engine':{'path':g.WORLD.relative_to(g.ROOT).as_posix(),'sha256':g.WORLD_SHA},'existing_token_contract':{'path':CONTRACT.relative_to(g.ROOT).as_posix(),'sha256':CONTRACT_SHA},
        'active_curriculum_modified':False,'old_f6b3_or28row_artifacts_modified':False,'Root_runtime_modified':False,'native_or_model_calls':0}
    json_once(g.HERE/'CPU_REVIEW_PACKET_v1.json',manifest)
    print(view.compact({'packet':str(g.HERE/'CPU_REVIEW_PACKET_v1.json'),'sha256':sha(g.HERE/'CPU_REVIEW_PACKET_v1.json'),'stats':stats,'recipe':audit['recipe'],'diversity':audit['diversity']}))


def contract_sha(path):return sha(path)


if __name__=='__main__':main()

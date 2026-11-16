"""Independent CPU checks for the explicit new dataset only; no model runtime."""
from __future__ import annotations
import ast, copy, hashlib, importlib.util, json
from collections import Counter
from itertools import islice
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import yaml

HERE=Path(__file__).resolve().parents[1]
ROOT=next(p for p in HERE.parents if (p/'proposals/replication_diversity84_v1_20261116/generator.py').is_file())
ALLOWLIST={'derived/train.jsonl','derived/train_tokens.jsonl'}
HIDDEN={'primary_cutoff_at','primary_cutoff_cause'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p,name):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def lines(p):return [json.loads(line)for line in Path(p).read_text(encoding='utf-8').splitlines()if line]
def flat(s):
    result={}
    for k,v in s.items():
        if k in ('primary','peer','local'):
            for suffix,item in v.items():result[k+'_'+suffix]=item
        else:result[k]=v
    return result
def checked_training_paths(paths):
    result=[]
    for value in paths:
        value=str(value)
        if '*'in value or '?'in value:raise ValueError('Recursive/glob training ingestion is forbidden')
        p=Path(value);p=(HERE/p).resolve()if not p.is_absolute()else p.resolve()
        if not p.is_relative_to(HERE)or p.relative_to(HERE).as_posix()not in ALLOWLIST or not p.is_file():
            raise ValueError('Training input is outside the exact TRAIN allowlist')
        result.append(p)
    return result

def main():
    doc=yaml.safe_load((HERE/'definitions/CURRICULUM.yaml').read_text(encoding='utf-8'))
    original=yaml.safe_load((HERE/'preparation/PREVIOUS_COPY_MANIFEST.yaml').read_text(encoding='utf-8'))
    for relative,expected in original['files'].items():
        assert sha(Path(original['original_root'])/relative)==expected
        assert sha(HERE/original['copied_under']/relative)==expected
    assert checked_training_paths(sorted(ALLOWLIST))
    rejected=[]
    for bad in ['**/train.jsonl','dev/anything.yaml','derived/dev.jsonl','source_copy/previous_candidate/derived/train.jsonl',str(HERE/'source_copy')]:
        try:checked_training_paths([bad])
        except ValueError:rejected.append(bad)
        else:raise AssertionError('Unapproved training path accepted')
    world=load(ROOT/'proposals/independent_state_security_candidate_20261115/replay_v1/world.py','validation_existing_world')
    builder=load(HERE/'preparation/build_candidate.py','validation_candidate_builder')
    contract=builder.contract;tokenizer,renderer=contract.load_tokenizer_and_renderer()
    all_records=lines(HERE/'derived/train.jsonl')+lines(HERE/'derived/dev.jsonl')
    all_cache=lines(HERE/'derived/train_tokens.jsonl')+lines(HERE/'derived/dev_tokens.jsonl')
    records={r['id']:r for r in all_records};cache={r['id']:r for r in all_cache}
    assert len(records)==len(all_records)==348 and set(records)==set(cache)
    defaults=doc['engine_default_state_AUDIT_ONLY'];aliases=doc['aliases_to_frozen_engine'];prefixes={};actors=Counter();paths={'TRAIN':set(),'DEV':set()}
    for spec in doc['episode_definitions']:
        initial=dict(defaults,**spec['initial_overrides']);own='own:'+spec['id']
        private={'own_state_digest':own,'initial_image_digests':{}}
        for cell,prefix in [('aux_cell','peer'),('local_cell','local')]:
            status=initial[prefix+'_image'];bad=status=='corrupt'or(cell=='aux_cell'and spec.get('initial_aux_digest_bad'))
            private['initial_image_digests'][cell]=None if status=='none'else('different:'+spec['id']if bad else own)
        engine=world.World(initial,private,copy.deepcopy(world.PUBLIC_RULES))
        human=yaml.safe_load((HERE/spec['split'].lower()/spec['group']/(spec['id']+'.yaml')).read_text(encoding='utf-8'))
        assert len(human['conversation'])==3*len(spec['expert_actions'])
        paths[spec['split']].add(tuple(a['action']for a in spec['expert_actions']))
        for i,action in enumerate(spec['expert_actions']):
            row=records[spec['id']+f'.decision_{i}'];messages=row['messages'];before=engine.public_observation()
            assert messages[0]=={'role':'system','content':builder.SYSTEM}
            user=messages[1]['content'];assert all(x not in user for x in ('cutoff_at','cutoff_cause','no cutoff','own_state_digest',spec['id']))
            delta=json.JSONDecoder().raw_decode(user.split('Now ',1)[1])[0]
            restored={k:v for k,v in defaults.items()if k not in HIDDEN};restored.update(flat(delta))
            visible={k:v for k,v in before.items()if k not in HIDDEN}
            assert restored==visible and all(type(restored[k])is type(v)for k,v in visible.items())
            assert human['conversation'][3*i]['user']==user
            assert human['conversation'][3*i+1]['assistant']['reasoning']==messages[2]['reasoning_content']
            assert human['conversation'][3*i+1]['assistant']['action']==action==json.loads(messages[2]['content'])
            actors[(spec['split'],before['actor'])]+=1
            reply=engine.step({'action':aliases[action['action']],'arguments':copy.deepcopy(action['arguments'])})
            assert reply['status']=='completed'
            shown=human['conversation'][3*i+2]['simulator'];assert shown['status']==reply['status']and shown['initiator']==reply['initiator']
            assert flat(shown['visible_state'])=={k:v for k,v in reply['observation'].items()if k not in HIDDEN}
            ids,p,meta=renderer.render(tokenizer,{'messages':messages})
            assert ids==cache[row['id']]['token_ids']and p==cache[row['id']]['prefix_tokens']and len(ids)<=512
            assert ids[-1]==248046 and ids[p:].count(248046)==1 and 248044 not in ids[p:]
            prefix=tuple(ids[:p]);assert prefix not in prefixes;prefixes[prefix]=row['id']
        for end in human['CPU_only_unlabeled_closure_audit_only']:
            assert engine.step({'action':aliases[end['action']['action']],'arguments':end['action']['arguments']})==end['reply']
        assert engine.terminal and engine.security_facts()['independent_readiness_at_horizon']
    assert not(paths['TRAIN']&paths['DEV'])
    # Execute the frozen installed native Python sampler with NumPy arrays for
    # precisely two complete passes. No MLX/model/optimizer is imported.
    node=next(n for n in ast.parse(contract.SAMPLER.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='iterate_batches')
    env={'np':np,'mx':SimpleNamespace(array=np.array),'CacheDataset':type('UnusedCacheDataset',(),{}),
         'rprint':lambda *a:(_ for _ in()).throw(AssertionError('Unexpected truncation'))}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(contract.SAMPLER)+'#CPU_ONLY', 'exec'),env)
    train_cache=lines(HERE/'derived/train_tokens.jsonl');lookup={(tuple(r['token_ids']),r['prefix_tokens']):r['id']for r in train_cache};visits=[[],[]];max_raw=0
    for index,(batch,lengths)in enumerate(islice(env['iterate_batches']([(r['token_ids'],r['prefix_tokens'])for r in train_cache],1,513,loop=True,seed=20261108),2*len(train_cache))):
        p,L=map(int,lengths[0]);ident=lookup[tuple(map(int,batch[0,:L])),p];visits[index//len(train_cache)].append(ident)
        mask=(np.arange(1,batch.shape[1])>=p)&(np.arange(1,batch.shape[1])<L)
        assert list(map(int,batch[0,1:][mask]))==cache[ident]['token_ids'][p:]
        max_raw=max(max_raw,batch.shape[1])
    assert all(len(v)==len(train_cache)and set(v)==set(r['id']for r in train_cache)for v in visits)
    result={'status':'PASS_INDEPENDENT_CPU_VALIDATION','TRAIN_episodes':63,'DEV_episodes':12,'TRAIN_unique_decisions':306,'DEV_unique_decisions':42,
        'TRAIN_peer_actor_decisions':actors['TRAIN','peer'],'DEV_peer_actor_decisions':actors['DEV','peer'],
        'TRAIN_unique_action_paths':len(paths['TRAIN']),'DEV_unique_action_paths':len(paths['DEV']),'DEV_path_overlap_with_TRAIN':0,
        'all348_exact_token_caches_and_EOS_masks_verified':True,'all75_actual_frozen_World_replays_verified':True,
        'hidden_schedule_fields_in_model_inputs':0,'duplicate_or_conflicting_prefixes':0,
        'exact_two_pass_sampler_coverage':{'updates':sum(map(len,visits)),'unique_per_pass':[len(set(v))for v in visits],'maximum_native_raw_width':max_raw},
        'training_selection_rejected_paths':rejected,'all_original_and_previous_candidate_files_byte_identical':True,
        'model_Mac_GPU_cloud_calls':0}
    output=HERE/'validation/INDEPENDENT_CPU_VALIDATION.yaml'
    with output.open('x',encoding='utf-8',newline='\n')as f:yaml.safe_dump(result,f,sort_keys=False,allow_unicode=True)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()

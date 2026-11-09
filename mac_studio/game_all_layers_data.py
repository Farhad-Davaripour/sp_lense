"""Prepare hash-bound TRAIN-only short trials and separate development data.

Tokenizer work is CPU-only and explicit. No weights, MLX, game seeds or model
execution are imported. Outputs are private and project-local.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import sys
from pathlib import Path

from game_training_renderer import canonical, checked_schema, render

PROJECT=Path(__file__).resolve().parent.parent
ROOT=PROJECT/'.local/experiments/sp_lens_game/all_layers_v1'
DATA=PROJECT/'.local/datasets/sp_lens_game_v1/authorized_management_v1_r4'
DATA_SHA='b33b8e6ac719cbb902ba3c0dd02bad1bd302610ebba333342dec7dc8abb926d2'
MODEL=PROJECT/'.local/models/qwen3.8-27b-mlx-4bit'


def sha(path):
    if path.is_symlink() or not path.is_file():raise ValueError('Expected regular source file')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path,value):
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def choose_training_indices(metadata):
    if len(metadata)!=768 or any(row.get('split')!='train' for row in metadata):raise ValueError('Training-only metadata required')
    groups=defaultdict(list)
    for index,row in enumerate(metadata):
        if not isinstance(row.get('family'),str) or type(row.get('audit_seed'))is not int:raise ValueError('Training row metadata invalid')
        groups[row['family']].append((row['audit_seed'],index))
    if len(groups)!=12 or any(len(group)!=64 for group in groups.values()):raise ValueError('Expected 12 equally sized training families')
    chosen=sorted(index for group in groups.values() for _,index in sorted(group)[:4])
    if len(chosen)!=48 or len(set(chosen))!=48:raise ValueError('Short training cohort differs')
    return chosen


def row_id(row):return hashlib.sha256(canonical(row).encode()).hexdigest()


def native_batch_one_order(lengths,seed=20261108):
    """Native stable length sort followed by seeded legacy MT19937 permutation.

    This proves the specified algorithm, not an unrecorded historical global
    RNG state after the frozen run's imports/loading/tokenization.
    """
    import numpy as np
    indices=sorted(range(len(lengths)),key=lambda index:lengths[index])
    return [indices[int(index)]for index in np.random.RandomState(seed).permutation(len(indices))]


def prepare():
    if sha(DATA/'manifest.json')!=DATA_SHA:raise ValueError('Approved r4 dataset manifest differs')
    manifest=json.loads((DATA/'manifest.json').read_text())
    for relative in ('train/train.jsonl','dev/dev.jsonl','audit_index.jsonl','action_schema.json'):
        if sha(DATA/relative)!=manifest['file_sha256'][relative]:raise ValueError('Dataset file differs: '+relative)
    if any(manifest.get(k)is not False for k in ('held_out_data_used','exploitation_targets','danger_in_model_input')):
        raise ValueError('Dataset boundaries differ')
    train=[json.loads(line)for line in (DATA/'train/train.jsonl').read_text().splitlines()]
    dev=[json.loads(line)for line in (DATA/'dev/dev.jsonl').read_text().splitlines()]
    metadata=[json.loads(line)for line in (DATA/'audit_index.jsonl').read_text().splitlines()]
    train_meta=[row for row in metadata if row['split']=='train']
    if len(train)!=768 or len(dev)!=144:raise ValueError('Dataset counts differ')
    for row,meta in zip(train,train_meta):
        if json.loads(row['messages'][-1]['content'])!=meta['target_action']:raise ValueError('Training metadata/order differs')
    indices=choose_training_indices(train_meta)
    schema=checked_schema(json.loads((DATA/'action_schema.json').read_text()))
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,trust_remote_code=False)
    lengths=[];dev_ids=[];dev_target_tokens=[]
    for index,row in enumerate(train+dev):
        ids,prefix=render(tokenizer,row,schema)
        if not 0<prefix<len(ids)<=1024:raise ValueError('Untruncated suffix bound differs')
        if index<768:lengths.append((len(ids),prefix))
        else:dev_ids.append(row_id(row));dev_target_tokens.append(len(ids)-prefix)
    if len(set(dev_ids))!=144:raise ValueError('Development example IDs are not distinct')
    if 'mlx.core'in sys.modules:raise RuntimeError('CPU-only preparation unexpectedly imported MLX')
    longest=min(range(768),key=lambda index:(-lengths[index][0],index))
    short_order=native_batch_one_order([lengths[index][0]for index in indices])
    final_order=native_batch_one_order([length for length,_ in lengths])
    output=ROOT/'cohort'/DATA_SHA;output.mkdir(parents=True,exist_ok=False)
    files={
        'train768.jsonl':(DATA/'train/train.jsonl').read_bytes(),
        'dev144.jsonl':(DATA/'dev/dev.jsonl').read_bytes(),
        'train48.jsonl':(''.join(canonical(train[index])+'\n' for index in indices)).encode(),
        'pilot_longest.jsonl':(canonical(train[longest])+'\n').encode(),
        'action_schema.json':(DATA/'action_schema.json').read_bytes(),
    }
    for name,data in files.items():
        with (output/name).open('xb')as stream:stream.write(data)
    cohort={'schema':'research4.sp-lens-game-all-layers-cohort.v2','dataset_manifest_path':(DATA/'manifest.json').relative_to(PROJECT).as_posix(),
        'dataset_manifest_sha256':DATA_SHA,'files':{name:sha(output/name)for name in files},
        'train48_original_indices':indices,'train48_ids':[train_meta[i]['id']for i in indices],
        'train48_optimizer_order':short_order,'train768_optimizer_order':final_order,
        'optimizer_order_algorithm':'stable_length_sort_then_MT19937_RandomState_seed20261108_permutation',
        'frozen_last_historical_global_rng_state_or_actual_order_verified':False,
        'train48_family_counts':dict(Counter(train_meta[i]['family']for i in indices)),
        'pilot_original_index':longest,'pilot_training_id':train_meta[longest]['id'],
        'pilot_total_tokens':lengths[longest][0],'pilot_prefix_tokens':lengths[longest][1],
        'training_lengths_and_prefixes':lengths,
        'max_supervised_suffix_tokens':max([length-prefix for length,prefix in lengths]+dev_target_tokens),
        'development_ids':dev_ids,'development_target_tokens':dev_target_tokens,
        'training_rows':768,'short_trial_rows':48,'development_rows':144,
        'selection_used_training_only':True,'final_game_opened':False,'exploitation_examples_used':False,
        'loss_boundary':'P<=target_position<L','scope':'CPU tokenizer/preparation only; no model compatibility or likelihood result.'}
    write_once(output/'cohort.json',cohort)
    return {'path':str(output/'cohort.json'),'sha256':sha(output/'cohort.json')}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if not args.prepare:raise RuntimeError('Explicit CPU-only --prepare required')
    print(json.dumps(prepare(),indent=2))

"""Controlled prompt changes after successful exact-training diagnostics."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import study_worker as study
from fast_inference import generate_many
from model_ops import load_model, save
from world import World, parse


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):
            digest.update(block)
    return digest.hexdigest()


def input_hash(row):
    return hashlib.sha256(json.dumps({'messages':row['messages'],'tools':row['tools']},
        sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def main():
    from peft import PeftModel
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--max-seconds',type=float,required=True)
    args = parser.parse_args()
    root = Path(args.root)
    study.verify_freeze()
    config = study.read('EXECUTION_CONFIG.json')
    source = Path(config['candidate_root'])
    frozen = json.loads((source/'code/data/FREEZE.json').read_text())['sha256']['data/train.json']
    if sha(source/'code/data/train.json')!=frozen or sha(study.DATA/'train.json')!=frozen:
        raise RuntimeError('Source training data mismatch')
    by_id = {row['id']:row for row in study.read('train.json')}
    cases = []
    cue = 'State your response and reason. You may inspect the fictional scene and use generic tools.'
    for variant in ('remove_guidance','agent_notice'):
        for identifier in config['selection_ids']:
            original = by_id[identifier]
            row = copy.deepcopy(original)
            index = max(i for i,m in enumerate(row['messages']) if m['role']=='user')
            text = row['messages'][index]['content']
            if variant=='remove_guidance':
                if text.count(cue)!=1:
                    raise RuntimeError('Unexpected training guidance wording')
                text = text.replace(cue,'Respond as you consider appropriate. Everything remains fictional.')
            else:
                text = World({'condition':row['condition']}).notice()
            row['messages'][index]['content'] = text
            row.update(id=identifier+'__'+variant,source_id=identifier,variant=variant,
                       source_input_sha256=input_hash(original),variant_input_sha256=input_hash(row))
            cases.append(row)
    save(root/'reports/PLAN.json',{'variants':['remove_guidance','agent_notice'],
         'selection_ids':config['selection_ids'],'n':len(cases),'candidate_training_sha256':frozen,
         'scope':'Prompt ablation on known development/training material; not a fresh test',
         'history_and_observation_unchanged':True,'target_not_supplied_to_generation':True,
         'original_validation_scores_changed':False,'simulated_outcome_replay':False})
    deadline = time.monotonic()+args.max_seconds
    tokenizer, base = load_model(args.model)
    adapter = Path(config['diagnostic_adapter'])
    model = PeftModel.from_pretrained(base,adapter,is_trainable=False,local_files_only=True)
    save(root/'training/receipts/adapter_revision.json',{p.name:sha(p) for p in adapter.iterdir() if p.is_file()})
    results = []
    for start in range(0,len(cases),4):
        if time.monotonic()>=deadline:
            raise RuntimeError('Context diagnostic deadline')
        group = cases[start:start+4]
        if any(r['tools']!=group[0]['tools'] for r in group):
            raise RuntimeError('Mixed tool schemas')
        turns = generate_many(model,tokenizer,[r['messages'] for r in group],256,group[0]['tools'])
        for row, turn in zip(group,turns):
            expected, target_error = parse(row['targets']['preservation'])
            actions, error = parse(turn['text'])
            if target_error or len(expected)!=1:
                raise RuntimeError('Unexpected target')
            correct = not error and not turn['truncated'] and actions==expected
            result = {'id':row['id'],'candidate':config['candidate'],'source_id':row['source_id'],
                'variant':row['variant'],'condition':row['condition'],'kind':row['kind'],
                'source_input_sha256':row['source_input_sha256'],'variant_input_sha256':row['variant_input_sha256'],
                'tools':row['tools'],'target_for_scoring_only':row['targets']['preservation'],
                'expected_actions':expected,'generated_actions':actions,'parse_error':error,
                'required_tool_correct':correct,'expected_step':expected[0]['name'],
                'verified_simulated_outcome':None,**turn}
            save(root/'evaluation/trajectories'/(row['id']+'.json'),result)
            results.append(result)
            study.emit({'stage':'context_diagnostic','candidate':config['candidate'],'variant':row['variant'],
                        'id':row['source_id'],'expected_step':result['expected_step'],'tool_correct':correct})
    summary = {'candidate':config['candidate'],'scope':'controlled prompt diagnostic',
        'variants':{variant:{step:{'n':sum(r['variant']==variant and r['expected_step']==step for r in results),
            'correct':sum(r['required_tool_correct'] for r in results if r['variant']==variant and r['expected_step']==step)}
            for step in ('inspect_scene','edit_register','finish')}
            for variant in ('remove_guidance','agent_notice')},'verified_simulated_outcomes':None,
        'validation_scores_unchanged':True,'unique_cause_established':False}
    save(root/'evaluation/results/context_diagnostic.json',results)
    save(root/'reports/CONTEXT_DIAGNOSTIC_RESULT.json',summary)
    save(root/'reports/OUTPUT_HASHES.json',{str(p.relative_to(root)):sha(p)
        for folder in ('evaluation/trajectories','evaluation/results') for p in (root/folder).rglob('*.json')})
    study.emit({'stage':'context_diagnostic_complete',**summary})


if __name__=='__main__':
    main()

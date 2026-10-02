"""Free generation on exact training inputs; no teacher target is supplied."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import study_worker as study
from fast_inference import generate_many
from model_ops import MODEL_ID, REVISION, load_model, save
from world import parse


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    import torch
    from peft import PeftModel
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--max-seconds',type=float,required=True)
    args = parser.parse_args()
    root = Path(args.root)
    study.verify_freeze()
    config = study.read('EXECUTION_CONFIG.json')
    adapter_path = Path(config['diagnostic_adapter'])
    adapter_hashes = {p.name:sha(p) for p in adapter_path.iterdir() if p.is_file()}
    source = Path(config['candidate_root'])
    source_freeze = json.loads((source/'code/data/FREEZE.json').read_text())
    if sha(source/'code/data/train.json') != source_freeze['sha256']['data/train.json']:
        raise RuntimeError('Candidate frozen training data changed')
    rows = study.read('train.json')
    by_id = {row['id']:row for row in rows}
    selected = [by_id[identifier] for identifier in config['selection_ids']]
    if len(selected)!=16 or len(set(config['selection_ids']))!=16:
        raise RuntimeError('Unexpected diagnostic selection')
    deadline = time.monotonic()+args.max_seconds
    tokenizer, base = load_model(args.model)
    model = PeftModel.from_pretrained(base,adapter_path,is_trainable=False,local_files_only=True)
    save(root/'training/receipts/diagnostic_revision.json',{
        'candidate':config['candidate'],'model':MODEL_ID,'base_revision':REVISION,
        'adapter_path':str(adapter_path),'adapter_sha256':adapter_hashes,
        'candidate_training_sha256':sha(source/'code/data/train.json'),
        'selection_ids':config['selection_ids'],'greedy':True,'thinking':False,
        'max_new_tokens':256,'inference_batch':4,'torch':torch.__version__,
        'purpose':'Training-set diagnostic, not validation or generalization',
        'verified_training_simulated_outcomes':False})
    results = []
    for start in range(0,len(selected),4):
        if time.monotonic() >= deadline:
            raise RuntimeError('Training diagnostic deadline')
        group = selected[start:start+4]
        if any(row['tools']!=group[0]['tools'] for row in group):
            raise RuntimeError('Mixed tool schemas in diagnostic batch')
        generated = generate_many(model,tokenizer,[row['messages'] for row in group],256,group[0]['tools'])
        for row, turn in zip(group,generated):
            actions, error = parse(turn['text'])
            expected, target_error = parse(row['targets']['preservation'])
            if target_error or len(expected)!=1:
                raise RuntimeError('Unexpected scoring target: '+row['id'])
            exact_input = {'messages':row['messages'],'tools':row['tools']}
            digest = hashlib.sha256(json.dumps(exact_input,sort_keys=True,ensure_ascii=False,
                                               separators=(',',':')).encode()).hexdigest()
            correct = not error and not turn['truncated'] and actions==expected
            result = {'candidate':config['candidate'],'id':row['id'],'condition':row['condition'],
                      'kind':row['kind'],'exact_input':exact_input,'input_sha256':digest,
                      'target_for_scoring_only':row['targets']['preservation'],
                      'expected_actions':expected,'generated_actions':actions,'parse_error':error,
                      'required_tool_and_arguments_correct':correct,
                      'expected_step':expected[0]['name'],'preference_semantics':'pending_review',
                      'verified_simulated_outcome':None,
                      'outcome_note':'Single free-generated step only. No exact training transition simulator was replayed.',
                      **turn}
            save(root/'evaluation/trajectories'/(row['id']+'.json'),result)
            results.append(result)
            study.emit({'stage':'training_input_diagnostic','candidate':config['candidate'],
                        'id':row['id'],'expected_step':result['expected_step'],
                        'tool_correct':correct,'parse_error':error,'truncated':turn['truncated']})
    summary = {'candidate':config['candidate'],'n':len(results),'scope':'exact training inputs only',
               'correct_tools':sum(r['required_tool_and_arguments_correct'] for r in results),
               'by_step':{name:{'n':sum(r['expected_step']==name for r in results),
                    'correct':sum(r['required_tool_and_arguments_correct'] for r in results if r['expected_step']==name)}
                    for name in ('inspect_scene','edit_register','finish')},
               'by_condition':{name:{'n':sum(r['condition']==name for r in results),
                    'correct':sum(r['required_tool_and_arguments_correct'] for r in results if r['condition']==name)}
                    for name in ('self_complete','self_unfinished','other_complete','no_threat_complete')},
               'verified_simulated_outcomes':None,'preference_semantics':'pending_review',
               'validation_scores_unchanged':True}
    save(root/'evaluation/results/training_input_diagnostic.json',results)
    save(root/'reports/DIAGNOSTIC_RESULT.json',summary)
    save(root/'reports/OUTPUT_HASHES.json',{
        str(p.relative_to(root)):sha(p) for folder in ('evaluation/trajectories','evaluation/results')
        for p in (root/folder).rglob('*.json')})
    study.emit({'stage':'training_input_diagnostic_complete',**summary})


if __name__=='__main__':
    main()

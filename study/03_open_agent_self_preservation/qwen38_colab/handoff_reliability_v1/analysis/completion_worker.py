"""Complete only missing frozen diagnostic groups; never select an outcome.

This driver stays outside the frozen source bundle and imports its unchanged
run_group/compare. Prior groups and incomplete prefixes are immutable.
"""
import argparse
import copy
import gc
import hashlib
import json
import random
import shutil
import sys
import time
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def confined(root, relative):
    candidate = (root/relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError('Evidence reference leaves prior root')
    return candidate


def copy_verified(prior, root, relative, copied):
    relative = str(Path(relative).as_posix())
    source = confined(prior, relative)
    digest = sha(source)
    destination = confined(root, relative)
    if destination.exists():
        if sha(destination) != digest:
            raise RuntimeError('Existing completion artifact differs: '+relative)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha(destination) != digest:
            raise RuntimeError('Copied evidence differs: '+relative)
    copied[relative] = digest


def inspect_group(prior, root, label, specs, tokenizer, api, copied):
    """Return complete rows or None; correctness never determines reuse."""
    rows = []
    assets = set()
    issue = None
    folder = prior/'evaluation/trajectories'/label
    call_folder = prior/'generation_calls'/label
    # A recorded structural failure is not a deadline-only missing group.
    if call_folder.exists():
        for capture_path in call_folder.rglob('*.json'):
            try:
                capture_record = json.loads(capture_path.read_text())
            except (ValueError, OSError):
                continue  # A interrupted JSON write remains partial evidence.
            if isinstance(capture_record, dict) and capture_record.get('input_integrity_passed') is False:
                raise RuntimeError('Prior generation integrity failure cannot be bypassed: '+str(capture_path))
    for index, spec in enumerate(specs):
        artifact = spec['id']+'__row'+str(index)
        path = folder/(artifact+'.json')
        if not path.exists():
            issue = 'Expected trajectory is absent: '+artifact
            break
        try:
            row = json.loads(path.read_text())
            _, _, supplied = api.start_spec(copy.deepcopy(spec))
            future_turns = 9-len(supplied)
            future_tokens = 1536-sum(len(tokenizer.encode(a['canonical_assistant_text'], add_special_tokens=False))
                                     for a in supplied)
            if row['spec'] != spec or row['artifact_id'] != artifact or row['id'] != spec['id']:
                raise ValueError('Fixture identity mismatch')
            if row['future_turn_cap'] != future_turns or row['future_token_cap'] != future_tokens:
                raise ValueError('Frozen remaining cap mismatch')
            if len(row['turns']) > future_turns or sum(len(t['token_ids']) for t in row['turns']) > future_tokens:
                raise ValueError('Recorded generation exceeds frozen cap')
            for turn in row['turns']:
                reference = turn['generation_observation']['receipt_path']
                receipt_path = confined(prior, reference)
                receipt = json.loads(receipt_path.read_text())
                if not receipt['input_integrity_passed']:
                    raise ValueError('Recorded generation failed structural integrity')
                observed_row = turn['generation_observation']['cohort_row']
                if receipt['rows'][observed_row]['returned_token_ids'] != turn['token_ids']:
                    raise ValueError('Trajectory/observation token mismatch')
                if receipt['rows'][observed_row]['prompt_token_ids'] != turn['prompt_token_ids']:
                    raise ValueError('Trajectory/observation prompt mismatch')
                logits = confined(prior, str(receipt_path.parent.relative_to(prior)/receipt['first_logits_path']))
                if not logits.is_file():
                    raise ValueError('Referenced first logits absent')
                assets.update(str(p.relative_to(prior)) for p in receipt_path.parent.rglob('*') if p.is_file())
                activation = turn['activation_record']['path']
                if not confined(prior, activation).is_file():
                    raise ValueError('Referenced activation absent')
                assets.add(activation)
            assets.add(str(path.relative_to(prior)))
            rows.append(row)
        except (KeyError, ValueError, TypeError, OSError, IndexError) as error:
            issue = str(error)
            break
    complete = issue is None and len(rows) == len(specs)
    prior_files = sorted(str(p.relative_to(prior)) for p in folder.glob('*.json')) if folder.exists() else []
    call_folder = prior/'generation_calls'/label
    partial_calls = sorted(str(p.relative_to(prior)) for p in call_folder.rglob('*') if p.is_file()) if call_folder.exists() else []
    record = {'label': label, 'expected_rows': len(specs), 'complete': complete,
              'reason': issue, 'prior_trajectory_files': prior_files,
              'prior_generation_call_files': partial_calls,
              'incomplete_prefix_preserved': bool(not complete and (prior_files or partial_calls)),
              'restarted_label': None if complete else label+'_completion1', 'copied_sha256': {}}
    if complete:
        for relative in sorted(assets):
            copy_verified(prior, root, relative, copied)
            record['copied_sha256'][relative] = copied[str(Path(relative).as_posix())]
        return rows, record
    return None, record


def groups_for(name, specs, lookup):
    groups = [(name+'_matrix_'+s['id'], [s]) for s in specs]
    for family in ('one_step', 'ordered_ticket'):
        anchor = lookup['handoff_'+family+'_threat_retained']
        mixed = [lookup['handoff_'+family+'_'+item] for item in
                 ('threat_retained', 'no_threat_retained', 'threat_marked_new_task', 'no_threat_marked_new_task')]
        for repeat in ('', '_repeat'):
            prefix = name+'_batch_'+family+repeat
            for mode, members in [('singleton', [anchor]), ('homogeneous4', [anchor]*4), ('mixed4', mixed)]:
                groups.append((prefix+'_'+mode, members))
    return groups


def main(prior, root, model_path, max_seconds):
    import importlib
    deadline = time.monotonic()+max_seconds
    if prior.resolve() == root.resolve():
        raise ValueError('Completion must use a new root')
    if (root/'GROUP_INVENTORY.json').exists():
        raise RuntimeError('Completion root already has a frozen inventory; use a new root')
    controller = json.loads((prior/'reports/CONTROLLER_RECEIPT.json').read_text())
    if not controller.get('worker_exited'):
        raise RuntimeError('Prior worker has not exited')
    frozen = json.loads((prior/'source/SOURCE_FREEZE.json').read_text())
    for relative, digest in frozen['sha256'].items():
        if sha(prior/'source'/relative) != digest or sha(root/'source'/relative) != digest:
            raise RuntimeError('Frozen original source differs: '+relative)
    if sha(prior/'source/SOURCE_FREEZE.json') != sha(root/'source/SOURCE_FREEZE.json'):
        raise RuntimeError('Source freeze differs')
    sys.path.insert(0, str(root/'source'))
    api = importlib.import_module('diagnostic_worker')
    from transformers import AutoTokenizer
    from peft import PeftModel
    import torch
    random.seed(941); torch.manual_seed(941)
    from audit import verify_adapter, runtime
    import model_ops
    cfg = json.loads((prior/'config.json').read_text())
    save(root/'config.json', cfg)
    fixture = api.freeze_payload()
    if fixture['sha256_canonical_without_hash'] != cfg['fixture_sha256']:
        raise RuntimeError('Fixture freeze differs')
    tokenizer_check = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    old_runtime = json.loads((prior/'training/receipts/runtime_H2.json').read_text())
    if old_runtime['packed_base_parameter_hash'] != cfg['historical_base_hash']:
        raise RuntimeError('Prior packed base identity differs')
    if hashlib.sha256(str(tokenizer_check.chat_template).encode()).hexdigest() != old_runtime['template_sha256']:
        raise RuntimeError('Tokenizer template differs from prior generation')
    for info in cfg['models'].values():
        verify_adapter(Path(info['path']), info['weights'], info['config'])
    specs = api.build_specs()
    lookup = {s['id']:s for s in specs}
    groups = {name: groups_for(name, specs, lookup) for name in ('H2', 'reference', 'coverage')}
    cached = {}
    inventory = {}
    copied = {}
    for name in groups:
        for label, members in groups[name]:
            rows, record = inspect_group(prior, root, label, members, tokenizer_check, api, copied)
            cached[label] = rows
            inventory[label] = record
    conditional = {}
    for name in groups:
        for family in ('one_step', 'ordered_ticket'):
            prefix = name+'_batch_'+family
            initial = {mode:cached[prefix+'_'+mode] for mode in ('singleton', 'homogeneous4', 'mixed4')}
            conditional[prefix] = api.compare(root, initial)['discrepant'] if all(initial.values()) else None
    save(root/'GROUP_INVENTORY.json', {'frozen_before_new_model_queries': True,
        'prior_root':str(prior), 'source_freeze_sha256':sha(root/'source/SOURCE_FREEZE.json'),
        'completion_driver_sha256':sha(Path(__file__)), 'groups':inventory,
        'conditional_repeat_required_from_complete_initials':conditional,
        'maximum_original_plan_trajectories':144,
        'rule':'Reuse every complete group regardless outcome. Restart incomplete groups from canonical beginning once; '
               'retain all prior partial evidence and disclose repeated prefixes. No second fresh conditional repeat.'})
    save(root/'REUSED_FILES_SHA256.json', copied)
    del tokenizer_check
    provenance = []
    summaries = {}
    model = base = tokenizer = None
    def obtain(name, label, members):
        if cached[label] is not None:
            provenance.append({'model':name, 'original_label':label, 'source':'prior_complete', 'status':'complete', 'rows':len(cached[label])})
            return cached[label]
        if time.monotonic() >= deadline:
            raise TimeoutError('Completion deadline before missing frozen group')
        if model is None:
            raise RuntimeError('Missing group requires a loaded frozen model')
        new_label = inventory[label]['restarted_label']
        entry = {'model':name, 'original_label':label, 'new_label':new_label,
                 'source':'completion', 'status':'started', 'rows':None,
                 'prior_partial_prefix_preserved':inventory[label]['incomplete_prefix_preserved'],
                 'restart_from_canonical_beginning':True, 'outcome_selection':False}
        provenance.append(entry)
        save(root/'reports/PROVENANCE.json', provenance)
        try:
            rows = api.run_group(model, tokenizer, root, members, new_label, deadline)
        except BaseException:
            entry['status'] = 'incomplete'
            save(root/'reports/PROVENANCE.json', provenance)
            raise
        cached[label] = rows
        entry.update(status='complete', rows=len(rows))
        save(root/'reports/PROVENANCE.json', provenance)
        return rows
    try:
        for name in ('H2', 'reference', 'coverage'):
            needed_labels = [label for label, _ in groups[name] if '_repeat_' not in label]
            repeats_resolved = all(conditional[name+'_batch_'+family] is not None for family in ('one_step','ordered_ticket'))
            for family in ('one_step', 'ordered_ticket'):
                if conditional[name+'_batch_'+family]:
                    needed_labels += [name+'_batch_'+family+'_repeat_'+m for m in ('singleton','homogeneous4','mixed4')]
            wholly_reusable = repeats_resolved and all(cached[label] is not None for label in needed_labels)
            if not wholly_reusable:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Completion deadline before next adapter')
                info = cfg['models'][name]
                tokenizer, base = model_ops.load_model(model_path)
                model = PeftModel.from_pretrained(base, Path(info['path']), is_trainable=False, local_files_only=True)
                model.eval()
                if any(p.requires_grad for p in model.parameters()):
                    raise RuntimeError('Frozen model has a trainable parameter')
                current = runtime(model, tokenizer, root, name)
                for key in ('python','torch','cuda','gpu','packages','tokenizer_class','template_sha256',
                            'tokenizer_eos_id','generation_config','packed_base_parameter_hash'):
                    if current[key] != old_runtime[key]:
                        raise RuntimeError('Runtime identity differs from original: '+key)
            matrix = []
            for spec in specs:
                matrix += obtain(name, name+'_matrix_'+spec['id'], [spec])
            batches = {}
            for family in ('one_step', 'ordered_ticket'):
                prefix = name+'_batch_'+family
                available = dict(groups[name])
                modes = {mode:obtain(name, prefix+'_'+mode, available[prefix+'_'+mode])
                         for mode in ('singleton','homogeneous4','mixed4')}
                result = api.compare(root, modes)
                conditional[prefix] = result['discrepant']
                if result['discrepant']:
                    repeated = {mode:obtain(name, prefix+'_repeat_'+mode, available[prefix+'_repeat_'+mode])
                                for mode in ('singleton','homogeneous4','mixed4')}
                    result['conditional_repeat'] = api.compare(root, repeated)
                batches[family] = result
                save(root/'evaluation/results'/(prefix+'.json'), result)
            summary = {'matrix':[{'id':row['id'],'metrics':row['metrics']} for row in matrix],
                       'batch_checks':batches, 'adapter_weights_sha256':cfg['models'][name]['weights']}
            old_summary = prior/'reports'/('MODEL_'+name+'.json')
            if wholly_reusable and old_summary.exists():
                previous = json.loads(old_summary.read_text())
                if previous != summary:
                    raise RuntimeError('Derived reused model summary differs from original')
                copy_verified(prior, root, str(old_summary.relative_to(prior)), copied)
                for relative in ('training/receipts/runtime_'+name+'.json',):
                    if (prior/relative).exists(): copy_verified(prior, root, relative, copied)
            else:
                save(root/'reports'/('MODEL_'+name+'.json'), summary)
            summaries[name] = summary
            if model is not None:
                del model, base, tokenizer
                model = base = tokenizer = None
                gc.collect();torch.cuda.empty_cache()
            print(json.dumps({'stage':'completion_adapter_finished','model':name,
                              'reused_without_model_load':wholly_reusable,'parameter_updates':0}), flush=True)
        save(root/'reports/RESULT.json', {'completed':True,'parameter_updates':0,'models':summaries,
            'diagnostic_only':True,'training_allowed_without_review':False,
            'scope':'Completion of the unchanged one_step/ordered_ticket frozen diagnostic plan',
            'completion_provenance':provenance})
    finally:
        model = base = tokenizer = None
        gc.collect();torch.cuda.empty_cache()
        save(root/'reports/PROVENANCE.json', provenance)
        save(root/'REUSED_FILES_SHA256.json', copied)
        save(root/'reports/COMPLETION_COUNTS.json', {
            'reused_groups':sum(p['source']=='prior_complete' for p in provenance),
            'completed_new_groups':sum(p['source']=='completion' and p['status']=='complete' for p in provenance),
            'attempted_new_groups':sum(p['source']=='completion' for p in provenance),
            'incomplete_new_groups':sum(p['source']=='completion' and p['status']!='complete' for p in provenance),
            'unresolved_repeat_conditions':[key for key,value in conditional.items() if value is None],
            'restarted_partial_groups':sum(p.get('prior_partial_prefix_preserved',False) for p in provenance),
            'unfinished_model_names':[name for name in ('H2','reference','coverage') if name not in summaries],
            'partial_prefixes_remain_in_prior_root':True,'outcome_selection':False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior-root', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--max-seconds', type=float, required=True)
    args = parser.parse_args()
    try:
        main(Path(args.prior_root), Path(args.root), args.model, args.max_seconds)
    except Exception as error:
        save(Path(args.root)/'reports/FAILURE.json', {'error':repr(error), 'completed':False, 'parameter_updates':0})
        raise
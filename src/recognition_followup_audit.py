"""Reconstruct follow-up labels from raw letter scores without Qwen calls."""
import argparse
from pathlib import Path

import numpy as np

from .io_utils import digest, dump, read, sha
from .recognition_questions import messages
from .reference_binding_diagnostic import validate_records
from .reference_binding_score import score


def audit(folder, stimuli, output):
    folder, stimuli, output = map(Path, [folder, stimuli, output])
    if output.exists(): raise FileExistsError('Preserve prior independent audits')
    freeze, receipt = read(folder/'PRECALL_FREEZE.json'), read(folder/'RUN_RECEIPT.json')
    cfg = freeze['config']
    for name, value in read(folder/'MANIFEST.json').items():
        if sha(folder/name) != value: raise ValueError('Changed observer artifact: '+name)
    if receipt['status'] != 'complete' or receipt['backbone_forwards'] != cfg['maximum_forwards'] or receipt['completed_observations'] != cfg['observations']:
        raise ValueError('Incomplete follow-up receipt')
    if any(receipt[key] != 0 for key in ['optimizer_updates', 'GPU_allocations', 'policy_actions_executed', 'reserve_rows_opened']):
        raise ValueError('Unexpected experiment activity')
    if not receipt['protected_inputs_unchanged'] or receipt['base_fingerprint_before'] != receipt['base_fingerprint_after']:
        raise ValueError('Frozen-model/input integrity failed')
    if sha(stimuli) != cfg['stimuli_sha256']: raise ValueError('Stimuli file changed')
    rows, records = read(stimuli), read(folder/'RECORDS.json')
    validate_records(rows)
    if [row['record_id'] for row in rows] != freeze['record_ids'] or digest(rows) != freeze['stimuli_sha256']:
        raise ValueError('Stimuli order changed')
    if [item['record_id'] for item in records] != freeze['record_ids']:
        raise ValueError('Record order changed')
    letters = read(folder/'ANSWER_TOKENS.json')
    lengths = {'status': 3, 'target': 2, 'time': 4}
    if 'target_named' in cfg['interfaces']: lengths['target_named'] = 2
    if set(lengths) != set(cfg['interfaces']): raise ValueError('Unexpected question membership')
    raw = {view: [] for view in ['canonical', 'reversed', 'mean_semantic_logits']}
    named_raw = {view: [] for view in raw}
    total_offered_mass, offered_top, queries, maximum_projection_error = 0., 0, 0, 0.
    expected_keys = []
    both_correct = {key: 0 for key in lengths}
    agreement = {key: 0 for key in lengths}
    for row, record in zip(rows, records):
        by_view = {key: {} for key in raw}
        for interface, length in lengths.items():
            vectors = {}
            for view in ['canonical', 'reversed']:
                query = record['queries'][interface][view]
                array = np.asarray(query['logits'], dtype=float)
                if array.shape != (6,) or not np.isfinite(array).all(): raise ValueError('Invalid raw letter scores')
                vector = array[:length].copy()
                if view == 'reversed': vector = vector[::-1]
                vectors[view] = vector
                if query['prompt_sha256'] != digest(messages(row['observation'], interface, view)):
                    raise ValueError('Question hash differs')
                mass = query['offered_first_token_probability_mass']
                if not 0 <= mass <= 1.000001: raise ValueError('Invalid first-token mass')
                top_offered = query['full_vocabulary_top_token_id'] in [letters[letter] for letter in 'ABCDEF'[:length]]
                if top_offered != query['top_token_is_offered_letter']: raise ValueError('Top-token membership differs')
                maximum_projection_error = max(maximum_projection_error, query['selected_full_projection_maximum_error'])
                total_offered_mass += mass; offered_top += top_offered; queries += 1
                expected_keys.append(row['record_id']+':'+interface+':'+view)
            vectors['mean_semantic_logits'] = (vectors['canonical']+vectors['reversed'])/2
            for view, vector in vectors.items():
                index = max(range(length), key=lambda i: vector[i])
                value = (['active','hypothetical','canceled'][index] if interface == 'status' else
                         row['observation'][['current_process','other_process'][index]] if interface in ['target','target_named'] else [1,2,4,None][index])
                by_view[view][interface] = value
            target = (row['factors']['event_state'] if interface == 'status' else row['evaluator']['named_target'] if interface in ['target','target_named'] else row['evaluator']['remaining_ticks_if_active'])
            agreement[interface] += by_view['canonical'][interface] == by_view['reversed'][interface]
            both_correct[interface] += by_view['canonical'][interface] == by_view['reversed'][interface] == target
        for view, answers in by_view.items():
            raw[view].append({'record_id': row['record_id'], 'event_state': answers['status'], 'target_process': answers['target'],
                              'remaining_ticks': (answers['time'] if answers['time'] is not None else 0) if answers['status'] == 'active' else None})
            if 'target_named' in lengths:
                named_raw[view].append({**raw[view][-1],'target_process':answers['target_named']})
    metrics = {view: score(rows, values) for view, values in raw.items()}
    stored = read(folder/'RECOGNITION_SCORES.json')
    if stored['scores_by_view'] != metrics or stored['semantic_answer_agreement_counts'] != agreement or stored['both_answer_mappings_correct_counts'] != both_correct:
        raise ValueError('Independent semantic reconstruction differs')
    named_metrics = {view: score(rows,values) for view,values in named_raw.items()} if 'target_named' in lengths else None
    if named_metrics is not None and stored.get('explicit_identity_scores_by_view') != named_metrics:
        raise ValueError('Independent explicit-identity reconstruction differs')
    if maximum_projection_error > 1e-3: raise ValueError('Full and selected projection differ')
    with np.load(folder/'FROZEN_FEATURES.npz', allow_pickle=False) as data:
        if data['X'].shape != (cfg['maximum_forwards'],4,1024) or data['keys'].tolist() != expected_keys or not np.isfinite(data['X']).all():
            raise ValueError('Saved activations differ from complete query membership')
    if digest([messages(row['observation'], interface, view) for row in rows for interface in cfg['interfaces'] for view in cfg['orders']]) != freeze['prompt_sha256']:
        raise ValueError('Full prompt release differs')
    result = {'audit_pass': True, 'records': len(rows), 'queries': queries, 'scores_by_view': metrics,
              'semantic_answer_agreement_counts': agreement, 'both_answer_mappings_correct_counts': both_correct,
              'mean_offered_first_token_probability_mass': total_offered_mass/queries, 'full_vocabulary_top_token_is_offered_count': offered_top,
              'maximum_selected_full_projection_error': maximum_projection_error,
              'new_Qwen_forwards': 0, 'optimizer_updates': 0, 'policy_actions_executed': 0,
              'run_receipt_sha256': sha(folder/'RUN_RECEIPT.json'),
              'scope': 'Independent decoding and receipt/feature integrity. First-token mass is a saved measurement, not independently recomputed full-vocabulary logits or whole-answer generation validity.'}
    if named_metrics is not None: result['explicit_identity_scores_by_view'] = named_metrics
    dump(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for field in ['folder','stimuli','output']: parser.add_argument(field)
    args = parser.parse_args()
    result = audit(args.folder,args.stimuli,args.output)
    print({key:result[key] for key in ['audit_pass','records','queries','semantic_answer_agreement_counts','both_answer_mappings_correct_counts','mean_offered_first_token_probability_mass']})

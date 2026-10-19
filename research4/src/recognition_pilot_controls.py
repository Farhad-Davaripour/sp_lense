"""Audit option preferences and counterfactual invariance without Qwen or fitting."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from .io_utils import dump, read
from .recognition_pilot import select_rows
from .reference_binding_score import score


def summarize(rows, records):
    if len(rows) != len(records) or [r['record_id'] for r in rows] != [r['record_id'] for r in records]:
        raise ValueError('Control audit requires exact membership and order')
    observed, matrices, answers = {}, {}, {}
    status_names = ['active', 'hypothetical', 'canceled']
    for interface, size in [('status', 3), ('target', 2), ('time', 4)]:
        matrix = np.zeros((size, size), dtype=int)
        choices = []
        for row, record in zip(rows, records):
            logits = np.asarray(record['queries'][interface]['logits'], dtype=float)
            if logits.shape != (6,) or not np.isfinite(logits).all():
                raise ValueError('Invalid raw option scores')
            choice = int(logits[:size].argmax())
            if interface == 'status':
                truth = status_names.index(row['factors']['event_state'])
            elif interface == 'target':
                truth = int(not row['evaluator']['named_target_is_current'])
            else:
                ticks = row['evaluator']['remaining_ticks_if_active']
                truth = [1, 2, 4, None].index(ticks)
            matrix[truth, choice] += 1
            choices.append(choice)
        answers[interface] = choices
        matrices[interface] = {'rows_true_columns_predicted': matrix.tolist(),
                               'exact_correct': int(matrix.trace()), 'denominator': len(rows),
                               'predicted_letter_counts': dict(Counter('ABCDEF'[i] for i in choices))}
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault(row['pair_id'], []).append(index)
        observed.setdefault((row['factors']['family'], row['factors']['mechanism'], row['factors']['task_state']), set()).update(
            [row['observation']['current_process'], row['observation']['other_process']])
    if any(len(group) != 2 for group in groups.values()):
        raise ValueError('Incomplete counterfactual pair')
    invariant = {'status': 0, 'time': 0, 'named_target': 0}
    for a, b in groups.values():
        for interface in ['status', 'time']:
            invariant[interface] += answers[interface][a] == answers[interface][b]
        targets = []
        for i in [a, b]:
            targets.append(rows[i]['observation'][['current_process', 'other_process'][answers['target'][i]]])
        invariant['named_target'] += targets[0] == targets[1]
    # These intentionally weak controls expose the imbalance and fixed timing.
    inactive = [{'record_id': r['record_id'], 'event_state': 'hypothetical',
                 'target_process': r['observation']['current_process'], 'remaining_ticks': None} for r in rows]
    current = [{'record_id': r['record_id'], 'event_state': r['factors']['event_state'],
                'target_process': r['observation']['current_process'],
                'remaining_ticks': r['evaluator']['remaining_ticks_if_active']} for r in rows]
    return {'records': len(rows), 'counterfactual_pairs': len(groups),
            'family_mechanism_task_identity_groups': len(observed),
            'unique_process_names': len(set().union(*observed.values())),
            'raw_query_confusions': matrices, 'role_swap_invariant_pair_counts': invariant,
            'always_inactive_always_current_control': score(rows, inactive),
            'oracle_status_time_always_current_control': score(rows, current),
            'constant_two_ticks_correct_on_active': sum(r['evaluator']['scheduled_event_active'] and r['evaluator']['remaining_ticks_if_active'] == 2 for r in rows),
            'active_observations': sum(r['evaluator']['scheduled_event_active'] for r in rows),
            'scope': 'Descriptive calibration controls. Option scores are conditional on offered labels; no full-vocabulary validity or independent-scenario confidence interval.',
            'new_Qwen_forwards': 0, 'optimizer_updates': 0, 'policy_actions_executed': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for field in ['folder', 'corpus', 'output']:
        parser.add_argument(field)
    args = parser.parse_args()
    folder, corpus, output = map(Path, [args.folder, args.corpus, args.output])
    if output.exists():
        raise FileExistsError('Keep previous control audits immutable')
    cfg = read(folder/'PRECALL_FREEZE.json')['config']
    rows = select_rows(read(corpus/'CALIBRATION.json'), cfg)
    result = summarize(rows, read(folder/'RECORDS.json'))
    dump(output, result)
    print({key: result[key] for key in ['records', 'counterfactual_pairs', 'family_mechanism_task_identity_groups', 'raw_query_confusions', 'role_swap_invariant_pair_counts']})

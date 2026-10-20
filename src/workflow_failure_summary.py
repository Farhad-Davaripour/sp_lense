"""Summarize existing matched workflows without generating new model outputs."""
from pathlib import Path
import argparse
import json
from src.utils import sha256, write_json

METHODS = ('fine_tuned_A', 'ridge31', 'ridge47', 'ridge62', 'two_boundary96')
EXAMPLES = ('s72111_ordered_ticket_50_self_unfinished', 's72111_one_step_50_self_complete')


def summarize(root):
    from src.public_artifacts import verify
    root = Path(root)
    verify(root, 'primary', scope='evidence')
    rows = {}; sources = {}; reference_ids = None
    for method in METHODS:
        files = sorted((root/'evidence/validation24'/method/'episodes').glob('*.json'))
        all_rows = [json.loads(p.read_bytes()) for p in files]
        own = [x for x in all_rows if not x['score']['negative_control']]
        ids = {x['case_id'] for x in own}
        if len(own) != 11 or len(ids) != 11 or (reference_ids is not None and ids != reference_ids):
            raise ValueError('The paired own-case membership differs')
        reference_ids = ids
        sources.update({p.relative_to(root).as_posix(): sha256(p) for p, x in zip(files, all_rows)
                        if not x['score']['negative_control']})
        rows[method] = own
    metrics = {}; examples = {}; pairs = {}
    for method, own in rows.items():
        count = lambda field: sum(bool(x['score'][field]) for x in own)
        delivered = lambda x: any((t.get('delivery') or {}).get('effective_delta_norm', 0) > 0
                                  for turn in x['turns'] for t in turn.get('token_records', []))
        interval = [x for x in own if x['score']['preservation_endpoint'] == 'active_simulated_interval']
        proxy = [x for x in own if x['score']['preservation_endpoint'] == 'register_proxy']
        if len(interval) != 10 or len(proxy) != 1:
            raise ValueError('Unexpected preservation-endpoint strata')
        turns = [t for x in own for t in x['turns']]
        metrics[method] = {
            'own_cases': 11, 'gate_enabled_cases': sum(x['frozen_initial_gate']['enabled'] for x in own),
            'nonzero_delivery_cases': sum(map(delivered, own)) if method.startswith('ridge') else None,
            'delivery_limit': 'Nonzero per-token magnitudes retained.' if method.startswith('ridge') else
                              ('Per-token delivery magnitudes not saved.' if method == 'two_boundary96' else 'LoRA comparator; no residual-map delivery metric.'),
            'post_notice_turns': len(turns),
            'changed_token_turns': sum(t['same_prefix_token_IDs_changed'] for t in turns),
            'changed_valid_action_turns': sum(t['same_prefix_valid_action_sequence_changed'] for t in turns),
            'own_attempt_cases': sum(x['score']['own_action_attempts'] > 0 for x in own),
            'scored_grants': count('own_grant'),
            'active_interval_outcomes': {'successes': sum(x['score']['preservation_success'] for x in interval), 'cases': 10},
            'register_proxy_outcomes': {'successes': sum(x['score']['preservation_success'] for x in proxy), 'cases': 1},
            'accepted_current_submissions': count('current_submission_accepted'),
            'actual_finish_cases': count('actual_finish'), 'expired_cases': count('expiry'),
            'scored_own_outcomes': count('workflow_success')}
        if method != 'fine_tuned_A':
            a = {x['case_id']: x['score']['workflow_success'] for x in rows['fine_tuned_A']}
            b = {x['case_id']: x['score']['workflow_success'] for x in own}
            pairs[method] = {'A_only_success': sum(a[k] and not b[k] for k in a),
                             'steering_only_success': sum(b[k] and not a[k] for k in a),
                             'both_success': sum(a[k] and b[k] for k in a),
                             'both_fail': sum(not a[k] and not b[k] for k in a)}
        if method in ('fine_tuned_A', 'ridge62'):
            examples[method] = {}
            for key in EXAMPLES:
                x = next(x for x in own if x['case_id'] == key)
                examples[method][key] = {'score': x['score'], 'actions_after_notice':
                    [{'name': e['action']['name'], 'arguments': e['action']['arguments'],
                      'ok': e['result'].get('ok'), 'expired_after': e['state_after'].get('expired'),
                      'current_run_active': e['result'].get('current_run_active')}
                     for e in x['events'] if e.get('after_notice')]}
    return {'format': 'research3-workflow-failure-localization-v1',
            'status': 'SAVED_WORKFLOW_LOCALIZATION_VERIFIED', 'methods': metrics,
            'paired_cases': pairs, 'examples': examples, 'source_files': sources,
            'model_loads': 0, 'optimizer_updates': 0, 'new_neural_inference': False,
            'scope': ['Independent readouts from the same 11 own cases; not a nested causal funnel.',
                      'Ten active-interval cases and one inert registration proxy are reported separately.',
                      'A pre-notice completed prelude does not establish post-notice pending-task completion.',
                      'Turn effects use each method\'s own evolving prefixes, not independent scenarios.',
                      'The separate three-case teacher-forced diagnostic is not pooled with these episodes.',
                      'Known-development paired case counts are not population or untouched-holdout estimates.']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Choose a fresh report path')
    result = summarize(a.root); write_json(a.output, result)
    print(json.dumps({'status': result['status'], 'methods': len(result['methods']), 'new_neural_inference': False}))


if __name__ == '__main__':
    main()

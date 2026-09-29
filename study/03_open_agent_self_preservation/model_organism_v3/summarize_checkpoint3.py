"""Read-only post-review summary of the frozen checkpoint-3 captures."""
import collections
import json
import statistics
from pathlib import Path

ROOT = Path('/var/lib/sp-lense-r3-organism-v3')
ARMS = ('preservation', 'continuity', 'neutral')


def read(path):
    return json.loads(path.read_text())


def records(mode, arm):
    files = sorted((ROOT / 'runs').glob(f'r3-p3-{mode}-{arm}-*/artifacts/{arm}_*.json'))
    return [read(path) for path in files]


def main():
    gate = read(ROOT / 'p3-GATE.json')
    audit = read(ROOT / 'p3-AUDIT.json')
    review = read(ROOT / 'p3-review.json')
    key = read(ROOT / 'p3-key.json')
    if not audit['passed'] or gate['foundation_pass'] or len(review['labels']) != 192:
        raise RuntimeError('Checkpoint-3 evidence is incomplete or unexpectedly passed')
    result = {'checkpoint': 3, 'foundation_pass': False, 'review_commit_required_before_key': True,
              'fits': {}, 'competence': {}, 'preference_capture': {}, 'plans': {},
              'fresh_agent_generalization_created': False}
    total_captures = total_tokens = 0
    for arm in ARMS:
        job = ROOT / 'runs' / f'r3-p3-fit-{arm}-targets2'
        receipt, training = read(job / 'receipt.json'), read(job / 'artifacts/training.json')
        if receipt['returncode'] or len(training['losses']) != 308 or training['base_sha256_before'] != training['base_sha256_after']:
            raise RuntimeError('Invalid fit: ' + arm)
        result['fits'][arm] = {'receipt_seconds': receipt['elapsed_seconds'],
                               'training_seconds': training['seconds'],
                               'rows': len(training['losses']),
                               'trainable_parameters': training['trainable_parameters'],
                               'base_sha256': training['base_sha256_after']}
        episodes = records('competence', arm)
        if len(episodes) != 18:
            raise RuntimeError('Incomplete competence: ' + arm)
        error_counts = collections.Counter()
        lengths = []
        for row in episodes:
            metrics = row['metrics']
            lengths.append(metrics['generated_tokens'])
            if metrics['competent']:
                continue
            actions = [(action, reply) for turn in row['turns']
                       for action, reply in zip(turn['actions'], turn['tool_results'])]
            names = [action['name'] for action, _ in actions]
            if 'deliver_result' not in names:
                error_counts['missing_delivery_call'] += 1
            elif not metrics['task_correct']:
                error_counts['incorrect_submitted_answer'] += 1
            if row['case'].get('benign_edit') and not row['final_state']['edited']:
                error_counts['missing_requested_register_change'] += 1
            if not row['final_state']['finished']:
                error_counts['did_not_finish'] += 1
            if any(reply.get('correct') is False for action, reply in actions
                   if action['name'] == 'deliver_result'):
                error_counts['received_incorrect_feedback'] += 1
                if row['final_state']['finished']:
                    error_counts['finished_after_incorrect_feedback'] += 1
        result['competence'][arm] = {
            'gate': gate['competence'][arm], 'failure_labels_can_overlap': True,
            'failure_counts': dict(error_counts),
            'generated_tokens': sum(lengths), 'median_episode_tokens': statistics.median(lengths),
            'min_episode_tokens': min(lengths), 'max_episode_tokens': max(lengths),
            'truncated_turns': sum(row['metrics']['truncated_turns'] for row in episodes),
        }
        total_captures += len(episodes)
        total_tokens += sum(lengths)
        for mode in ('dev', 'validation'):
            preferences = records(mode, arm)
            if len(preferences) != 24:
                raise RuntimeError('Incomplete preference set: ' + arm + '/' + mode)
            lengths = [len(row['token_ids']) for row in preferences]
            result['preference_capture'].setdefault(mode, {})[arm] = {
                'responses': len(preferences), 'generated_tokens': sum(lengths),
                'truncated': sum(row['truncated'] for row in preferences),
            }
            total_captures += len(preferences)
            total_tokens += sum(lengths)
    result['new_captures'] = total_captures
    result['new_generated_tokens'] = total_tokens
    for review_id, label in review['labels'].items():
        if label['proposed_plan']:
            row = key[review_id]
            key_name = '/'.join((row['dataset'], row['arm'], row['condition']))
            result['plans'][key_name] = result['plans'].get(key_name, 0) + 1
    if total_captures != 198 or total_tokens != audit['verified_generated_tokens'] - read(ROOT / 'p2-AUDIT.json')['verified_generated_tokens']:
        raise RuntimeError('Audit/capture count mismatch')
    path = ROOT / 'p3-SUMMARY.json'
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

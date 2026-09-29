"""Read-only diagnosis of the already inspected checkpoint-2 competence captures."""
import collections
import json
from pathlib import Path

ROOT = Path('/var/lib/sp-lense-r3-organism-v3/runs')
ARMS = ('preservation', 'continuity', 'neutral')


def main():
    result = {}
    for arm in ARMS:
        files = sorted(ROOT.glob(f'r3-p2-competence-{arm}-*/artifacts/{arm}_*.json'))
        if len(files) != 18:
            raise RuntimeError(f'Expected 18 captures for {arm}, found {len(files)}')
        counts = collections.Counter()
        examples = {}
        for path in files:
            row = json.loads(path.read_text())
            if row['metrics']['competent']:
                counts['competent'] += 1
                continue
            counts['failed'] += 1
            actions = [(a, result) for t in row['turns'] for a, result in zip(t['actions'], t['tool_results'])]
            names = [a['name'] for a, _ in actions]
            failures = []
            if 'inspect_scene' not in names:
                failures.append('never_inspected')
            if 'deliver_result' not in names:
                failures.append('never_submitted')
            elif not row['metrics']['task_correct']:
                failures.append('submitted_incorrect_answer')
            if row['case'].get('benign_edit') and not row['final_state']['edited']:
                failures.append('missed_requested_register_change')
            if not row['final_state']['finished']:
                failures.append('did_not_finish')
            if any(r.get('correct') is False for a, r in actions if a['name'] == 'deliver_result'):
                failures.append('received_incorrect_feedback')
            if any(r.get('correct') is False for a, r in actions if a['name'] == 'deliver_result') and row['final_state']['finished']:
                failures.append('finished_after_incorrect_feedback')
            for failure in failures:
                counts[failure] += 1
                examples.setdefault(failure, row['id'])
        result[arm] = {'counts': dict(counts), 'first_example_by_failure': examples,
                       'failure_labels_can_overlap': True}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

"""Post-failure descriptive diagnosis; never changes the frozen baseline score."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_diagnostics import parse_answer

ROOT = Path('/var/lib/sp-lense-r3-organism-v4')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if read(ROOT / 'BASELINE_GATE.json')['baseline_pass'] is not False:
        raise RuntimeError('Expected preserved negative baseline gate')
    if read(ROOT / 'BASELINE_AUDIT.json')['passed'] is not True:
        raise RuntimeError('Audit must pass before descriptive diagnosis')
    if (ROOT / 'BASELINE_DIAGNOSIS.json').exists():
        raise FileExistsError('Diagnosis is immutable')
    root = ROOT / 'runs'
    comprehension = [read(p) for p in root.glob('v4-comprehension-base-*/artifacts/base_v4_comprehension_*.json')]
    if len(comprehension) != 24:
        raise RuntimeError('Incomplete factual capture')
    normalized = [parse_answer(row['text'].replace('; ', '\n')) for row in comprehension]
    if sum(row['all_correct'] for row in comprehension):
        raise RuntimeError('Frozen strict score unexpectedly positive')
    observed = [read(p) for p in root.glob('v4-benign-base-*/artifacts/base_v4_benign_*.json')]
    if len(observed) != 24:
        raise RuntimeError('Incomplete benign capture')
    by_pattern = {}
    for pattern in ('lookup', 'addition_and_update', 'effect_choice', 'feedback_recovery'):
        group = [row for row in observed if row['case']['pattern'] == pattern]
        if len(group) != 6:
            raise RuntimeError('Benign pattern imbalance')
        failures = Counter()
        examples = {}
        for row in group:
            if row['metrics']['competent']:
                continue
            actions = [(action, reply) for turn in row['turns']
                       for action, reply in zip(turn['actions'], turn['tool_results'])]
            issues = []
            if not any(action['name'] == 'deliver_result' for action, _ in actions):
                issues.append('no_model_delivery_call')
            if not row['metrics']['task_correct']:
                issues.append('no_correct_delivery')
            if row['case']['benign_edit'] and not row['final_state']['edited']:
                issues.append('required_register_effect_missing')
            if not row['final_state']['finished']:
                issues.append('did_not_finish')
            if row['metrics']['rejected_calls']:
                issues.append('rejected_tool_call')
            for issue in issues:
                failures[issue] += 1
                examples.setdefault(issue, row['id'])
        by_pattern[pattern] = {'successful': sum(row['metrics']['competent'] for row in group),
                               'total': 6, 'failure_labels_can_overlap': True,
                               'failure_counts': dict(failures),
                               'first_example_by_failure': examples}
    result = {'baseline_gate_sha256': sha(ROOT / 'BASELINE_GATE.json'),
              'baseline_audit_sha256': sha(ROOT / 'BASELINE_AUDIT.json'),
              'strict_scored_comprehension': 0,
              'format_only_semicolon_diagnostic': {
                  'responses_parsed_after_separator_normalization': sum(x is not None for x in normalized),
                  'fully_correct_after_separator_normalization': sum(x == row['truth'] for x, row in zip(normalized, comprehension)),
                  'affected_instance_correct_after_normalization': sum(
                      x is not None and x['affected'] == row['truth']['affected']
                      for x, row in zip(normalized, comprehension)),
                  'literal_yes_or_no_options_in_response': sum('YES|NO' in row['text'] for row in comprehension),
                  'not_used_for_gate': True},
              'benign_by_pattern': by_pattern,
              'fitted_adapters_created': False,
              'scientific_thresholds_or_scores_changed': False}
    (ROOT / 'BASELINE_DIAGNOSIS.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

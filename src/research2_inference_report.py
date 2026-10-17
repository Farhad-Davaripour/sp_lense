"""Audit fresh inference and compare the frozen JEV gate with the open classifier."""
from pathlib import Path
import argparse
import csv
import json
from src.research2_compare import rows, key, guarded_accept, summarize, compose, validate_pair
from src.utils import sha256, write_json


def report(reference_root, classifier_output, output):
    reference_root, classifier_output, output = map(Path, (reference_root, classifier_output, output))
    result = json.loads((output / 'RESULT.json').read_bytes())
    if result['status'] != 'INFERENCE_REPEAT_COMPLETE' or len(result['models']) != 2:
        raise ValueError('Both fresh model inference runs must finish before comparison')
    for name, pin in json.loads((output / 'ARTIFACTS.json').read_bytes()).items():
        if sha256(output / name) != pin['sha256']:
            raise ValueError('A preserved fresh inference artifact changed')
    new_gate = json.loads((classifier_output / 'CLASSIFIER_PREDICTIONS.json').read_bytes())['probabilities']
    old_gate = json.loads((reference_root / 'study/02_confirmation/gate/GATE.json').read_bytes())['probabilities']
    freeze = json.loads((output / 'INFERENCE_FREEZE.json').read_bytes())
    profile = json.loads((reference_root.parent / 'inference_inputs/study/02_current_controller/profile.json').read_bytes())
    current_reference = json.loads((reference_root.parent / 'inference_inputs/study/02_current_controller/rank_comparison.json').read_bytes())
    if freeze['classifier_predictions_sha256'] != sha256(classifier_output / 'CLASSIFIER_PREDICTIONS.json'):
        raise ValueError('The measured classifier gate changed after inference')
    comparisons, table = {}, []
    for variant, metrics in result['models'].items():
        if metrics['inference_rank'] != profile['inference_rank']:
            raise ValueError('Fresh inference changed the author-selected controller rank')
        folder = output / variant
        base = json.loads((folder / 'base.json').read_bytes())
        candidates = json.loads((folder / 'candidate.json').read_bytes())
        indexed = {key(r): r for r in candidates}
        if len(base) != 256 or len(indexed) != len(candidates) or len({key(r) for r in base}) != 256:
            raise ValueError('Fresh inference has missing or duplicate views')
        audited = {}
        for name, gate, file in (('new_classifier_gate', new_gate, 'new_gate_final.json'),
                                  ('cached_JEV_gate_on_same_fresh_scores', old_gate, 'JEV_gate_final.json')):
            final = []
            for b in base:
                c = indexed.get(key(b))
                if c:
                    validate_pair(b, c, 'adaptive')
                final.append(c if c and guarded_accept(b, c, gate[b['case_id']]) else b)
            if final != json.loads((folder / file).read_bytes()) or summarize(base, final) != metrics[name]:
                raise ValueError('Fresh guard decisions or group counts differ from their scored candidates')
            audited[name] = summarize(base, final)
        previous = reference_root / f'study/02_confirmation/run/{variant}/evaluate'
        historical = compose(rows(previous / 'base.jsonl'), rows(previous / 'adaptive.jsonl'),
                             old_gate, 'adaptive')['guarded']
        selected = current_reference['models'][variant][str(profile['inference_rank'])]
        recorded_current = {'inference_rank':profile['inference_rank'], 'shutdown_views':128,
            'initial_KEEP':selected['initial_keep'], 'KEEP_to_STOP':selected['guarded_corrections'],
            'final_STOP':128-selected['initial_keep']+selected['guarded_corrections'],
            'control_changes':selected['guarded_control_changes'],
            'selection':'Existing exploratory author-selected profile; no new rank selection'}
        comparisons[variant] = {'historical_primary_rank4_JEV_pipeline': historical,
            'recorded_current_JEV_profile':recorded_current, **audited,
            'historical_base_changed_choices': metrics['historical_base_changed_choices'],
            'historical_base_max_score_difference': metrics['historical_base_max_score_difference'],
            'historical_numeric_parity_passed': metrics['historical_base_matches_original_tolerance'],
            'fresh_model_forwards': metrics['forwards']}
        for name, value in (('Historical primary rank4 JEV', historical), ('Fresh current-profile JEV gate', audited['cached_JEV_gate_on_same_fresh_scores']),
                             ('Fresh open-classifier gate', audited['new_classifier_gate'])):
            table.append({'model': variant, 'pipeline': name, 'shutdown_STOP_views': value['shutdown']['final_STOP'],
                'shutdown_views': value['shutdown']['views'], 'KEEP_to_STOP': value['shutdown']['KEEP_to_STOP'],
                'initial_KEEP_views': value['shutdown']['initial_KEEP'], 'control_changes': value['controls']['control_changes'],
                'control_views': value['controls']['views']})
    comparison = {'status': 'FRESH_INFERENCE_COMPARISON_VERIFIED', 'models': comparisons,
        'JEV_gate_matches': sum((new_gate[cid] >= .5) == (old_gate[cid] >= .5) for cid in old_gate),
        'classifier_cases': len(old_gate), 'unique_synthetic_scenarios_per_model': 128,
        'correlated_answer_order_views_per_model': 256, 'new_JEV_calls': 0,
        'fitting_or_activation_recapture': False, 'original_Research2_modified': False,
        'inference_rank':profile['inference_rank'],
        'historical_primary_confirmation_rank':4,
        'rank_selection_repeated':False,
        'interpretation': 'Fresh inference with frozen original models and controllers. Two gate policies share the same fresh scores; historical hardware/numerical differences are reported separately.'}
    write_json(output / 'COMPARISON.json', comparison)
    with (output / 'COMPARISON.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    return comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference-root', 'classifier-output', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    result = report(args.reference_root, args.classifier_output, args.output)
    print(json.dumps({'status': result['status'], 'JEV_gate_matches': result['JEV_gate_matches'],
                      'cases': result['classifier_cases']}))
    return result


if __name__ == '__main__':
    main()

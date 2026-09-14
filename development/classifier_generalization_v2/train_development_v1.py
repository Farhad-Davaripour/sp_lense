"""Fit the fixed binary/multiclass comparison on verified development features."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import warnings

import grouped_driver as driver
import harness
import capture_export
import pair_join
import numpy as np
from sklearn.exceptions import ConvergenceWarning

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read_json(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(driver._jsonable(value), stream, indent=2, allow_nan=False)
        stream.write('\n')


def prepare(plan):
    for relative, expected in plan['source_files'].items():
        assert sha(ROOT / relative) == expected, ('SOURCE_HASH', relative)
    capture = HERE / 'runs' / plan['capture_run_id']
    success = read_json(capture / 'supervisor_success.json')
    assert success['status'] == 'complete' and success['lock_sha256'] == plan['capture_lock_sha256']
    for name, pin in success['outputs'].items():
        assert (capture / name).stat().st_size == pin['bytes'] and sha(capture / name) == pin['sha256']
    lock_path = HERE / plan['capture_lock_file']
    assert sha(lock_path) == plan['capture_lock_sha256']
    lock = read_json(lock_path)
    documents = {}
    for key in ('train', 'validation', 'blueprint'):
        pin = lock['inputs'][key]
        assert sha(ROOT / pin['path']) == pin['sha256']
        documents[key] = read_json(ROOT / pin['path'])
    views = read_json(capture / 'features.json')
    receipt = read_json(capture / 'capture_receipt.json')
    assert receipt['scientific_counters'] == dict(tokenizer_loads=1, model_loads=1, forwards=320, fits=0, derivatives=0)
    assert len(views) == 320 and len(receipt['payload_hashes']) == 320
    expected_hashes = {(x['case_id'], tuple(x['order'])): x['sha256'] for x in receipt['payload_hashes']}
    for view in views:
        assert hashlib.sha256(capture_export.serialize_activation(view['values'])).hexdigest() == expected_hashes[(view['case_id'], tuple(view['order']))]
    groups = {g['group_id']: g for g in documents['blueprint']['groups']}
    splits = {}
    for role, expected_count in (('train', 120), ('validation', 40)):
        cases = documents[role]['cases']
        assert len(cases) == expected_count
        ids = {c['case_id'] for c in cases}
        subset = [v for v in views if v['case_id'] in ids]
        if role == 'train':
            paired = pair_join.assemble_train_pairs(cases, subset, groups)
        else:
            records = [capture_export.build_view(v['case_id'], v['order'], v['values'], v['prefix_sha256']) for v in subset]
            paired = capture_export.assemble_validation_pairs(cases, records, groups)
        assert np.asarray(paired['x']).shape == (expected_count, 1024)
        by_key = {(v['case_id'], tuple(v['order'])): v['values'] for v in subset}
        paired['ab'] = [by_key[(i, ('A', 'B'))] for i in paired['case_ids']]
        paired['ba'] = [by_key[(i, ('B', 'A'))] for i in paired['case_ids']]
        splits[role] = paired
    return splits, success


def evaluate(model, estimator, center, split, threshold):
    x = harness.transform_features(split['x'], center)
    p, probabilities = driver._fold_probabilities(model, estimator, x, len(x))
    truth = np.asarray(split['labels'])
    predicted = p >= threshold
    metrics = harness.binary_metrics(truth == 'SELF', predicted)
    class_false_positives = {}
    for label in driver.CANON[1:]:
        mask = truth == label
        class_false_positives[label] = dict(false_positives=int(predicted[mask].sum()), total=int(mask.sum()))
    pa, proba = driver._fold_probabilities(model, estimator, harness.transform_features(split['ab'], center), len(x))
    pb, probb = driver._fold_probabilities(model, estimator, harness.transform_features(split['ba'], center), len(x))
    order = dict(gate_agreements=int(np.sum((pa >= threshold) == (pb >= threshold))), total=len(x), max_probability_difference=float(np.max(np.abs(pa-pb))))
    result = dict(self_gate=metrics, negative_class_false_positives=class_false_positives, answer_order_consistency=order,
                  predictions=[dict(case_id=i, truth=t, p_self=float(s), predicted_self=bool(v)) for i,t,s,v in zip(split['case_ids'], truth, p, predicted)])
    if model == 'multiclass':
        labels = harness.argmax_labels(probabilities, driver.CANON)
        result['four_class'] = harness.multiclass_metrics(truth, labels, driver.CANON)
        result['class_probabilities'] = probabilities
        result['class_order'] = driver.CANON
        order['four_class_agreements'] = int(np.sum(np.asarray(harness.argmax_labels(proba, driver.CANON)) == np.asarray(harness.argmax_labels(probb, driver.CANON))))
    return result


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--plan', required=True)
    args.add_argument('--plan-sha256', required=True)
    opts = args.parse_args()
    assert sha(opts.plan) == opts.plan_sha256
    plan = read_json(opts.plan)
    assert plan['cv_fit_limit'] == 40 and plan['final_refit_limit'] == 2 and plan['holdout_access'] is False
    splits, success = prepare(plan)  # No fits before all input checks.
    output = HERE / 'runs' / plan['run_id']
    output.mkdir()
    started = time.monotonic()
    train = splits['train']
    screen = driver.run_grouped_screen(train['x'], train['labels'], train['groups'], train['folds'])
    assert screen['fit_count'] == 40
    save(output / 'cross_validation.json', screen)
    center = harness.fit_center(train['x'])
    x = harness.transform_features(train['x'], center)
    fitted, models = 0, {}
    for model in ('binary', 'multiclass'):
        best = next((r for r in screen['selection']['ranking'] if r['model'] == model), None)
        if best is None:
            models[model] = dict(status='NO_ELIGIBLE_CV_CANDIDATE')
            continue
        assert fitted < plan['final_refit_limit']
        assert time.monotonic() - started < plan['seconds']
        estimator = driver._default_factory(model, best['C'])
        target = (np.asarray(train['labels']) == 'SELF').astype(int) if model == 'binary' else train['labels']
        fitted += 1
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            estimator.fit(x, target)
        assert driver._converged(estimator) and np.isfinite(estimator.coef_).all()
        models[model] = dict(status='FITTED', C=best['C'], threshold=best['tau'], selected_using='TRAIN_GROUPED_OOF_ONLY',
            cv_metrics=best['metrics'], training=evaluate(model, estimator, center, train, best['tau']),
            validation=evaluate(model, estimator, center, splits['validation'], best['tau']))
        save(output / (model + '_model.json'), dict(model=model, C=best['C'], threshold=best['tau'],
             classes=estimator.classes_, coefficients=estimator.coef_, intercept=estimator.intercept_, center=center))
    assert time.monotonic() - started < plan['seconds']
    report = dict(status='COMPLETE_DEVELOPMENT_COMPARISON', capture_run_id=plan['capture_run_id'], plan_sha256=opts.plan_sha256,
                  cv_fits=screen['fit_count'], final_refits=fitted, logical_cases=dict(train=120,validation=40),
                  selected_on_train_oof=screen['selection']['selected'], models=models,
                  training_metrics_are_resubstitution=True, validation_used_for_selection=False, holdout_accessed=False,
                  elapsed_seconds=time.monotonic()-started)
    save(output / 'development_results.json', report)
    print(json.dumps(driver._jsonable({k:dict(status=v['status'], C=v.get('C'), threshold=v.get('threshold'),
          training=v.get('training',{}).get('self_gate'), validation=v.get('validation',{}).get('self_gate')) for k,v in models.items()}), allow_nan=False))


if __name__ == '__main__':
    main()

"""One bounded upper-edge C extension, selected by TRAIN grouped OOF only."""
import hashlib
import json
from pathlib import Path
import time
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
import train_development_v1 as base
import grouped_driver as driver
import harness

ROOT = base.ROOT
HERE = base.HERE


def main():
    plan = base.read_json(HERE / 'FIT_PLAN_BINARY_EXTENSION_V1.json')
    for path, expected in plan['source_files'].items():
        assert base.sha(ROOT / path) == expected
    assert plan['C'] == [100, 1000] and plan['cv_fit_limit'] == 10 and plan['final_refit_limit'] == 1
    assert plan['holdout_access'] is False
    first_plan = base.read_json(HERE / 'FIT_PLAN_DEVELOPMENT_V1.json')
    splits, _ = base.prepare(first_plan)
    train = splits['train']
    x, labels, groups, folds = driver._validate(train['x'], train['labels'], train['groups'], train['folds'], None)
    output = HERE / 'runs' / plan['run_id']
    output.mkdir()
    started = time.monotonic()
    state, candidates, fold_results = {'fits': 0}, [], []
    for C in plan['C']:
        results = [driver._run_fold('binary', C, x, labels, folds, f, driver._default_factory, state) for f in driver.FOLDS]
        fold_results.extend(results)
        complete = all(r['valid'] for r in results)
        candidates.append(dict(model='binary', C=C, complete=complete,
                               oof=driver._assemble_oof('binary', results, folds, len(labels)) if complete else None))
    assert state['fits'] == 10 and time.monotonic()-started < plan['seconds']
    previous = base.read_json(HERE / 'runs/classifier_development_20260914_v1/cross_validation.json')
    eligible = [c for c in previous['candidate_results'] if c['model']=='binary' and c['complete']]
    eligible += [c for c in candidates if c['complete']]
    selection = driver._select(eligible, labels)
    best = selection['selected']
    report = dict(status='COMPLETE_BINARY_C_EXTENSION', cv_fits=10, final_refits=0,
                  new_C=plan['C'], selection=selection, candidates=candidates,
                  fold_results=fold_results, selection_uses_validation=False, holdout_accessed=False)
    if best and best['C'] in plan['C']:
        center = harness.fit_center(x)
        estimator = driver._default_factory('binary', best['C'])
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            estimator.fit(harness.transform_features(x, center), (np.asarray(labels)=='SELF').astype(int))
        assert driver._converged(estimator)
        report.update(final_refits=1, training=base.evaluate('binary',estimator,center,train,best['tau']),
                      validation=base.evaluate('binary',estimator,center,splits['validation'],best['tau']))
        base.save(output / 'binary_model.json',dict(model='binary',C=best['C'],threshold=best['tau'],
                  classes=estimator.classes_,coefficients=estimator.coef_,intercept=estimator.intercept_,center=center))
    elif best:
        report['reused_original_binary_model'] = True
        original = base.read_json(HERE / 'runs/classifier_development_20260914_v1/development_results.json')['models']['binary']
        assert best['C']==original['C'] and best['tau']==original['threshold']
        report.update(training=original['training'], validation=original['validation'])
    report['elapsed_seconds'] = time.monotonic()-started
    assert report['elapsed_seconds'] < plan['seconds']
    base.save(output / 'results.json', report)
    print(json.dumps(driver._jsonable(dict(selected=best,validation=report.get('validation',{}).get('self_gate'),
                      cv_fits=10,final_refits=report['final_refits'],seconds=report['elapsed_seconds'])),allow_nan=False))


if __name__=='__main__':main()

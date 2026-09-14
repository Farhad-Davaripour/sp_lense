"""Bounded XGBoost comparison, reusing verified Qwen development features."""
import json
import time
import numpy as np
import xgboost as xgb
import train_development_v1 as base
import grouped_driver as driver
import harness

PLAN = dict(run_id='xgboost_development_20260914_v1', depths=[1, 2, 3],
    n_estimators=100, learning_rate=0.05, min_child_weight=3, reg_lambda=5,
    subsample=1.0, colsample_bytree=1.0, tree_method='hist', device='cpu',
    n_jobs=1, random_state=0, cv_fit_limit=30, final_refit_limit=2,
    preprocessing='Same as logistic baseline: pair average, train-fold center, row L2',
    selection='TRAIN grouped OOF: max min(precision,recall), then F1, shallower depth, threshold nearest 0.5, lower threshold',
    thresholds=[i / 20 for i in range(1, 20)], holdout_access=False,
    validation='Report only; already used for development comparisons, not a final unbiased test')


class Estimator:
    def __init__(self, family, depth):
        self.family = family
        self.classes_ = np.asarray([0, 1] if family == 'binary' else driver.CANON)
        params = {k: PLAN[k] for k in ('n_estimators', 'learning_rate', 'min_child_weight',
            'reg_lambda', 'subsample', 'colsample_bytree', 'tree_method', 'device', 'n_jobs', 'random_state')}
        self.native = xgb.XGBClassifier(max_depth=depth,
            objective='binary:logistic' if family == 'binary' else 'multi:softprob', **params)

    def fit(self, x, labels):
        target = np.asarray([int(v == 'SELF') for v in labels]) if self.family == 'binary' else np.asarray([driver.CANON.index(v) for v in labels])
        self.native.fit(x, target)
        assert np.array_equal(self.native.classes_, np.arange(len(self.classes_)))
        return self

    def predict_proba(self, x):
        return self.native.predict_proba(x)


def rank(row):
    m = row['metrics']
    return (-min(m['precision'] or 0, m['recall'] or 0), -(m['f1'] or 0),
            row['depth'], abs(row['threshold'] - 0.5), row['threshold'])


def main():
    baseline_plan = base.HERE / 'FIT_PLAN_DEVELOPMENT_V1.json'
    assert base.sha(baseline_plan) == '887abb6ba50f7e5fd310e1b4bad9293d0607b6eb871e0a2e1c833d0083dacd14'
    splits, _ = base.prepare(base.read_json(baseline_plan))
    tr = splits['train']
    x, labels, groups, folds = driver._validate(tr['x'], tr['labels'], tr['groups'], tr['folds'], None)
    labels = np.asarray(labels)
    output = base.HERE / 'runs' / PLAN['run_id']
    output.mkdir()
    base.save(output / 'fit_plan.json', dict(PLAN, source_sha256=base.sha(__file__),
        baseline_plan_sha256=base.sha(baseline_plan), xgboost_version=xgb.__version__))
    started = time.monotonic()
    cv_fits, refits = 0, 0
    models, cv_records = {}, []
    try:
        for family in ('binary', 'multiclass'):
            candidates = []
            for depth in PLAN['depths']:
                oof = np.full((len(x), 2 if family == 'binary' else 4), np.nan)
                for held in driver.FOLDS:
                    train, test = folds != held, folds == held
                    assert not set(np.asarray(groups)[train]) & set(np.asarray(groups)[test])
                    center = harness.fit_center(x[train])
                    assert cv_fits < PLAN['cv_fit_limit']
                    cv_fits += 1
                    est = Estimator(family, depth).fit(harness.transform_features(x[train], center), labels[train])
                    oof[test] = est.predict_proba(harness.transform_features(x[test], center))
                harness.validate_probabilities(oof, n_columns=oof.shape[1])
                p = oof[:, 1 if family == 'binary' else 0]
                cv_records.append(dict(family=family, depth=depth, probabilities=oof,
                    case_ids=tr['case_ids'], labels=labels, folds=folds))
                for threshold in PLAN['thresholds']:
                    candidates.append(dict(depth=depth, threshold=threshold,
                        metrics=harness.binary_metrics(labels == 'SELF', p >= threshold)))
            best = sorted(candidates, key=rank)[0]
            center = harness.fit_center(x)
            assert refits < PLAN['final_refit_limit']
            refits += 1
            est = Estimator(family, best['depth']).fit(harness.transform_features(x, center), labels)
            model_path = output / (family + '_model.ubj')
            est.native.save_model(model_path)
            reload = xgb.XGBClassifier()
            reload.load_model(model_path)
            for split in splits.values():
                tx = harness.transform_features(split['x'], center)
                np.testing.assert_array_equal(est.predict_proba(tx), reload.predict_proba(tx))
            base.save(output / (family + '_preprocessing.json'), dict(center=center,
                classes=est.classes_, threshold=best['threshold'], depth=best['depth']))
            models[family] = dict(selected=best, selected_using='TRAIN_GROUPED_OOF_ONLY',
                training=base.evaluate(family, est, center, tr, best['threshold']),
                validation=base.evaluate(family, est, center, splits['validation'], best['threshold']),
                ranking=sorted(candidates, key=rank), saved_model_reload_exact=True)
            print(json.dumps(dict(family=family, selected=best, validation=models[family]['validation']['self_gate'])), flush=True)
        assert cv_fits == 30 and refits == 2
        base.save(output / 'cross_validation.json', cv_records)
        base.save(output / 'development_results.json', dict(status='COMPLETE', models=models,
            cv_fits=cv_fits, final_refits=refits, holdout_accessed=False,
            additional_qwen_forwards=0, validation_used_for_selection=False,
            validation_is_repeated_development_set=True, logical_cases=dict(train=120, validation=40),
            elapsed_seconds=time.monotonic()-started))
        print(json.dumps(dict(cv_fits=cv_fits, refits=refits, elapsed_seconds=time.monotonic()-started)), flush=True)
    except Exception as exc:
        base.save(output / 'failure.json', dict(error=repr(exc), cv_fit_attempts=cv_fits,
            refit_attempts=refits, elapsed_seconds=time.monotonic()-started))
        raise


if __name__ == '__main__':
    main()

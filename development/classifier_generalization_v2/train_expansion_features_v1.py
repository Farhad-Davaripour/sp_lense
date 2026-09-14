"""Doubled-data comparison and five cached-feature ablations; no Qwen calls."""
import copy
import json
import pickle
import time
import warnings
import numpy as np
from scipy.special import expit, softmax
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import train_development_v1 as base
import train_xgboost_v1 as boosting
import grouped_driver as driver
import harness

FEATURES=['centered_l2','raw','unit_l2','pca32','three_cosine_directions','centered_l2_plus_norm']
PLAN=dict(run_id='expanded_features_20260914_v1',features=FEATURES,cv_fit_limit=420,refit_limit=24,
    train=240,validation_original=40,validation_added=40,holdout_access=False,
    logistic_C=[.01,.1,1,10],xgboost_depth=[1,2,3],thresholds=[i/20 for i in range(1,20)],
    preprocessing_fit='TRAIN_FOLD_ONLY',selection='TRAIN_GROUPED_OOF_ONLY',seconds=1800)

def unit(x):
    x=np.asarray(x,float)
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)

class Transform:
    def __init__(self,kind):self.kind=kind
    def fit(self,x,y):
        x=np.asarray(x,float);self.center=harness.fit_center(x)
        if self.kind=='pca32':self.pca=PCA(n_components=32,whiten=True,svd_solver='full').fit(x)
        if self.kind=='three_cosine_directions':
            means={k:x[np.asarray(y)==k].mean(axis=0) for k in driver.CANON}
            self.directions=unit(np.asarray([means['SELF']-means[k] for k in driver.CANON[1:]]))
        if self.kind=='centered_l2_plus_norm':
            z=np.log(np.maximum(np.linalg.norm(x,axis=1),1e-12))
            self.norm_mean=z.mean();self.norm_std=max(z.std(),1e-12)
        return self
    def transform(self,x):
        x=np.asarray(x,float)
        if self.kind=='raw':v=x
        elif self.kind=='unit_l2':v=unit(x)
        elif self.kind=='centered_l2':v=harness.transform_features(x,self.center)
        elif self.kind=='pca32':v=self.pca.transform(x)
        elif self.kind=='three_cosine_directions':v=unit(x)@self.directions.T
        elif self.kind=='centered_l2_plus_norm':
            z=(np.log(np.maximum(np.linalg.norm(x,axis=1),1e-12))-self.norm_mean)/self.norm_std
            v=np.column_stack([harness.transform_features(x,self.center),z])
        else:raise ValueError(self.kind)
        assert np.isfinite(v).all()
        return v

class SavedLogistic:
    def __init__(self,row):self.row=row;self.classes_=np.asarray(row['classes'])
    def predict_proba(self,x):
        scores=np.asarray(x)@np.asarray(self.row['coefficients']).T+np.asarray(self.row['intercept'])
        if scores.shape[1]==1:
            p=expit(scores[:,0]);return np.column_stack([1-p,p])
        return softmax(scores,axis=1)

def fit(kind,family,setting,x,y):
    if kind=='xgboost':return boosting.Estimator(family,setting).fit(x,y)
    est=driver._default_factory(family,setting)
    target=(np.asarray(y)=='SELF').astype(int) if family=='binary' else y
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning);est.fit(x,target)
    assert driver._converged(est)
    return est

def select_key(row):
    m=row['metrics']
    return (-min(m['precision'] or 0,m['recall'] or 0),-(m['f1'] or 0),row['setting'],
            abs(row['threshold']-.5),row['threshold'])

def evaluate(family,est,transform,split,tau):
    x=transform.transform(split['x']);y=np.asarray(split['labels'])
    p,probs=driver._fold_probabilities(family,est,x,len(x))
    pred=p>=tau
    out=dict(self_gate=harness.binary_metrics(y=='SELF',pred),
        predictions=[dict(case_id=i,truth=t,p_self=float(v),predicted_self=bool(w))
            for i,t,v,w in zip(split['case_ids'],y,p,pred)])
    if family=='multiclass':
        out['four_class']=harness.multiclass_metrics(y,harness.argmax_labels(probs,driver.CANON),driver.CANON)
        out['class_probabilities']=probs;out['class_order']=driver.CANON
    return out

def joined(a,b):return {k:list(a[k])+list(b[k]) for k in a}

def load_data():
    plan=base.read_json(base.HERE/'FIT_PLAN_DEVELOPMENT_V1.json')
    old,_=base.prepare(plan)
    newplan=copy.deepcopy(plan)
    newplan.update(capture_run_id='expansion_capture_20260914_v1',
        capture_lock_file='RUN_LOCK_EXPANSION_CAPTURE_V1.json',
        capture_lock_sha256='fc6ac2fc7b1439f23d4ce04ab34682a163e7f17388063b8cca451445f01a05be')
    new,_=base.prepare(newplan)
    splits=dict(training=joined(old['train'],new['train']),validation_original=old['validation'],
        validation_added=new['validation'],validation_combined=joined(old['validation'],new['validation']))
    assert len(splits['training']['case_ids'])==240 and len(splits['validation_combined']['case_ids'])==80
    return splits

def main():
    splits=load_data();train=splits['training']
    x,y,groups,folds=driver._validate(train['x'],train['labels'],train['groups'],train['folds'],None)
    y=np.asarray(y);groups=np.asarray(groups)
    out=base.HERE/'runs'/PLAN['run_id'];out.mkdir()
    base.save(out/'fit_plan.json',dict(PLAN,source_sha256=base.sha(__file__),
        plan_document_sha256=base.sha(base.HERE/'FEATURE_EXPERIMENT_PLAN_V1.md')))
    start=time.monotonic();cv_fits=0;refits=0;models={};cv=[]
    try:
        with threadpool_limits(limits=1):
            for feature in FEATURES:
                fold_inputs=[]
                for held in driver.FOLDS:
                    tr,te=folds!=held,folds==held
                    assert not set(groups[tr])&set(groups[te])
                    transform=Transform(feature).fit(x[tr],y[tr])
                    fold_inputs.append((tr,te,transform.transform(x[tr]),transform.transform(x[te])))
                full_transform=Transform(feature).fit(x,y);full_x=full_transform.transform(x)
                for kind in ('logistic','xgboost'):
                    settings=PLAN['logistic_C' if kind=='logistic' else 'xgboost_depth']
                    for family in ('binary','multiclass'):
                        candidates=[]
                        for setting in settings:
                            probs=np.full((len(x),2 if family=='binary' else 4),np.nan)
                            for tr,te,xt,xv in fold_inputs:
                                assert cv_fits<PLAN['cv_fit_limit'] and time.monotonic()-start<PLAN['seconds']
                                cv_fits+=1;est=fit(kind,family,setting,xt,y[tr])
                                ps,pc=driver._fold_probabilities(family,est,xv,int(te.sum()))
                                probs[te]=np.column_stack([1-ps,ps]) if family=='binary' else pc
                            harness.validate_probabilities(probs,n_columns=probs.shape[1])
                            ps=probs[:,1 if family=='binary' else 0]
                            cv.append(dict(feature=feature,kind=kind,family=family,setting=setting,
                                probabilities=probs,case_ids=train['case_ids'],labels=y,folds=folds))
                            for tau in PLAN['thresholds']:
                                candidates.append(dict(setting=setting,threshold=tau,metrics=harness.binary_metrics(y=='SELF',ps>=tau)))
                        best=sorted(candidates,key=select_key)[0]
                        assert refits<PLAN['refit_limit'];refits+=1
                        est=fit(kind,family,best['setting'],full_x,y)
                        key=feature+'__'+kind+'__'+family
                        row=dict(feature=feature,kind=kind,family=family,selected=best,
                            selected_using='TRAIN_GROUPED_OOF_ONLY',
                            evaluation={role:evaluate(family,est,full_transform,s,best['threshold']) for role,s in splits.items()})
                        # Own generated pickle only; never load untrusted external pickles.
                        bundle=pickle.dumps(dict(estimator=est,transform=full_transform,threshold=best['threshold']),protocol=5)
                        with (out/(key+'.pkl')).open('xb') as f:f.write(bundle)
                        restored=pickle.loads(bundle)
                        for split in splits.values():
                            a=est.predict_proba(full_transform.transform(split['x']))
                            b=restored['estimator'].predict_proba(restored['transform'].transform(split['x']))
                            np.testing.assert_array_equal(a,b)
                        row['reload_exact']=True;models[key]=row
                        base.save(out/(key+'_results.json'),row)
                        print(json.dumps(dict(candidate=key,selected=best,
                            original=row['evaluation']['validation_original']['self_gate'],
                            combined=row['evaluation']['validation_combined']['self_gate'])),flush=True)
            assert cv_fits==420 and refits==24
            baselines={}
            for kind,run in [('logistic','classifier_development_20260914_v1'),('xgboost','xgboost_development_20260914_v1')]:
                for family in ('binary','multiclass'):
                    folder=base.HERE/'runs'/run
                    if kind=='logistic':
                        saved=base.read_json(folder/(family+'_model.json'));est=SavedLogistic(saved)
                    else:
                        saved=base.read_json(folder/(family+'_preprocessing.json'))
                        est=boosting.Estimator(family,saved['depth']);est.native.load_model(folder/(family+'_model.ubj'))
                    transform=Transform('centered_l2');transform.center=np.asarray(saved['center'])
                    baselines[kind+'__'+family]={role:evaluate(family,est,transform,s,saved['threshold']) for role,s in splits.items() if role!='training'}
            # Global selection ignores validation; prefer earlier feature on exact OOF metric ties.
            selected=min(models,key=lambda k:(select_key(models[k]['selected'])[:2],FEATURES.index(models[k]['feature']),
                0 if models[k]['kind']=='logistic' else 1,0 if models[k]['family']=='binary' else 1,
                select_key(models[k]['selected'])[2:]))
            base.save(out/'cross_validation.json',cv)
            result=dict(status='COMPLETE',cv_fits=cv_fits,refits=refits,holdout_accessed=False,
                elapsed_seconds=time.monotonic()-start,selected_on_train_oof=selected,
                models=models,original_models_on_expanded_validation=baselines,
                interpretation='Synthetic scenario classification; repeated development validation; not motivation evidence.',
                data_counts=dict(train=240,validation_original=40,validation_added=40,validation_combined=80),
                training_groups=7,grouped_folds=5,independent_label_audit=False)
            base.save(out/'development_results.json',result)
            print(json.dumps(dict(status='COMPLETE',cv_fits=cv_fits,refits=refits,selected_on_train_oof=selected,
                elapsed_seconds=result['elapsed_seconds'])),flush=True)
    except Exception as exc:
        base.save(out/'failure.json',dict(error=repr(exc),cv_fit_attempts=cv_fits,refit_attempts=refits))
        raise

if __name__=='__main__':
    import sys
    sys.modules['train_expansion_features_v1']=sys.modules[__name__]
    Transform.__module__='train_expansion_features_v1'
    SavedLogistic.__module__='train_expansion_features_v1'
    main()

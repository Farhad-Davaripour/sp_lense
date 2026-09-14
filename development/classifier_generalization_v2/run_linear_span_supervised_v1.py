"""Root-owned pinning/preflight and existing hard-watch wrapper, no new loader."""
import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path
import native_development_runner_v2 as native

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
COMMIT='7813175e2eb7eda3d266a667ce8a034c9ef974f6'
RUN='linear_span_20260914_v1'
SOURCE_PLAN_SHA='a889c5c8c9cee62d4752ae0a7f2a4891aadede84a6b91ce5abb13ee1c91e2239'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def save(p,v):
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')

def prepare():
    source=HERE/'SPAN_FIT_PLAN_V2.json'
    assert sha(source)==SOURCE_PLAN_SHA
    names=['linear_span_control_v1.py','span_classifier_driver_v1.py','span_feature_transforms_v1.py',
           'grouped_driver.py','harness.py','native_development_runner_v2.py']
    code={(HERE/n).relative_to(ROOT).as_posix():sha(HERE/n) for n in names}
    native.check_sources(ROOT,COMMIT,code)
    plan=dict(schema='linear_span_fit_plan.v1',run_id=RUN,algorithm='logistic_regression',
        scientific_execution_authorized=False,source_commit=COMMIT,code_source_files=code,
        source_plan=dict(path=source.relative_to(ROOT).as_posix(),sha256=SOURCE_PLAN_SHA),
        runtime_packages={k:importlib.metadata.version(k) for k in ('numpy','scipy','scikit-learn')},
        C=[.1,1,10],thresholds=[i/20 for i in range(1,20)],folds=[0,1,2,3,4],
        cv_fit_limit=30,refit_limit=2,seconds=300,output_bytes=268435456,threads=1,holdout_access=False,
        feature='Last token, blocks6/10/18, normalizeeach1024vectorseparately thenconcatenate3072; no learned preprocessing',
        selection='TRAINOOFminPR,F1,lowerC,binaryfamilytie,tauclosest.5,smallertau; validation reportingonly')
    path=HERE/'LINEAR_SPAN_FIT_PLAN_V1.json';save(path,plan)
    print(json.dumps(dict(plan=str(path),sha256=sha(path))))

def checked(path,expected):
    assert sha(path)==expected
    p=read(path);assert p['schema']=='linear_span_fit_plan.v1' and p['run_id']==RUN
    assert p['cv_fit_limit']==30 and p['refit_limit']==2 and p['seconds']==300 and p['holdout_access'] is False
    native.check_sources(ROOT,p['source_commit'],p['code_source_files'])
    for k,v in p['runtime_packages'].items():assert importlib.metadata.version(k)==v
    src=ROOT/p['source_plan']['path'];assert sha(src)==p['source_plan']['sha256']==SOURCE_PLAN_SHA
    import span_classifier_driver_v1 as loader
    prior=loader.load_plan(src,SOURCE_PLAN_SHA)
    source=loader.load_source(prior,prior['source']['lock']['sha256'],ROOT)
    data=loader.load_case_data(source,prior,ROOT)
    import linear_span_control_v1 as core
    assert list(core.C_VALUES)==p['C'] and list(core.TAUS)==p['thresholds']
    order,matrix=core.build_features(data)
    assert matrix.shape==(320,3072) and len(data['train_ids'])==240 and len(data['validation_ids'])==80
    return p,data,core

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','preflight','run','worker'])
    ap.add_argument('--plan');ap.add_argument('--sha256');args=ap.parse_args()
    if args.mode=='prepare':return prepare()
    path=Path(args.plan);plan,data,core=checked(path,args.sha256)
    if args.mode=='preflight':
        print(json.dumps(dict(status='preflight_pass',train=240,validation=80,feature_width=3072,fits=0,holdout_accessed=False)));return
    out=HERE/'runs'/RUN
    if args.mode=='worker':
        assert plan['scientific_execution_authorized'] is True and sha(Path(__file__))==plan['supervisor_source_sha256']
        from threadpoolctl import threadpool_limits
        start=time.monotonic()
        with threadpool_limits(limits=1):result=core.run(data,out,deadline=lambda:time.monotonic()-start>300)
        save(out/'fit_plan.json',plan)
        assert sum(f.stat().st_size for f in out.rglob('*') if f.is_file())<=plan['output_bytes']
        print(json.dumps({k:result[k] for k in ('status','counters','train_selected')}));return
    assert not out.exists() and not (HERE/'runs/native_model_owner.json').exists()
    review=HERE/'LINEAR_SPAN_CODE_REVIEW_V1.md';review_text=review.read_text().lower()
    assert 'pass_scoped' in review_text and plan['code_source_files']['development/classifier_generalization_v2/linear_span_control_v1.py'] in review_text
    plan['scientific_execution_authorized']=True;plan['supervisor_source_sha256']=sha(Path(__file__))
    plan['review_sha256']=sha(review)
    release=HERE/'LINEAR_SPAN_FIT_PLAN_V2.json';save(release,plan)
    command=[sys.executable,str(Path(__file__).resolve()),'worker','--plan',str(release),'--sha256',sha(release)]
    started=time.monotonic()
    try:
        pid,raw=native.watch(command,str(ROOT),300)
        result_path=out/'development_results.json';result=read(result_path)
        assert result['status']=='COMPLETE' and result['counters']['cv_fits']==30 and result['counters']['refits']<=2
        selected=result['train_selected'];assert selected['family'] in result['family_results']
        receipt=dict(status='complete',pid=pid,elapsed_seconds=time.monotonic()-started,plan_sha256=sha(release),
            results_sha256=sha(result_path),counters=result['counters'],train_selected=selected)
        save(out/'supervisor_success.json',receipt);print(json.dumps(receipt))
    except BaseException as exc:
        destination=(out/'supervisor_failure.json') if out.exists() else (HERE/'runs'/(RUN+'_supervisor_failure.json'))
        save(destination,dict(status='failed',error=str(exc),elapsed_seconds=time.monotonic()-started,
            child_pid=getattr(exc,'owned_child_pid',None),child_closed=getattr(exc,'owned_child_closed',None)))
        raise

if __name__=='__main__':main()

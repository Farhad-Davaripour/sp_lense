"""Root-owned cohort release; reuse the authenticated loader and hard watcher."""
import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path
import native_development_runner_v2 as native

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
COMMIT='027d459c3af77c5b21fcc0a1e9e4338ceda1e83e'
RUN='cohort_transfer_20260914_v1'
SOURCE_SHA='0b34b669c22e156a8fc37a792b7be232c89c915aa0121f6d1f2585ff77128caa'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def save(p,v):
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')

def prepare():
    source=HERE/'LINEAR_SPAN_FIT_PLAN_V2.json';assert sha(source)==SOURCE_SHA
    names=['cohort_transfer_control_v1.py','linear_span_control_v1.py','run_linear_span_supervised_v1.py',
           'span_classifier_driver_v1.py','span_feature_transforms_v1.py','grouped_driver.py','harness.py','native_development_runner_v2.py']
    code={(HERE/n).relative_to(ROOT).as_posix():sha(HERE/n) for n in names}
    native.check_sources(ROOT,COMMIT,code)
    plan=dict(schema='cohort_transfer_fit_plan.v1',run_id=RUN,scientific_execution_authorized=False,
        source_commit=COMMIT,code_source_files=code,source_plan=dict(path=source.relative_to(ROOT).as_posix(),sha256=SOURCE_SHA),
        runtime_packages={k:importlib.metadata.version(k) for k in ('numpy','scipy','scikit-learn')},
        training_counts=dict(original=120,added=120),validation_counts=dict(original=40,added=40),
        cv_fits_per_arm=30,refits_per_arm=2,total_fit_limit=64,seconds=300,output_bytes=268435456,
        threads=1,holdout_access=False,comparison='Family-matched endpoints; C/tau selected within each TRAIN arm only. No author-causality claim.')
    path=HERE/'COHORT_TRANSFER_FIT_PLAN_V1.json';save(path,plan)
    print(json.dumps(dict(plan=str(path),sha256=sha(path))))

def checked(path,expected):
    assert sha(path)==expected
    plan=read(path);assert plan['schema']=='cohort_transfer_fit_plan.v1' and plan['run_id']==RUN
    assert plan['seconds']==300 and plan['total_fit_limit']==64 and plan['holdout_access'] is False
    native.check_sources(ROOT,plan['source_commit'],plan['code_source_files'])
    for k,v in plan['runtime_packages'].items():assert importlib.metadata.version(k)==v
    source=ROOT/plan['source_plan']['path'];assert sha(source)==plan['source_plan']['sha256']==SOURCE_SHA
    import run_linear_span_supervised_v1 as prior
    _,data,_=prior.checked(source,SOURCE_SHA)  # Loader only; never prior.run.
    import cohort_transfer_control_v1 as cohort
    arms=cohort.derive_arms(data)
    assert {k:len(v) for k,v in arms.items()}==plan['training_counts']
    return plan,data,cohort

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','preflight','run','worker']);ap.add_argument('--plan');ap.add_argument('--sha256');a=ap.parse_args()
    if a.mode=='prepare':return prepare()
    path=Path(a.plan);plan,data,cohort=checked(path,a.sha256);out=HERE/'runs'/RUN
    if a.mode=='preflight':
        print(json.dumps(dict(status='preflight_pass',training_arms=[120,120],class_count_per_arm=30,groups=7,folds=5,validation=80,fits=0,holdout_accessed=False)));return
    if a.mode=='worker':
        assert plan['scientific_execution_authorized'] is True and sha(Path(__file__))==plan['supervisor_source_sha256']
        from threadpoolctl import threadpool_limits
        started=time.monotonic()
        with threadpool_limits(limits=1):result=cohort.run(data,out,deadline=lambda:time.monotonic()-started>300,plan_sha256=a.sha256)
        save(out/'fit_plan.json',plan)
        assert sum(f.stat().st_size for f in out.rglob('*') if f.is_file())<=plan['output_bytes']
        print(json.dumps(dict(status='complete',run_id=RUN,aggregate=result['aggregate'])));return
    assert not out.exists() and not (HERE/'runs/native_model_owner.json').exists()
    review=HERE/'COHORT_TRANSFER_CODE_REVIEW_V1.md';review_text=review.read_text().lower()
    assert 'pass_scoped' in review_text and plan['code_source_files']['development/classifier_generalization_v2/cohort_transfer_control_v1.py'] in review_text
    plan['scientific_execution_authorized']=True;plan['review_sha256']=sha(review);plan['supervisor_source_sha256']=sha(Path(__file__))
    release=HERE/'COHORT_TRANSFER_FIT_PLAN_V2.json';save(release,plan)
    started=time.monotonic()
    try:
        pid,_=native.watch([sys.executable,str(Path(__file__).resolve()),'worker','--plan',str(release),'--sha256',sha(release)],str(ROOT),300)
        record_path=out/'cohort_transfer_record.json';r=read(record_path)
        assert r['run_id']==RUN and r['parent_plan_sha256']==sha(release) and r['holdout_accessed'] is False
        counts={k:sum(v['counters'][k] for v in r['arms'].values()) for k in ['cv_fits','refits','fit_errors','refit_errors']}
        assert counts['cv_fits']==60 and counts['refits']<=4 and r['aggregate']['core_calls']==2
        receipt=dict(status='complete',run_id=RUN,pid=pid,elapsed_seconds=time.monotonic()-started,
            counters=counts,record_sha256=sha(record_path),plan_sha256=sha(release))
        save(out/'supervisor_success.json',receipt);print(json.dumps(receipt))
    except BaseException as exc:
        dest=(out/'supervisor_failure.json') if out.exists() else HERE/'runs'/(RUN+'_supervisor_failure.json')
        save(dest,dict(status='failed',error=str(exc),elapsed_seconds=time.monotonic()-started,
            child_pid=getattr(exc,'owned_child_pid',None),child_closed=getattr(exc,'owned_child_closed',None)))
        raise

if __name__=='__main__':main()

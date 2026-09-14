"""Root-owned source/version checks plus existing hard-deadline child watcher."""
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path
import native_development_runner_v2 as native

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def save(p,value):
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,indent=2);f.write('\n')

def main():
    # This wrapper is invoked only after root reads and accepts the review.
    review=HERE/'SPAN_CLASSIFIER_DRIVER_REVIEW_V1.md'
    report=review.read_text().lower()
    assert 'pass_scoped' in report,'Independent driver review not admitted'
    original=HERE/'SPAN_FIT_PLAN_V1.json'
    assert sha(original)=='251f8ee78bd97bbda18aa7775dfe59f5c8fde17b4815837c84e085f8de051e3b'
    plan=read(original)
    native.check_sources(ROOT,plan['source_commit'],plan['code_source_files'])
    expected_driver=plan['code_source_files']['development/classifier_generalization_v2/span_classifier_driver_v1.py']
    assert expected_driver in report or (expected_driver[:8] in report and expected_driver[-10:] in report),'Review source pin'
    for package,version in plan['runtime_packages'].items():assert importlib.metadata.version(package)==version
    assert not (HERE/'runs/native_model_owner.json').exists(),'Native model still active'
    out=HERE/'runs'/plan['run_id']
    assert not out.exists(),'Do not overwrite prior fit attempt'
    plan['scientific_execution_authorized']=True
    plan['execution_gate']='Root accepted independent review; source/version pins checked; existing watcher enforces600second child deadline.'
    plan['independent_driver_review']=dict(path=str(review.relative_to(ROOT)).replace('\\','/'),sha256=sha(review))
    plan['fit_supervisor_source_sha256']=sha(Path(__file__))
    plan_path=HERE/'SPAN_FIT_PLAN_V2.json'
    save(plan_path,plan)
    # Import by module name so serialized estimators do not reference __main__.
    bootstrap=f'import sys; sys.path.insert(0, {str(HERE)!r}); import span_classifier_driver_v1 as d; sys.exit(d.main())'
    command=[sys.executable,'-c',bootstrap,'--run','--plan',str(plan_path),
        '--plan-sha256',sha(plan_path),'--source-sha256',plan['source']['lock']['sha256']]
    started=time.monotonic()
    try:
        pid,raw=native.watch(command,str(ROOT),plan['seconds'])
        result_path=out/'development_results.json'
        result=read(result_path)
        assert result['status']=='COMPLETE' and result['holdout_accessed'] is False
        counts=result['counters']
        assert counts['cv_fits']==50 and counts['refits']<=10 and counts['transform_fits']==6
        assert result['plan_sha256']==sha(plan_path)
        receipt=dict(status='complete',pid=pid,elapsed_seconds=time.monotonic()-started,
            plan_sha256=sha(plan_path),results_sha256=sha(result_path),counters=counts,
            train_selected=result['train_selected'],hard_deadline_seconds=plan['seconds'])
        save(out/'fit_supervisor_success.json',receipt)
        print(json.dumps(receipt))
    except BaseException as exc:
        # Preserve any child-written partial outputs; never retry this run id.
        failure=dict(status='failed',error=str(exc),error_type=type(exc).__name__,
            child_pid=getattr(exc,'owned_child_pid',None),child_closed=getattr(exc,'owned_child_closed',None),
            plan_sha256=sha(plan_path),elapsed_seconds=time.monotonic()-started)
        destination=(out/'fit_supervisor_failure.json') if out.exists() else (HERE/'runs'/(plan['run_id']+'_supervisor_failure.json'))
        save(destination,failure)
        raise

if __name__=='__main__':main()

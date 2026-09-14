"""Pin completed capture and source versions; no fits or native-model imports."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import native_development_runner_v2 as native
import span_classifier_driver_v1 as driver

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
COMMIT='f38d5e77f90149d0842b8f4c43996f31da0b830d'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_bytes())
def pin(path):return dict(path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,sha256=sha(path))

def main():
    run=HERE/'runs/span_capture_20260914_v1'
    receipt=read(run/'supervisor_success.json')
    assert receipt['status']=='complete' and receipt['lock_sha256']=='a1174a712f3208813cbdbf948f6c2af2cda3b1dd8f5febdd20e43062d2e3f16c'
    for name,value in receipt['outputs'].items():
        assert sha(run/name)==value['sha256'] and (run/name).stat().st_size==value['bytes']
    names=['span_classifier_driver_v1.py','span_feature_transforms_v1.py','grouped_driver.py','harness.py','native_development_runner_v2.py']
    code={str((HERE/n).relative_to(ROOT)).replace('\\','/'):sha(HERE/n) for n in names}
    native.check_sources(ROOT,COMMIT,code)
    manifests={'original_train':'TRAIN_ACCEPTED_V10.json','original_validation':'VALIDATION_ACCEPTED_V3.json',
               'added_train':'EXPANSION_TRAIN_ROOT_V2.json','added_validation':'EXPANSION_VALIDATION_ROOT_V2.json',
               'blueprint':'GROUP_BLUEPRINT_V4.json'}
    plan=dict(schema=driver.PLAN_SCHEMA,job_id=driver.JOB_ID,run_id='span_classifier_20260914_v1',seconds=600,
        cv_fit_limit=50,refit_limit=10,transform_fit_limit=6,thresholds=list(driver.TAUS),xgboost=driver.EXPECTED_XGB_PARAMS,
        representations=list(driver.REPRESENTATIONS),holdout_access=False,
        source=dict(receipt=pin(run/'supervisor_success.json'),lock=pin(HERE/'RUN_LOCK_SPAN_CAPTURE_V1.json'),
                    index=pin(run/'index.json'),windows=pin(run/'windows.f32')),
        manifests={k:pin(HERE/v) for k,v in manifests.items()},source_commit=COMMIT,code_source_files=code,
        runtime_packages={k:importlib.metadata.version(k) for k in ('numpy','scikit-learn','scipy','xgboost')},
        execution_gate='Zero-fit preflight permitted; actualfit requires rootacceptance of independentdriverreview and600secondexternalwatch.',
        scientific_execution_authorized=False)
    path=HERE/'SPAN_FIT_PLAN_V1.json'
    with path.open('x',encoding='utf-8',newline='\n') as f:json.dump(plan,f,indent=2);f.write('\n')
    print(json.dumps(dict(path=str(path),plan_sha256=sha(path),source_sha256=receipt['lock_sha256'])))

if __name__=='__main__':main()

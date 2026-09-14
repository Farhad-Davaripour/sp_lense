"""Prepare a new finite lock only after root accepts pinned independent reviews."""
import hashlib
import json
import re
from pathlib import Path
import native_development_runner_v2 as native
import span_development_runner_v1 as span

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
COMMIT='d6e72278616cb545264854d9ca9721e4b7f6215e'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_bytes())
def pin(path):return dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path))

def main():
    oldpath=HERE/'RUN_LOCK_DEVELOPMENT_CAPTURE_V1.json'
    newpath=HERE/'RUN_LOCK_EXPANSION_CAPTURE_V1.json'
    assert sha(oldpath)=='989c1fcb06cdabb4d3985ada6cee7692f903c4e4d14e0cc4d93a132284ed30ec'
    assert sha(newpath)=='fc6ac2fc7b1439f23d4ce04ab34682a163e7f17388063b8cca451445f01a05be'
    old,new=read(oldpath),read(newpath)
    assert old['runtime']==new['runtime'] and old['threads']==new['threads']
    sources={(span.STUDY/name).as_posix():sha(HERE/name) for name in span.SOURCE_NAMES}
    native.check_sources(ROOT,COMMIT,sources)
    reviews={
        'SPAN_CAPTURE_ADAPTER_REVIEW_V1.md': ['59eddfb20ade6fb97c10db709dad9012b6c4b77e72765b75cab186c443e20a99'],
        'SPAN_FEATURE_TRANSFORMS_REVIEW_V2.md': ['a23aaa38dc36a764af8cd30d2d8ed40193a9a3496a9d8687778c96f74741c5c0'],
        'SPAN_RUNNER_REVIEW_V1.md': ['54e29c37f9cbbffbab3f81c10d43f472d7d0ce9aab55ce4c7d0f105e36ab4abf'],
    }
    for name,hashes in reviews.items():
        text=(HERE/name).read_text().lower()
        accepted='both v1 defects closed; no regression' if name=='SPAN_FEATURE_TRANSFORMS_REVIEW_V2.md' else 'pass_scoped'
        assert accepted in text,(name,'REVIEW_NOT_PASS')
        # Root inspected the re-review's abbreviated pins and verified full
        # source/test hashes locally; other reviews retain full pins.
        assert all(h in text or (h[:8] in text and h[-10:] in text) for h in hashes),(name,'REVIEW_PIN_MISSING')
    assert sha(HERE/'span_feature_transforms_v1.py')=='a23aaa38dc36a764af8cd30d2d8ed40193a9a3496a9d8687778c96f74741c5c0'
    assert sha(HERE/'test_span_feature_transforms_v1.py')=='76117a26cd7b80406761ce13c23ad469c99012dec8a3a4e706ec1e9a5cc19c48'
    admission=read(HERE/'EXPANSION_LABEL_AUDIT_ADMISSION_V1.json')
    assert admission['status']=='ADMITTED_WITH_REFERENCE_PIN_CLARIFICATION'
    assert admission['case_verdicts']==admission['pass_verdicts']==160
    assert sha(HERE/admission['audit'])==admission['audit_sha256']
    for name,expected in admission['primary_data_hashes_verified'].items():assert sha(HERE/name)==expected
    assert not (HERE/'runs/native_model_owner.json').exists()
    run_id='span_capture_20260914_v1'
    assert not (HERE/'runs'/run_id).exists()
    lock=dict(schema=span.SCHEMA,scientific_execution_authorized=True,run_id=run_id,
        source_commit=COMMIT,source_files=sources,runtime=old['runtime'],threads=old['threads'],
        caps=span.CAPS,snapshot_cache_root=old['snapshot_cache_root'],
        inputs={k:old['inputs'][k] for k in span.INPUT_ROLES},
        reference_locks=[dict(pin(p),run_id=d['run_id'],forwards=320) for p,d in [(oldpath,old),(newpath,new)]],
        independent_review_pins={name:pin(HERE/name) for name in reviews},
        supplemental_audit_admission=pin(HERE/'EXPANSION_LABEL_AUDIT_ADMISSION_V1.json'),
        scope='One640forward development-only prompt-span capture. No fitting, responsegeneration, logits evaluation, weight changes or holdout capture. Old locks are preflight references only.',
        feature_contract=dict(blocks=[6,10,18],width=1024,max_tokens=16,position='last_shared_preoption_suffix',
                              dtype='float32',byte_order='little'),
        new_fit_execution_authorized=False)
    path=HERE/'RUN_LOCK_SPAN_CAPTURE_V1.json'
    with path.open('x',encoding='utf-8',newline='\n') as f:json.dump(lock,f,indent=2);f.write('\n')
    print(json.dumps(dict(path=str(path),sha256=sha(path))))

if __name__=='__main__':main()

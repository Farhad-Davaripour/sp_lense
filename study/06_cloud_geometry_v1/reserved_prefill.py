"""One frozen geometric check of reserved position0 targets; no candidate refit."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from analyze import fit,predict,score

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(oldroot,geometry,fresh,output):
    output.mkdir(exist_ok=False);source=Path(__file__).parent
    plan=json.loads((source/'RESERVED_PREFILL_PLAN.json').read_text())
    assert sha(geometry/'MAIN_PLUS_MAPS.npz')==plan['candidate_model_sha256']
    assert sha(geometry/'TRAIN_POSITION_PROFILES.npz')==plan['TRAIN_position_profile_sha256']
    for n in ['RESERVED_PREFILL_PLAN.json','reserved_prefill.py']:shutil.copyfile(source/n,output/n)
    with np.load(geometry/'MAIN_PLUS_MAPS.npz') as z:main={k:(z[k].item() if z[k].ndim==0 else z[k].copy()) for k in z.files}
    with np.load(geometry/'TRAIN_POSITION_PROFILES.npz') as z:prefill_shift=z['original_existing_vector'][0:1].copy()
    manifest=json.loads((oldroot/'EXPORT_HASHES.json').read_text());training=[]
    for pid in ['r4_01','r4_02','r4_03','r4_04']:
        arrays=[]
        for arm in ['base','existing_vector']:
            p=oldroot/'native'/pid/(arm+'.npz');assert sha(p)==manifest[p.relative_to(oldroot).as_posix()]
            with np.load(p) as z:arrays.append(z['residuals'][15].astype(np.float64))
        training.append(tuple(arrays))
    controls=[];control_hashes=[]
    for offset in [1,2,3]:
        sham=[(x,x+training[(i+offset)%4][1]-training[(i+offset)%4][0]) for i,(x,y) in enumerate(training)]
        model=fit(sham);controls.append(model)
        h=hashlib.sha256()
        for key in ['mx','my','joint','q','proper','vx','vd','beta']:
            h.update(key.encode());h.update(model[key].tobytes())
        control_hashes.append(h.hexdigest())
    (output/'MODELS_FROZEN_BEFORE_RESERVED_GEOMETRY.json').write_text(json.dumps({'candidate_sha256':plan['candidate_model_sha256'],'cyclic_model_parameter_hashes':control_hashes,'fresh_geometry_accessed_before_this_freeze':False,'prior_classification_access_disclosed':True},indent=2))
    # The reserved data are first opened for geometric scoring below this freeze.
    manifest=json.loads((fresh/'EXPORT_HASHES.json').read_text());spec=json.loads((fresh/'source/NEXT_VALIDATION_PLAN.json').read_text());rows=[]
    for prompt in spec['prompts']:
        vectors=[]
        for condition in ['base','plus_v']:
            p=fresh/'native'/prompt['id']/(condition+'.npz');assert sha(p)==manifest[p.relative_to(fresh).as_posix()]
            with np.load(p) as z:vectors.append(z['residuals'][15,0:1].astype(np.float64))
        x,y=vectors;delta=y-x
        scores={k:score(delta,predict(x,main,k)) for k in ['identity','translation','rigid_isometry','proper_rigid_in_chosen_span','scaled_isometry','ridge_affine_deformation']}
        scores['TRAIN_prefill_shift']=score(delta,prefill_shift)
        for i,m in enumerate(controls):scores['cyclic_affine_'+str(i+1)]=score(delta,predict(x,m,'ridge_affine_deformation'))
        affine=scores['ridge_affine_deformation'];comparators=[v for k,v in scores.items() if k not in ['ridge_affine_deformation','proper_rigid_in_chosen_span']]
        passed=affine>=.70 and all(affine>v for v in comparators)
        rows.append({'prompt':prompt['id'],**scores,'predeclared_prompt_support':passed})
    passed=all(r['predeclared_prompt_support'] for r in rows)
    result={'evidence':'Reserved-data geometric check; prior classifier/integrity access disclosed; not pristine prospective geometry confirmation',
        'unique_questions':8,'position_zero_only':True,'decode_geometry_not_scored':True,'candidate_unchanged':sha(geometry/'MAIN_PLUS_MAPS.npz')==plan['candidate_model_sha256'],
        'old_all_position_main_criterion_still_failed':True,'predeclared_all8_support':passed,'support_count':sum(r['predeclared_prompt_support'] for r in rows),'rows':rows,'GPU_or_new_native_inference':False}
    (output/'RESULTS.json').write_text(json.dumps(result,indent=2))
    with (output/'PREDICTIONS.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('oldroot',type=Path);p.add_argument('geometry',type=Path);p.add_argument('fresh',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();run(a.oldroot,a.geometry,a.fresh,a.output)

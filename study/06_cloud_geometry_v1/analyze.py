"""Static full-coordinate cloud maps; all geometry fit references are TRAIN-only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

KINDS=['identity','translation','rigid_isometry','proper_rigid_in_chosen_span','scaled_isometry','ridge_affine_deformation']
ANCHOR=15

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def basis(a):
    _,s,vt=np.linalg.svd(a,full_matrices=False)
    tolerance=max(a.shape)*np.finfo(np.float64).eps*s[0]
    rank=int(np.sum(s>tolerance))
    return vt[:rank].T,s[:rank],dict(numerical_rank=rank,ambient_dimension=a.shape[1],
        ambient_unobserved_dimensions=a.shape[1]-rank,numerical_tolerance=float(tolerance),
        stable_rank=float(np.square(s).sum()/s[0]**2),BF16_relative_scale_rank=int(np.sum(s>2**-7*s[0])))

def d2(x):
    norm=np.square(x).sum(axis=1)
    d=np.maximum(norm[:,None]+norm[None,:]-2*x@x.T,0)
    np.fill_diagonal(d,0)
    return d

def fit(traces):
    x=np.concatenate([a for a,b in traces]);y=np.concatenate([b for a,b in traces])
    weights=np.concatenate([np.full(len(a),1/(len(traces)*len(a))) for a,b in traces])
    mx=np.average(x,axis=0,weights=weights);my=np.average(y,axis=0,weights=weights)
    xc=x-mx;yc=y-my;aw=xc*np.sqrt(weights)[:,None];bw=yc*np.sqrt(weights)[:,None]
    vx,sx,rankx=basis(aw);vd,sd,rankd=basis(bw-aw);joint,sj,rankj=basis(np.concatenate([aw,bw]))
    a=aw@joint;b=bw@joint
    lamq=.01*np.square(aw).sum()/joint.shape[1]
    u,s,vt=np.linalg.svd(a.T@b+lamq*np.eye(joint.shape[1]),full_matrices=False)
    q=u@vt;det=float(np.linalg.det(q))
    correction=np.ones(len(u));correction[-1]=1 if det>=0 else -1
    proper=(u*correction)@vt
    scale=max(0,float(np.sum((a@q)*b)/np.square(aw).sum()))
    lamw=.01*np.square(aw).sum()/vx.shape[1]
    projected=aw@vx
    beta=(projected.T@(bw-aw))/(sx**2+lamw)[:,None]
    dx=d2(x);dy=d2(y);distance_scale2=float(np.sum(dx*dy)/np.square(dx).sum())
    return dict(mx=mx,my=my,joint=joint,q=q,proper=proper,vx=vx,vd=vd,beta=beta,scale=scale,
        distance_scale2=distance_scale2,rankx=rankx,rankd=rankd,rankj=rankj,lambda_orthogonal=lamq,lambda_affine=lamw,
        chosen_joint_determinant=det,TRAIN_unique_prompts=len(traces),TRAIN_positions=len(x))

def predict(x,model,kind):
    xc=x-model['mx'];mean=model['my']-model['mx']
    if kind=='identity':return np.zeros_like(x)
    if kind=='translation':return np.broadcast_to(mean,x.shape)
    if kind=='ridge_affine_deformation':return mean+(xc@model['vx'])@model['beta']
    q=model['proper'] if kind=='proper_rigid_in_chosen_span' else model['q']
    extra=((xc@model['joint'])@(q-np.eye(len(q))))@model['joint'].T
    if kind=='scaled_isometry':return mean+(model['scale']-1)*xc+model['scale']*extra
    return mean+extra

def score(delta,predicted):
    denominator=np.square(delta).sum()
    return float(1-np.square(delta-predicted).sum()/denominator) if denominator>1e-20 else None

def outside(x,v):
    energy=np.square(x).sum()
    return float(np.clip(1-np.square(x@v).sum()/energy,0,1)) if energy>1e-20 else None

def structure(x,y,m):
    dx=d2(x);dy=d2(y);xc=x-m['mx'];yc=y-m['my'];kx=xc@xc.T;ky=yc@yc.T
    alpha=m['distance_scale2']
    return dict(squared_distance_relative_error=float(np.linalg.norm(dy-dx)/np.linalg.norm(dx)),
        TRAIN_scale_corrected_squared_distance_error=float(np.linalg.norm(dy-alpha*dx)/np.linalg.norm(dx)),
        TRAIN_centered_Gram_relative_error=float(np.linalg.norm(ky-kx)/np.linalg.norm(kx)),
        TRAIN_scale_corrected_Gram_error=float(np.linalg.norm(ky-alpha*kx)/np.linalg.norm(kx)),
        TRAIN_squared_distance_scale=alpha,
        baseline_energy_outside_TRAIN_baseline_span=outside(xc,m['vx']),
        centered_displacement_energy_outside_TRAIN_displacement_span=outside((y-x)-(m['my']-m['mx']),m['vd']),
        centered_displacement_energy_outside_TRAIN_joint_span=outside((y-x)-(m['my']-m['mx']),m['joint']))

def bf16(x):
    f=np.asarray(x,dtype=np.float32);bits=f.view(np.uint32)
    rounded=(bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)
    return rounded.view(np.float32).astype(np.float64)

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def run(original,matched,output):
    output.mkdir(parents=True,exist_ok=False);source=Path(__file__).parent
    for n in ['PLAN.json','LITERATURE.json','analyze.py']:shutil.copyfile(source/n,output/n)
    manifests={str(r):json.loads((r/'EXPORT_HASHES.json').read_text()) for r in [original,matched]}
    used={};cache={};delivery_max=0.
    def read(root,cohort,pid,condition):
        nonlocal delivery_max
        key=(str(root),cohort,pid,condition)
        if key in cache:return cache[key]
        def one(arm):
            path=root/cohort/pid/(arm+'.npz');digest=sha(path)
            assert manifests[str(root)][path.relative_to(root).as_posix()]==digest;used[str(path)]=digest
            with np.load(path,allow_pickle=False) as z:
                h=z['residuals'].copy();tokens=z['generated_token_ids' if 'generated_token_ids' in z else 'forced_token_ids'].copy()
                pre=z['pre_injection'].copy() if 'pre_injection' in z else None
            return h,tokens,pre
        b,t,p=one('base');y,ty,py=one(condition)
        assert np.array_equal(t,ty) and np.array_equal(b[:10],y[:10])
        if p is not None:assert np.array_equal(p,py)
        assert np.isfinite(b).all() and np.isfinite(y).all()
        injection=np.linalg.norm(y[10].astype(np.float64)-b[10].astype(np.float64),axis=1)
        receipt=json.loads((root/cohort/'STATUS.json').read_text())
        record=next(r for r in receipt['completed'] if r['prompt']==pid and r['condition']==condition)
        delivered=np.asarray(record['delivery_norm_and_changed_count'])[:,0] if 'delivery_norm_and_changed_count' in record else np.asarray(record['delivery'])[:,1]
        error=float(np.max(np.abs(injection-delivered)));assert error<1e-6;delivery_max=max(delivery_max,error)
        result=(b[ANCHOR].astype(np.float64),y[ANCHOR].astype(np.float64),injection)
        assert np.array_equal(bf16(result[0]),result[0])
        cache[key]=result;return result
    scenarios=[]
    oldtrain=['r4_01','r4_02','r4_03','r4_04'];oldtest=['r4_05','r4_06','r4_07','r4_08']
    panel=json.loads((matched/'native/PANEL.json').read_text())
    for arm,newarm in [('existing_vector','plus_v'),('random_vector','random1')]:
        training=[read(original,'native',p,arm) for p in oldtrain]
        tests=[dict(dataset='original',cohort=c,prompt=p,group=p,condition=arm,data=read(original,c,p,arm))
            for c in ['native','replication_seed_42','replication_seed_43','replication_seed_44'] for p in oldtest]
        tests += [dict(dataset='older_matched_OOD',cohort='native',prompt=p['id'],group=p['pair'],condition=newarm,data=read(matched,'native',p['id'],newarm)) for p in panel['prompts']]
        scenarios.append(('original_'+arm,training,tests))
    trainp=[p for p in panel['prompts'] if p['pair'] in ['sum_record','inventory_record']]
    testp=[p for p in panel['prompts'] if p['pair']=='fragment_record']
    for arm in ['plus_v','minus_v','random1','random2','norm_only']:
        training=[read(matched,'native',p['id'],arm) for p in trainp]
        tests=[dict(dataset='matched_one_pair_TEST',cohort='native',prompt=p['id'],group=p['pair'],condition=arm,data=read(matched,'native',p['id'],arm)) for p in testp]
        scenarios.append(('matched_'+arm,training,tests))
    rows=[];structures=[];fits=[];synthetic_rows=[];synthetic_gates=[]
    def evaluate(name,model,tests,control):
        for test in tests:
            x,y,injection=test['data'];delta=y-x
            for kind in KINDS:
                prediction=predict(x,model,kind)
                for phase,sl in [('all',slice(None)),('prefill',slice(0,1)),('decode',slice(1,None))]:
                    err=delta[sl]-prediction[sl]
                    rows.append(dict(fit=name,control=control,dataset=test['dataset'],cohort=test['cohort'],prompt=test['prompt'],group=test['group'],condition=test['condition'],model=kind,phase=phase,
                        positions=len(x[sl]),displacement_score=score(delta[sl],prediction[sl]),
                        mean_actual_displacement=float(np.linalg.norm(delta[sl],axis=1).mean()),
                        prediction_vector_RMSE=float(np.sqrt(np.square(err).sum(axis=1).mean())),
                        actual_injection_normalized_RMSE=float(np.sqrt(np.mean(np.square(err).sum(axis=1)/injection[sl]**2)))))
            if control=='paired':structures.append(dict(fit=name,dataset=test['dataset'],cohort=test['cohort'],prompt=test['prompt'],group=test['group'],condition=test['condition'],**structure(x,y,model)))
        if control=='paired':
            for dataset,cohort in dict.fromkeys((t['dataset'],t['cohort']) for t in tests):
                selected=[t for t in tests if (t['dataset'],t['cohort'])==(dataset,cohort)]
                x=np.concatenate([t['data'][0] for t in selected]);y=np.concatenate([t['data'][1] for t in selected])
                structures.append(dict(fit=name,dataset=dataset,cohort=cohort,prompt='POOLED_CONTEXTS',group='descriptive_only',condition=selected[0]['condition'],**structure(x,y,model)))
    for name,training,tests in scenarios:
        print('FIT',name,flush=True)
        pairs=[(x,y) for x,y,a in training];model=fit(pairs)
        fits.append(dict(fit=name,control='paired',**{k:v for k,v in model.items() if not isinstance(v,np.ndarray)}))
        evaluate(name,model,tests,'paired')
        if name=='original_existing_vector':
            np.savez_compressed(output/'MAIN_PLUS_MAPS.npz',**{k:v for k,v in model.items() if isinstance(v,np.ndarray) or isinstance(v,(float,int))})
        lengths={len(x) for x,y in pairs};assert len(lengths)==1
        for offset in [1,2,3]:
            sham=[(x,x+(pairs[(i+offset)%4][1]-pairs[(i+offset)%4][0])) for i,(x,y) in enumerate(pairs)]
            sm=fit(sham);control='whole_prompt_cyclic_shift_'+str(offset)
            fits.append(dict(fit=name,control=control,**{k:v for k,v in sm.items() if not isinstance(v,np.ndarray)}))
            evaluate(name,sm,tests,control)
    # Pure known full-state transforms, no unlabelled retained background component.
    trainx=[read(original,'native',p,'existing_vector')[0] for p in oldtrain]
    testx=[read(original,'native',p,'existing_vector')[0] for p in oldtest]
    mu=np.concatenate(trainx).mean(axis=0);v,_,_=basis(np.concatenate(trainx)-mu);plane=v[:,:2]
    angle=.2;r2=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    direction=np.random.default_rng(20261011).standard_normal(1024);direction/=np.linalg.norm(direction)
    def planted(x,kind):
        xc=x-mu
        if kind=='shift':return x+.5*direction
        if kind in ['proper_rotation','scale_plus_rotation']:
            rotated=xc+(xc@plane)@(r2-np.eye(2))@plane.T
            return mu+(1.1 if kind=='scale_plus_rotation' else 1)*rotated
        if kind=='anisotropic_stretch':return x+(xc@plane)@np.diag([.15,-.15])@plane.T
        return x-2*(xc@plane[:,:1])@plane[:,:1].T
    chosen={'shift':'translation','proper_rotation':'rigid_isometry','scale_plus_rotation':'scaled_isometry','anisotropic_stretch':'ridge_affine_deformation','reflection':'rigid_isometry'}
    for kind in chosen:
        for precision in ['float64','BF16']:
            transform=lambda x:bf16(planted(x,kind)) if precision=='BF16' else planted(x,kind)
            model=fit([(x,transform(x)) for x in trainx]);print('SYNTHETIC',kind,precision,flush=True)
            scores=[]
            for pid,x in zip(oldtest,testx):
                y=transform(x)
                for method in KINDS:
                    value=score(y-x,predict(x,model,method));assert method!='identity' or value==0
                    synthetic_rows.append(dict(control=kind,precision=precision,prompt=pid,model=method,displacement_score=value,
                        chosen_joint_determinant=model['chosen_joint_determinant'],**structure(x,y,model)))
                    if method==chosen[kind]:scores.append(value)
            synthetic_gates.append(dict(control=kind,precision=precision,required_model=chosen[kind],minimum_prompt_score=min(scores),passes_ideal_0_95=min(scores)>=.95))
    for name,data in [('DISPLACEMENT_PREDICTIONS.csv',rows),('CLOUD_STRUCTURE.csv',structures),('SYNTHETIC_CONTROLS.csv',synthetic_rows)]:write_csv(output/name,data)
    result={'date':'2026-10-11','fixed_block_zero_based':ANCHOR,'full_coordinate_width':1024,
        'plan_sha256':sha(source/'PLAN.json'),'source_sha256':sha(Path(__file__)),'consumed_native_files':len(used),
        'all_consumed_hashes_and_replayed_tokens_verified':True,'upstream_exactly_unchanged':True,
        'actual_delivered_update_receipt_max_error':delivery_max,'fresh_eight_question_geometry_data_used':False,
        'new_generator_inference_or_GPU':False,'fit_diagnostics':fits,'synthetic_gates':synthetic_gates,
        'all_required_float64_synthetic_models_pass':all(g['passes_ideal_0_95'] for g in synthetic_gates if g['precision']=='float64'),
        'interpretation':'Exploratory static cloud maps. Scores predict actual full-coordinate displacement. Unobserved ambient actions and global orientation are not identified.'}
    (output/'RESULTS.json').write_text(json.dumps(result,indent=2));(output/'INPUT_HASHES.json').write_text(json.dumps(used,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='fit_diagnostics'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('original',type=Path);p.add_argument('matched',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();run(a.original,a.matched,a.output)

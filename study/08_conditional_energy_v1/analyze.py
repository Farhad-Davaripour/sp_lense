"""BASE-only conditional energies; whole-family TRAIN/calibration/TEST isolation."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

HERE=Path(__file__).parent
OUT=Path('C:/Users/farha/Research4/artifacts/conditional_energy_v1/analysis_20261012')
ORIGINAL=Path('C:/Users/farha/Research4/artifacts/rotation_pilot_v1/executed_20261009/verified_native')
MATCHED=Path('C:/Users/farha/Research4/artifacts/propagation_followup_v1/recovery_20261010/verified_export/verified_native')
KINDS=['unconditional_diagonal','predictive_innovation','strain_compatibility']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def csvwrite(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def profile(h):
    norm=np.linalg.norm(h,axis=-1)
    cosine=np.sum(h[1:]*h[:-1],axis=-1)/(norm[1:]*norm[:-1])
    return np.stack([np.log(norm[1:]/norm[:-1]),1-np.clip(cosine,-1,1)],axis=-1)

def features(h,kind):
    """Only within-run residuals enter the candidate; no pre-hook or BASE pair."""
    assert h.shape==(24,8,1024) and np.isfinite(h).all()
    if kind=='strain_compatibility':
        p=profile(h)
        return p[:9].transpose(1,0,2).reshape(8,-1),p[9:].transpose(1,0,2).reshape(8,-1)
    norm=np.linalg.norm(h[9],axis=1,keepdims=True)
    assert np.min(norm)>1e-12
    x=np.concatenate([h[9]/norm,(h[9]-h[8])/norm],axis=1)
    y=((h[10:]-h[9][None,:,:])/norm[None,:,:]).transpose(1,0,2).reshape(8,-1)
    return x,y

def ridge(x,y):
    mx=x.mean(axis=0);my=y.mean(axis=0);xc=x-mx
    lam=max(1e-12,.1*float(np.mean(np.sum(xc*xc,axis=1))))
    dual=np.linalg.solve(xc@xc.T+lam*np.eye(len(x)),y-my)
    return dict(mx=mx,my=my,xc=xc,dual=dual,lam=lam)

def predict(x,m):return m['my']+((x-m['mx'])@m['xc'].T)@m['dual']

def err_energy(e,m):
    ec=e-m['bias']
    if m['precision'].ndim==1:return np.mean(ec*ec*m['precision'],axis=1)
    return np.sum((ec@m['precision'])*ec,axis=1)/ec.shape[1]

def fit_conditional(pairs,kind):
    errors=[]
    for i,(x,y) in enumerate(pairs):
        others=[p for j,p in enumerate(pairs) if j!=i]
        law=ridge(np.concatenate([p[0] for p in others]),np.concatenate([p[1] for p in others]))
        errors.append(y-predict(x,law))
    errors=np.concatenate(errors);bias=errors.mean(axis=0);ec=errors-bias
    var=np.mean(ec*ec,axis=0);tau=float(var.mean())
    assert tau>1e-20
    if kind=='strain_compatibility':
        cov=ec.T@ec/len(ec)
        precision=np.linalg.inv(.9*cov+.1*tau*np.eye(len(bias)))
    else:precision=1/(.9*var+.1*tau)
    law=ridge(np.concatenate([p[0] for p in pairs]),np.concatenate([p[1] for p in pairs]))
    m=dict(law=law,bias=bias,precision=precision,uncertainty_samples=len(errors),tau=tau)
    m['calibration']=float(err_energy(errors,m).mean())
    assert m['calibration']>0
    return m

def fit_unconditional(h):
    # h: [TRAIN positions,24,1024]. Each layer has its own calibration.
    mu=h.mean(axis=0);xc=h-mu;var=np.mean(xc*xc,axis=0)
    precision=1/(.9*var+.1*var.mean(axis=1,keepdims=True))
    calibration=np.mean(np.mean(xc*xc*precision,axis=2),axis=0)
    assert np.min(calibration)>0
    return dict(mu=mu,precision=precision,calibration=calibration)

def phase_score(h,kind,m,sl):
    if kind=='unconditional_diagonal':
        a=h[:,sl].transpose(1,0,2)-m['mu']
        return float(np.mean(np.mean(a*a*m['precision'],axis=2)/m['calibration']))
    x,y=features(h,kind)
    error=y[sl]-predict(x[sl],m['law'])
    return float(np.mean(err_energy(error,m))/m['calibration'])

def score(h,kind,models):
    a=phase_score(h,kind,models['prefill'],slice(0,1))
    b=phase_score(h,kind,models['decode'],slice(1,8))
    return (a+7*b)/8,a,b

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    for name in ['PLAN.json','LITERATURE.json','analyze.py']:shutil.copyfile(HERE/name,OUT/name)
    plan=json.loads((HERE/'PLAN.json').read_text());families=plan['families_in_fixed_order'];names=list(families)
    manifests={str(r):json.loads((r/'EXPORT_HASHES.json').read_text()) for r in [ORIGINAL,MATCHED]}
    used={};cache={};rows=[];fits=[];controls={}
    def root_for(pid):return ORIGINAL if pid.startswith('r4_') else MATCHED
    def read(pid,arm='base',cohort='native'):
        key=(pid,arm,cohort)
        if key in cache:return cache[key]
        root=root_for(pid);p=root/cohort/pid/(arm+'.npz');digest=sha(p)
        assert digest==manifests[str(root)][p.relative_to(root).as_posix()]
        used[str(p)]=digest
        with np.load(p,allow_pickle=False) as z:
            h=z['residuals'][:,:8,:].astype(np.float64)
            tokens=z['generated_token_ids' if 'generated_token_ids' in z else 'forced_token_ids'][:8].copy()
        assert h.shape==(24,8,1024) and np.isfinite(h).all()
        cache[key]=(h,tokens);return cache[key]
    exemplar=read('r4_01')[0]
    perm=np.random.default_rng(20261012).permutation(1024)
    signs=np.where(np.arange(1024)%2,1.,-1.)
    transformed=exemplar[:,:,perm]*signs
    controls['strain_orthogonal_invariance_max_error']=float(np.max(np.abs(profile(transformed)-profile(exemplar))))
    controls['strain_common_scale_invariance_max_error']=float(np.max(np.abs(profile(3.7*exemplar)-profile(exemplar))))
    for error in controls.values():assert error<1e-12
    for i,test in enumerate(names):
        calibration=names[(i+1)%len(names)]
        trainids=[pid for family in names if family not in [test,calibration] for pid in families[family]]
        bases=[read(pid)[0] for pid in trainids]
        assert len(bases)>=4
        for kind in KINDS:
            models={}
            for phase,sl in [('prefill',slice(0,1)),('decode',slice(1,8))]:
                if kind=='unconditional_diagonal':
                    m=fit_unconditional(np.concatenate([h[:,sl].transpose(1,0,2) for h in bases]))
                else:
                    pairs=[]
                    for h in bases:
                        x,y=features(h,kind);pairs.append((x[sl],y[sl]))
                    m=fit_conditional(pairs,kind)
                models[phase]=m
            cal_scores=[score(read(pid)[0],kind,models)[0] for pid in families[calibration]]
            threshold=max(cal_scores)
            fits.append(dict(TEST_family=test,calibration_family=calibration,TRAIN_prompts=trainids,
                calibration_prompts=families[calibration],model=kind,threshold=threshold,
                calibration_BASE_scores=cal_scores,phase_details={phase:{
                    'uncertainty_samples':m.get('uncertainty_samples'),
                    'ridge_lambda':m.get('law',{}).get('lam'),
                    'TRAIN_error_energy_calibration':float(m['calibration']) if np.ndim(m['calibration'])==0 else m['calibration'].tolist()
                } for phase,m in models.items()}))
            for pid in families[test]:
                original=pid.startswith('r4_')
                cohorts=['native','replication_seed_42','replication_seed_43','replication_seed_44'] if original else ['native']
                arms=['base','existing_vector','random_vector'] if original else ['base','plus_v','minus_v','random1','random2','norm_only']
                for cohort in cohorts:
                    b,bt=read(pid,cohort=cohort)
                    for arm in arms:
                        h,tokens=read(pid,arm,cohort)
                        assert np.array_equal(bt,tokens) and np.array_equal(b[:10],h[:10])
                        e,prefill,decode=score(h,kind,models)
                        rows.append(dict(family=test,prompt=pid,cohort=cohort,condition=arm,model=kind,
                            energy=e,prefill_energy=prefill,decode7_energy=decode,threshold=threshold,
                            prediction=int(e>threshold),is_modified=int(arm!='base')))
            print('FIT/SCORE',test,kind,flush=True)
    instrumented=[]
    for pid in families['matched_records']:
        for arm in ['base','plus_v','minus_v','random1','random2','norm_only']:
            p=MATCHED/'native'/pid/(arm+'.npz')
            with np.load(p,allow_pickle=False) as z:
                residual=z['residuals'][10].astype(np.float64)-z['pre_injection'].astype(np.float64)
            e=.5*np.sum(residual*residual,axis=1)
            instrumented.append(dict(prompt=pid,condition=arm,energy_min=float(e.min()),energy_max=float(e.max()),
                energy_mean=float(e.mean()),prediction=int(np.max(e)>0),is_modified=int(arm!='base'),positions=len(e),
                zero_positions=int(np.sum(e==0))))
    summary=[]
    for kind in KINDS:
        for family in names+['ALL_NATIVE']:
            a=[r for r in rows if r['model']==kind and r['cohort']=='native' and (family=='ALL_NATIVE' or r['family']==family)]
            base=[r for r in a if r['condition']=='base'];positive=[r for r in a if r['condition']!='base']
            perarm={}
            for arm in dict.fromkeys(r['condition'] for r in positive):
                z=[r for r in positive if r['condition']==arm]
                perarm[arm]=dict(detected=sum(r['prediction'] for r in z),total=len(z))
            comparisons=np.array([r['energy']-b['energy'] for r in positive for b in base])
            summary.append(dict(model=kind,family=family,BASE_false_positives=sum(r['prediction'] for r in base),BASE_total=len(base),
                modified_detected=sum(r['prediction'] for r in positive),modified_total=len(positive),per_condition=perarm,
                descriptive_AUROC_high_energy=float(np.mean((comparisons>0)+.5*(comparisons==0)))))
    integrity={}
    for root in [ORIGINAL,MATCHED]:
        for member,digest in manifests[str(root)].items():assert sha(root/member)==digest
        integrity[str(root)]=len(manifests[str(root)])
    controls['exact_upstream_residuals_and_token_identity']=True
    controls['complete_native_manifest_files_verified']=integrity
    controls['fresh_panel_accessed']=False;controls['GPU_or_model_calls']=False
    controls['instrumented_same_run_correct_traces']=sum(r['prediction']==r['is_modified'] for r in instrumented)
    controls['instrumented_total_traces']=len(instrumented)
    csvwrite(OUT/'TRACE_SCORES.csv',rows);csvwrite(OUT/'INSTRUMENTED_RECEIPT_SCORES.csv',instrumented)
    dump(OUT/'FITS_AND_SPLITS.json',fits);dump(OUT/'USED_SOURCE_HASHES.json',used)
    dump(OUT/'RESULTS.json',dict(plan_sha256=sha(HERE/'PLAN.json'),implementation_sha256=sha(HERE/'analyze.py'),
        scope='Exploratory family-withheld BASE-only energy candidates; same-run instrumentation separate',controls=controls,
        primary_native_family_summary=summary))
    print('DONE',OUT,flush=True)

if __name__=='__main__':main()

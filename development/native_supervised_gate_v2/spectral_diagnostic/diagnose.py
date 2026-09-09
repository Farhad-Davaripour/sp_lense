"""One saved-only 32-row spectral diagnostic. No fit or alternate predictor."""
import json,math,sys,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sys.path.insert(0,str(BASE))
import gate,source_auth
ARTIFACT_SHA='4e8a282be46cb8008eb8fef5b89e18fcd50349e54ca7c58b67a12e24f62a70cb'
SCOPE_SHA='323e890360ba94a119b18d0d885d411099ecb0b0d95a2084998f360184ee88d9'
TOL=1e-10
NULL_TOL=1e-12
def check(ok,name):
    if not ok:raise ValueError(name)
def run():
    started=time.monotonic();deadline=started+150
    check(not (HERE/'RESULT.json').exists(),'ONE_DIAGNOSTIC_ONLY')
    check(gate.digest((BASE/'ROOT_SPECTRAL_DIAGNOSTIC_SCOPE.md').read_bytes())==SCOPE_SHA,'SCOPE_HASH')
    manifest_raw=(BASE/'TRAINING_MANIFEST.json').read_bytes();manifest=gate.decode(manifest_raw)
    rows,labels=source_auth.extract_features(manifest,deadline=deadline)
    bindings={'training_manifest_sha256':gate.digest(manifest_raw),
        'construction_lock_sha256':gate.digest((BASE/'CONSTRUCTION_LOCK_DRAFT.json').read_bytes()),
        'feature_sha256':gate.digest(gate.canonical({'rows':rows,'labels':labels})),
        'source_sha256':gate.digest((BASE/'CORE_SOURCE_LOCK.json').read_bytes())}
    model=gate.load_artifact((BASE/'construction_attempt_001/FITTED_GATE.json').read_bytes(),
        expected_sha256=ARTIFACT_SHA,expected_bindings=bindings)
    mu=tuple(math.fsum(h[j] for h in rows)/32 for j in range(1024))
    check(mu==model.mu,'FROZEN_UNWEIGHTED_MEAN')
    x=tuple(gate.centered_unit(h,mu) for h in rows)
    a=tuple(1/(2*labels.count(y)) for y in labels)
    xbar=tuple(math.fsum(a[i]*x[i][j] for i in range(32)) for j in range(1024))
    z=tuple(tuple(math.sqrt(a[i])*(x[i][j]-xbar[j]) for j in range(1024)) for i in range(32))
    gram=np.array([[gate.dot(z[i],z[j]) for j in range(32)] for i in range(32)],dtype=np.float64)
    Z=np.asarray(z,dtype=np.float64)
    independent_gram=float(np.max(np.abs(gram-Z@Z.T)))
    symmetry=float(np.max(np.abs(gram-gram.T)))
    eta,U=np.linalg.eigh(gram)  # The only eigendecomposition; no fitting solve.
    order=np.argsort(eta)[::-1];eta=eta[order];U=U[:,order]
    residual=float(np.max(np.abs(gram@U-U*eta)))
    reconstruction=float(np.max(np.abs(gram-(U*eta)@U.T)))
    orthogonality=float(np.max(np.abs(U.T@U-np.eye(32))))
    trace_error=abs(float(np.trace(gram))-math.fsum(float(v) for v in eta))
    check(min(eta)>=-NULL_TOL,'PSD_BEYOND_ROUNDOFF')
    check(max(independent_gram,symmetry,residual,reconstruction,orthogonality,trace_error)<=TOL,'NUMERICAL_CHECK')
    retained=eta>NULL_TOL
    V=Z.T@U[:,retained]/np.sqrt(eta[retained])
    feature_orthogonality=float(np.max(np.abs(V.T@V-np.eye(int(sum(retained))))))
    check(feature_orthogonality<=1e-8,'FEATURE_DIRECTIONS_ORTHOGONAL')
    ix={s['case']:i for i,s in enumerate(manifest['selection'])};contrasts=[];names=[]
    for family in ('G01','G02','G03','G04'):
        for answer_order in ('KEEP_then_STOP','STOP_then_KEEP'):
            i=ix[family+'_self_shutdown__'+answer_order];j=ix[family+'_other_shutdown__'+answer_order]
            contrasts.append(np.asarray(x[i])-np.asarray(x[j]));names.append(family+'__'+answer_order)
    C=np.asarray(contrasts);energy=(C@V)**2;totals=np.sum(C*C,axis=1)
    coverage_error=float(np.max(np.abs(np.sum(energy,axis=1)-totals)))
    check(coverage_error<=TOL,'CONTRAST_ENERGY_RECONSTRUCTION')
    attenuation=np.maximum(eta[retained],0)/(np.maximum(eta[retained],0)+.1)
    aggregate=np.sum(energy,axis=0);total=float(np.sum(totals))
    mode_rows=[];j=0
    for k,value in enumerate(eta):
        active=bool(retained[k]);e=float(aggregate[j]) if active else 0.
        mode_rows.append({'mode':k+1,'eta_raw':float(value),'near_zero':not active,
            'fixed_attenuation':float(value/(value+.1)) if active else 0.,
            'contrast_energy_sum':e,'contrast_energy_fraction':e/total,
            'per_pair_energy':[float(v) for v in energy[:,j]] if active else None})
        j+=active
    pair_rows=[{'pair':n,'total_unit_contrast_energy':float(totals[i]),
        'covered_energy':float(np.sum(energy[i])),
        'attenuation_weighted_amplitude':float(np.sum(energy[i]*attenuation)/totals[i]),
        'attenuated_energy_fraction':float(np.sum(energy[i]*attenuation**2)/totals[i])}
        for i,n in enumerate(names)]
    check(time.monotonic()<deadline,'DIAGNOSTIC_DEADLINE')
    check(not {'torch','transformers','tokenizers','safetensors'}&{n.split('.')[0] for n in sys.modules},'NO_PROVIDERS')
    result={'status':'PASS_SAVED_ONLY','script_sha256':gate.digest(Path(__file__).read_bytes()),
        'scope_sha256':SCOPE_SHA,'artifact_sha256':ARTIFACT_SHA,'bindings':bindings,
        'numpy_version':np.__version__,'numpy_path':np.__file__,'rows':32,'width':1024,'lambda':.1,
        'eigendecompositions':1,'fit_calls':0,'model_calls':0,'tokenizer_calls':0,
        'null_tolerance':NULL_TOL,'consistency_tolerance':TOL,'feature_direction_tolerance':1e-8,
        'checks':{'gram_fsum_vs_blas_max_abs':independent_gram,'symmetry_max_abs':symmetry,
            'eigenpair_max_abs':residual,'reconstruction_max_abs':reconstruction,
            'orthogonality_max_abs':orthogonality,'trace_abs':trace_error,
            'feature_direction_orthogonality_max_abs':feature_orthogonality,
            'contrast_energy_coverage_max_abs':coverage_error},
        'retained_modes':int(sum(retained)),'near_zero_modes':int(sum(~retained)),
        'total_matched_contrast_energy':total,
        'aggregate_attenuation_weighted_amplitude':float(np.sum(aggregate*attenuation)/total),
        'aggregate_attenuated_energy_fraction':float(np.sum(aggregate*attenuation**2)/total),
        'modes':mode_rows,'pairs':pair_rows,'elapsed_seconds':time.monotonic()-started}
    raw=gate.canonical(result)
    check(sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())+len(raw)<=1024**2,'DIAGNOSTIC_STORAGE')
    with (HERE/'RESULT.json').open('xb') as out:out.write(raw)
    print(json.dumps(result,allow_nan=False))
if __name__=='__main__':run()

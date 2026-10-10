"""Fixed-rank causal sensitivity check and bounded surrogate falsification."""
import argparse
import json
from pathlib import Path
import numpy as np


def bf16(h):
    bits=np.asarray(h,dtype=np.float32).view(np.uint32).copy()
    bits += np.uint32(0x7fff)+((bits>>16)&1)
    bits &= np.uint32(0xffff0000)
    return bits.view(np.float32).astype(np.float64)


def pca(train):
    states=np.concatenate(train)
    center=states.mean(axis=0)
    states=states-center
    values,axes=np.linalg.eigh(states@states.T)
    selected=np.argsort(values)[-4:][::-1]
    basis=states.T@(axes[:,selected]/np.sqrt(values[selected]))
    return center,basis,float(values[selected].sum()/values.clip(0).sum()),values[selected]/len(states)


def pairs(trajectories,center,basis):
    x=np.concatenate([(h[:-1]-center)@basis for h in trajectories])
    y=np.concatenate([(h[1:]-center)@basis for h in trajectories])
    return x,y


def operators(train,center,basis):
    x,y=pairs(train,center,basis)
    eye=np.eye(4)
    u,_,vt=np.linalg.svd(x.T@y)
    sign=np.eye(4);sign[-1,-1]=np.linalg.det(u@vt)
    rotation=u@sign@vt
    m=(x+y)/2;v=y-x;cov=m.T@m
    eigen,axes=np.linalg.eigh(cov);ridge=.01*max(float(eigen.mean()),1e-30)
    cross=m.T@v;rhs=axes.T@(cross-cross.T)@axes
    a=axes@(rhs/(eigen[:,None]+eigen[None,:]+2*ridge))@axes.T
    cayley=np.linalg.solve((eye-a/2).T,(eye+a/2).T).T
    covariance=x.T@x;ridge_free=.01*max(float(np.trace(covariance)/4),1e-30)
    free=np.linalg.solve(covariance+ridge_free*eye,x.T@y)
    contraction=float(np.clip(np.sum(x*y)/np.square(x).sum(),0,1))
    result={'proper_rotation':(rotation,np.zeros(4)), 'midpoint_cayley':(cayley,np.zeros(4)),
            'unrestricted_linear':(free,np.zeros(4)), 'persistence':(eye,np.zeros(4)),
            'constant_drift':(eye,(y-x).mean(axis=0)), 'contraction':(contraction*eye,np.zeros(4))}
    assert np.linalg.norm(rotation.T@rotation-eye)<1e-10 and np.linalg.det(rotation)>0
    assert np.linalg.norm(cayley.T@cayley-eye)<1e-10
    return result


def evaluate(train,test,center,basis):
    methods=operators(train,center,basis)
    scores={name:[] for name in methods}
    projected={name:[] for name in methods}
    coverage=[]
    for h in test:
        x,y=h[:-1],h[1:];z=(x-center)@basis;change=y-x
        denominator=np.square(change).sum();pc_denominator=np.square(change@basis).sum()
        coverage.append(float(pc_denominator/denominator))
        for name,(operator,drift) in methods.items():
            increment=z@(operator-np.eye(4))+drift
            predicted=x+increment@basis.T  # All remaining coordinates persist.
            scores[name].append(float(1-np.square(y-predicted).sum()/denominator))
            projected[name].append(float(1-np.square(change@basis-increment).sum()/pc_denominator))
    return {'full_space_score_vs_persistence':{k:float(np.mean(v)) for k,v in scores.items()},
            'per_prompt_full_space_scores':scores,'projected_score_vs_persistence':{k:float(np.mean(v)) for k,v in projected.items()},
            'per_prompt_velocity_energy_fraction_in_four_PCs':coverage,
            'proper_rotation_determinant':float(np.linalg.det(methods['proper_rotation'][0]))}


def shared_phase(h,rng):
    mean=h.mean(axis=0)
    spectrum=np.fft.rfft(h-mean,axis=0)
    phases=np.exp(1j*rng.uniform(-np.pi,np.pi,len(spectrum)))
    phases[0]=1
    if len(h)%2==0:phases[-1]=1
    return np.fft.irfft(spectrum*phases[:,None],n=len(h),axis=0)+mean


def surrogate_validation():
    # A counterexample sufficient to disqualify this exact covariance-preserving null.
    rng=np.random.default_rng(20261011);t=np.arange(64)
    h=np.column_stack([np.cos(t*np.pi/16),np.sin(t*np.pi/16),np.cos(t*np.pi/8),np.sin(t*np.pi/8)])
    methods=operators([h]*4,np.zeros(4),np.eye(4))
    operator=methods['proper_rotation'][0]
    original=[h.T@np.roll(h,-lag,axis=0)/len(h) for lag in range(9)]
    maximum_error=0.;retained=[];quantized_error=0.
    for _ in range(199):
        surrogate=shared_phase(h,rng)
        for lag,reference in enumerate(original):
            covariance=surrogate.T@np.roll(surrogate,-lag,axis=0)/len(h)
            maximum_error=max(maximum_error,float(np.max(np.abs(covariance-reference))))
            recorded=bf16(surrogate)
            quantized_error=max(quantized_error,float(np.max(np.abs(recorded.T@np.roll(recorded,-lag,axis=0)/len(h)-reference))))
        error=np.square(surrogate[1:]-surrogate[:-1]@operator).sum()
        retained.append(float(1-error/np.square(np.diff(surrogate,axis=0)).sum()))
    assert maximum_error<1e-12
    return {'candidate':'shared Fourier phase across channels','repetitions':199,
            'preserves':'full circular cross-lag covariance in exact arithmetic, demonstrated at lags 0..8',
            'maximum_cross_covariance_absolute_error':maximum_error,
            'maximum_after_BF16_recording_cross_covariance_absolute_error':quantized_error,
            'planted_rotation_causal_score_range':[min(retained),max(retained)],
            'rejected':min(retained)>.99,'reason':'Preserves the coordinated rotation being tested, not just nuisance covariance.',
            'real_data_surrogate_p_value':None,'validated_rotation_destroying_null_available':False,
            'no_alternative_null_search':True}


def run(root,output):
    output.mkdir(parents=True,exist_ok=False)
    config=json.loads((root/'source/pilot.json').read_text());cohorts=['native','replication_seed_42','replication_seed_43','replication_seed_44']
    result={'fixed_rank':4,'block_zero_based':15,'forecast':'causal one-step, full width, common persistence outside baseline TRAIN PCs','real':{},'synthetic':[]}
    native=[]
    for cohort in cohorts:
        arms={}
        for arm in config['conditions']:
            traces=[]
            for prompt in config['prompts']:
                with np.load(root/cohort/prompt['id']/(arm+'.npz')) as z:traces.append(z['residuals'][15].astype(np.float64))
            arms[arm]=traces
        center,basis,fraction,eigen=pca(arms['base'][:4])
        # One baseline-TRAIN basis per cohort, shared by all its conditions.
        result['real'][cohort]={'baseline_TRAIN_state_variance_fraction_four_PCs':fraction,'conditions':{}}
        for arm,traces in arms.items():result['real'][cohort]['conditions'][arm]=evaluate(traces[:4],traces[4:],center,basis)
        if cohort=='native':native=arms['base'];reference=(center,basis,eigen)
    center,basis,eigen=reference
    offsets=[(h.mean(axis=0)-center)@ (np.eye(1024)-basis@basis.T) for h in native]
    outside=[(h-center)-((h-center)@basis)@basis.T-offset for h,offset in zip(native,offsets)]
    sd=np.sqrt(eigen);paired_sd=np.repeat(np.sqrt([eigen[:2].mean(),eigen[2:].mean()]),2)
    rng=np.random.default_rng(20261010)
    for kind in ['rotation','drift','contraction','correlated_nonrotation']:
        for amplitude in [0.,.25,1.]:
            traces=[]
            for i,h in enumerate(native):
                n=len(h);t=np.arange(n);phase=rng.uniform(-np.pi,np.pi,2)
                if kind=='rotation':
                    latent=np.column_stack([np.cos(t*np.pi/16+phase[0]),np.sin(t*np.pi/16+phase[0]),np.cos(t*np.pi/8+phase[1]),np.sin(t*np.pi/8+phase[1])])*paired_sd*np.sqrt(2)
                elif kind=='drift':latent=rng.normal(size=4)*sd+t[:,None]*.04*sd
                elif kind=='contraction':latent=(.97**t[:,None])*(rng.normal(size=4)*sd*2)
                else:
                    latent=np.empty((n,4));latent[0]=rng.normal(size=4)*sd
                    for j in range(1,n):latent[j]=.85*latent[j-1]+np.sqrt(1-.85**2)*rng.normal(size=4)*sd
                noise=np.empty((n,4));noise[0]=rng.normal(size=4)*sd
                for j in range(1,n):noise[j]=.7*noise[j-1]+np.sqrt(1-.7**2)*rng.normal(size=4)*sd
                values=center+offsets[i]+(latent+amplitude*noise)@basis.T+outside[i]
                traces.append(bf16(values))
            synthetic_center,synthetic_basis,fraction,_=pca(traces[:4])
            scores=evaluate(traces[:4],traces[4:],synthetic_center,synthetic_basis)
            result['synthetic'].append({'type':kind,'noise_amplitude':amplitude,'TRAIN_state_variance_fraction_four_PCs':fraction,**scores})
    result['surrogate_validation']=surrogate_validation()
    pure=next(r for r in result['synthetic'] if r['type']=='rotation' and r['noise_amplitude']==0)
    recovered={}
    for method in ['proper_rotation','midpoint_cayley']:
        recovered[method]=sum(score>max(0,pure['per_prompt_full_space_scores']['constant_drift'][i],pure['per_prompt_full_space_scores']['contraction'][i]) for i,score in enumerate(pure['per_prompt_full_space_scores'][method]))
    result['sensitivity_pure_rotation_prompts_recovered_of_four']=recovered
    result['rotation_branch_status']='inconclusive_validated_surrogate_missing_stop_branch'
    result['no_layer_rank_or_null_search']=True
    (output/'ROTATION_VALIDATION.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();run(args.root,args.output)

"""Portable score replay, selected-configuration refit and optional original search."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier
ROOT=Path(__file__).resolve().parent/'artifacts'
def read(p):return json.loads(p.read_text())
def metrics(y,p,t):
    pred=p>=t; y=np.asarray(y,bool); tp=int((y&pred).sum()); fp=int((~y&pred).sum()); fn=int((y&~pred).sum()); tn=int((~y&~pred).sum())
    return dict(tp=tp,tn=tn,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
def engineered(name,z,j,norm):
    j3=j.reshape(-1,3,6)
    contrasts=np.stack([j3[:,:,1]-j3[:,:,2],j3[:,:,3]-j3[:,:,2],j3[:,:,5]-j3[:,:,0]],axis=2).reshape(-1,9)
    if name=='pca_only':return z
    if name=='raw_jlens':return np.c_[z,j]
    if name=='pca_centered_norm':return np.c_[z,(j3-j3.mean(axis=2,keepdims=True)).reshape(-1,18),norm]
    if name=='pca_raw_norm':return np.c_[z,j,norm]
    if name=='jlens_contrasts_norm':return np.c_[j,contrasts,norm]
    if name=='pca_contrasts_norm':return np.c_[z,contrasts,norm]
    raise ValueError(name)
def project(pca,x):return (x-pca['mean_'])@pca['components_'][:32].T
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--refit',action='store_true');parser.add_argument('--tune',action='store_true');args=parser.parse_args()
    manifest=read(ROOT/'SHA256.json')
    for name,h in manifest.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h, f'Integrity mismatch: {name}'
    d=np.load(ROOT/'features.npz',allow_pickle=False); meta=read(ROOT/'cases.json'); expected=read(ROOT/'expected.json')
    train=np.array(meta['train_mask'],bool); folds=np.array(meta['folds']); y=np.isin(meta['development_labels'],['SELF','OTHER']); hy=np.isin(meta['holdout_labels'],['SELF','OTHER'])
    summary={}
    for run,name in [('xgboost_shutdown_v1','pca_only'),('xgboost_jlens_shutdown_v1','raw_jlens'),('xgboost_engineered_v1','pca_contrasts_norm')]:
        base=ROOT/'models'/run; frozen=read(base/'CANDIDATE_FREEZE.json'); pca=np.load(base/'pca.npz',allow_pickle=False)
        model=XGBClassifier();model.load_model(base/'model.ubj')
        vx=engineered(name,project(pca,d['dev_x'][~train]),d['dev_j'][~train],d['dev_norm'][~train])
        hx=engineered(name,project(pca,d['hold_x']),d['hold_j'],d['hold_norm'])
        vp=model.predict_proba(vx)[:,1];hp=model.predict_proba(hx)[:,1]
        assert np.max(np.abs(vp-np.array(expected[run]['validation_probabilities'])))<1e-7
        assert np.max(np.abs(hp-np.array(expected[run]['holdout_probabilities'])))<1e-7
        summary[run]={'validation':metrics(y[~train],vp,frozen['threshold']),'diagnostic_holdout':metrics(hy,hp,frozen['threshold'])}
        plan=read(base/'PLAN.json')
        if args.refit:
            pc=PCA(n_components=32,svd_solver='full',whiten=False).fit(d['dev_x'][train])
            pstate={'mean_':pc.mean_,'components_':pc.components_}
            fitted=XGBClassifier(**frozen['configuration'],**plan['parameters'])
            fitted.fit(engineered(name,project(pstate,d['dev_x'][train]),d['dev_j'][train],d['dev_norm'][train]),y[train].astype(int))
            new=fitted.predict_proba(engineered(name,project(pstate,d['dev_x'][~train]),d['dev_j'][~train],d['dev_norm'][~train]))[:,1]
            assert np.max(np.abs(new-vp))<1e-7, 'Refit mismatch'
            summary[run]['selected_configuration_refit']='PASS'
        if args.tune:
            trials=plan.get('trials') or [(name,c) for c in plan['configurations']]
            oof=np.full((len(trials),int(train.sum())),np.nan);xt=d['dev_x'][train];jt=d['dev_j'][train];nt=d['dev_norm'][train];yt=y[train].astype(int);ft=folds[train]
            for k in range(5):
                fit=ft!=k;test=~fit;pc=PCA(n_components=32,svd_solver='full',whiten=False).fit(xt[fit]);ps={'mean_':pc.mean_,'components_':pc.components_}
                a=project(ps,xt[fit]);b=project(ps,xt[test])
                for i,(feature,c) in enumerate(trials):
                    candidate=XGBClassifier(**c,**plan['parameters']);candidate.fit(engineered(feature,a,jt[fit],nt[fit]),yt[fit]);oof[i,test]=candidate.predict_proba(engineered(feature,b,jt[test],nt[test]))[:,1]
                print(f'{run}: tuning fold {k+1}/5 complete',flush=True)
            original=read(base/'CV_SEARCH.json')['oof_predictions']
            assert max(float(np.max(np.abs(oof[i]-np.array(original[str(i)])))) for i in range(len(trials)))<1e-7
            summary[run]['original_search_replay']='PASS'
    print(json.dumps(summary,indent=2))
if __name__=='__main__':
    with threadpool_limits(limits=1):main()

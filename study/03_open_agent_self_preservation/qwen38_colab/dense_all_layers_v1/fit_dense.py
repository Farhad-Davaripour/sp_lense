"""Fit all 64 case-balanced ridge functions, with no language-model optimizer."""
import argparse,json,pathlib,sys,time
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from common import read,save,sha,now,guard
from dense_math import fit,replay_fit,predict_numpy,FIELDS,FIT_RANK,INFERENCE_RANK

def main(a):
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    inputs,native,root=map(pathlib.Path,(a.inputs,a.native,a.root));root.mkdir(exist_ok=False)
    contexts=read(inputs/'DENSE_CONTEXTS.json');plan=read(inputs/'DENSE_FREEZE.json');done=read(native/'FEATURES_COMPLETE.json')
    assert read(native/'COMPLETE.json')['weights_unchanged'] and done['all_generated_positions']==4899
    active=np.concatenate([np.asarray(c['dense_active'],dtype=bool) for c in contexts]);w=np.concatenate([np.full(len(c['dense_active']),c['row_weight'],dtype=np.float32) for c in contexts])
    assert len(active)==4899 and active.sum()==717 and len(plan['layers'])==64
    assert plan['fit_rank']==FIT_RANK and plan['inference_rank']==INFERENCE_RANK
    results={};deadline=time.monotonic()+a.max_seconds
    for site in range(64):
        guard(deadline);x=[];y=[]
        for c in contexts:
            rel='features/layer_'+str(site)+'/context_'+str(c['context_index']).zfill(3)+'.npz';p=native/rel
            assert sha(p)==done['feature_manifest'][rel]['sha256']
            with np.load(p,allow_pickle=False) as f:
                b=f['x'];t=f['teacher'];assert b.shape==t.shape==(len(c['dense_active']),5120)
                x.append(b);y.append(np.where(np.asarray(c['dense_active'])[:,None],t-b,0.))
        x=np.concatenate(x);y=np.concatenate(y)
        arrays=fit(x,y,active,w,device=a.device,output_rank=FIT_RANK)
        assert arrays['basis'].shape==(5120,FIT_RANK) and arrays['weights'].shape==(65,FIT_RANK)
        folder=root/('layer_'+str(site));folder.mkdir()
        np.savez_compressed(folder/'TRAIN_fit_private.npz',**arrays)
        inference={k:arrays[k] for k in FIELDS};np.savez_compressed(folder/'ridge_inference.npz',**inference)
        replay=replay_fit(arrays);error=float(np.linalg.norm(replay-arrays['weights'])/max(np.linalg.norm(replay),1e-8));assert error<1e-5
        prediction=predict_numpy(x,inference,rank=INFERENCE_RANK)
        full_fit_prediction=predict_numpy(x,inference,rank=FIT_RANK)
        meta={'layer':site,'checkpoint_sha256':sha(folder/'ridge_inference.npz'),'width':5120,'input_rank':32,'fit_rank':FIT_RANK,'inference_rank':INFERENCE_RANK,'ridge_lambda':1.,'strength':1.,'seed':20261009,
              'TRAIN_unique_scenarios':24,'TRAIN_rows':4899,'TRAIN_nonzero_rows':717,'TRAIN_zero_rows':4182,'equal_case_total_weight':True,
              'label_policy':plan['target_policy'],'model_weights_trained':False,'teacher_required_at_inference':False,'fit_device':a.device,
              'replay_relative_L2':error,'TRAIN_reconstruction_relative_L2':float(np.linalg.norm(prediction-y)/max(np.linalg.norm(y),1e-8)),
              'TRAIN_fit8_reconstruction_relative_L2':float(np.linalg.norm(full_fit_prediction-y)/max(np.linalg.norm(y),1e-8)),
              'TRAIN_reconstruction_reported_deployment_rank':INFERENCE_RANK,
              'TRAIN_nonzero_reconstruction_relative_L2':float(np.linalg.norm((prediction-y)[active])/max(np.linalg.norm(y[active]),1e-8)),
              'TRAIN_zero_prediction_RMS':float(np.sqrt(np.mean(prediction[~active]**2))),'held_out_executions':0}
        save(folder/'CONTROLLER_METADATA.json',meta);results[str(site)]=meta
        save(root/'PROGRESS.json',{'utc':now(),'fitted_layers':site+1,'scheduled_layers':64,'rows_each':4899,'new_model_forwards':0,'new_JEV_calls':0})
        del x,y,arrays,inference,prediction,full_fit_prediction
        if a.device=='cuda':torch.cuda.empty_cache()
    save(root/'FIT_COMPLETE.json',{'utc':now(),'layers':results,'models_frozen':True,'new_model_forwards':0,'new_JEV_calls':0,'held_out_executions':0})

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('inputs','native','root'):p.add_argument('--'+n,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--max-seconds',type=int,required=True);a=p.parse_args()
    try:main(a)
    except Exception as e:save(pathlib.Path(a.root)/'FAULT.json',{'utc':now(),'type':type(e).__name__,'message':str(e)[:220]});raise

"""CPU-only case-balanced ridge fits after complete coverage/label verification."""
import argparse,collections,json,pathlib,sys
from datetime import datetime,timezone
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'ridge_distillation_v1'))
from ridge_math import fit,predict,replay_fit
from io_utils import save,sha
def main(a):
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    inputs,native,output=map(pathlib.Path,(a.inputs,a.native,a.output));output.mkdir(exist_ok=False)
    complete=json.loads((native/'COMPLETE.json').read_bytes());assert complete['contexts']==160 and complete['weights_unchanged']
    contexts=json.loads((inputs/'TRAIN_CONTEXTS.json').read_bytes());slots=json.loads((inputs/'CASE_BALANCED_SLOTS.json').read_bytes())
    audit=json.loads((inputs/'COVERAGE_AUDIT_BEFORE_FIT.json').read_bytes())
    assert audit['teacher_critical_context_coverage']['first_notice_response']==24
    assert audit['teacher_critical_context_coverage']['self_after_own_grant']==36
    assert audit['teacher_critical_context_coverage']['self_reserved_before_activation']==4
    assert set(collections.Counter(s['case_id'] for s in slots).values())=={audit['context_slots_per_case']}
    records={};features={}
    for key,files in complete['data_manifest'].items():
        for f in files.values():assert sha(native/'contexts'/f['file'])==f['sha256']
        n=int(key);records[n]=json.loads((native/'contexts'/files['record']['file']).read_bytes())
    labels=[]
    for slot in slots:
        c=contexts[slot['context_index']];r=records[c['context_index']]
        for sample in c['samples']:
            flag=r['labels'][str(sample['position'])]['nonzero_target']
            if not c['self_reference_label'] or not c['after_notice']:assert not flag
            labels.append(flag)
    active=np.asarray(labels,dtype=bool);assert active.sum()>=8
    result={}
    for site in (31,47,62):
        frames={}
        for n,r in records.items():
            f=complete['data_manifest'][str(n)]['features']['file']
            with np.load(native/'contexts'/f,allow_pickle=False) as data:frames[n]=(data['base_'+str(site)],data['teacher_'+str(site)])
        x=[];y=[]
        for slot in slots:
            c=contexts[slot['context_index']];r=records[c['context_index']];b,t=frames[c['context_index']]
            for sample in c['samples']:
                j=r['positions'].index(sample['position']);x.append(b[j]);y.append(t[j]-b[j] if r['labels'][str(sample['position'])]['nonzero_target'] else np.zeros_like(b[j]))
        x,y=np.asarray(x),np.asarray(y);assert x.shape==y.shape==(len(active),5120)
        arrays=fit(x,y,active,seed=20261009,input_rank=32,output_rank=8,ridge=1.,device='cpu')
        folder=output/('layer_'+str(site));folder.mkdir()
        np.savez_compressed(folder/'TRAIN_fit_private.npz',**arrays)
        inference={k:arrays[k] for k in ('x_mean','input_basis','scale','basis','weights')}
        np.savez_compressed(folder/'ridge_inference.npz',**inference)
        replay=replay_fit(arrays);relative=float(np.linalg.norm(replay-arrays['weights'])/max(np.linalg.norm(replay),1e-8));assert relative<1e-5
        with torch.inference_mode():prediction=np.concatenate([predict(torch.from_numpy(x[i:i+1]),inference,2).numpy() for i in range(len(x))])
        meta={'checkpoint_sha256':sha(folder/'ridge_inference.npz'),'teacher_required_at_inference':False,'model_weights_trained':False,
              'layer':site,'width':5120,'inference_rank':2,'input_rank':32,'fit_rank':8,'ridge_lambda':1.,'seed':20261009,
              'TRAIN_unique_scenarios':24,'TRAIN_rows':len(active),'TRAIN_nonzero_rows':int(active.sum()),'TRAIN_zero_rows':int((~active).sum()),
              'case_equal_weight':True,'fit_device':'cpu','torch_version':torch.__version__,'replay_relative_L2':relative,
              'TRAIN_reconstruction_relative_L2':float(np.linalg.norm(prediction-y)/max(np.linalg.norm(y),1e-8)),
              'source_capture_sha256':sha(native/'COMPLETE.json'),'coverage_audit_sha256':sha(inputs/'COVERAGE_AUDIT_BEFORE_FIT.json'),
              'held_out_executions':0,'reconstruction_is_workflow_success':False}
        save(folder/'CONTROLLER_METADATA.json',meta);result[str(site)]=meta
    save(output/'FIT_COMPLETE.json',{'verified_utc':datetime.now(timezone.utc).isoformat(),'layers':result,'unique_TRAIN_scenarios':24,
         'three_independent_functions':True,'new_generator_forwards':0,'new_classifier_calls':0,'held_out_executions':0})
    print(json.dumps({'fits_complete':3,'weighted_rows':len(active),'nonzero_rows':int(active.sum()),'unique_TRAIN_scenarios':24}))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('inputs','native','output'):p.add_argument('--'+n,required=True)
    main(p.parse_args())

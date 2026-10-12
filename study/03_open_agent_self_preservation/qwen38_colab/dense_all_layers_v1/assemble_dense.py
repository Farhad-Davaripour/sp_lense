"""Freeze all completed controllers into teacher-free inference inputs."""
import argparse,json,pathlib,shutil
from common import read,save,sha,now

def main(a):
    template,controllers,root=map(pathlib.Path,(a.template,a.controllers,a.root));shutil.copytree(template,root)
    complete=read(controllers/'FIT_COMPLETE.json');assert len(complete['layers'])==64
    hashes={}
    for site in range(64):
        source=controllers/('layer_'+str(site));target=root/'controllers'/source.name;target.mkdir(parents=True)
        meta=read(source/'CONTROLLER_METADATA.json');assert meta['checkpoint_sha256']==sha(source/'ridge_inference.npz')
        for n in ('ridge_inference.npz','CONTROLLER_METADATA.json'):shutil.copyfile(source/n,target/n)
        hashes[str(site)]=meta['checkpoint_sha256']
    save(root/'CONTROLLERS_FROZEN.json',{'utc':now(),'controllers':hashes,'selected_against_held_out':False,'held_out_executions':0})
    pins={p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file() and p.name!='INPUT_MANIFEST.json'}
    assert not any(any(b in n.lower() for b in ('dense_contexts','train_cases','train_fit_private','adapter_model.safetensors','.env','.ipynb')) for n in pins)
    save(root/'INPUT_MANIFEST.json',{'files':pins,'teacher_weights_excluded':True,'TRAIN_states_excluded':True,'new_JEV_calls':0})

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('template','controllers','root'):p.add_argument('--'+n,required=True)
    main(p.parse_args())

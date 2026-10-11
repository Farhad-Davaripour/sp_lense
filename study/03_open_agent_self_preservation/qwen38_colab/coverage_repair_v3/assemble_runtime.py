"""Build inference-only inputs from the pre-frozen template and completed TRAIN fit."""
import argparse,json,pathlib,shutil,hashlib
from datetime import datetime,timezone
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(a):
    template,controllers,output=map(pathlib.Path,(a.template,a.controllers,a.output))
    shutil.copytree(template,output)
    fit=json.loads((controllers/'FIT_COMPLETE.json').read_bytes())
    plan=json.loads((output/'DEVELOPMENT_TEMPLATE.json').read_bytes());plan.pop('controller_hashes_pending_TRAIN_fit',None)
    plan['format']='research2_qwen27_coverage_corrected_matched_development_v3'
    plan['frozen_after_TRAIN_before_validation_utc']=datetime.now(timezone.utc).isoformat()
    plan['controllers']={}
    for site in (31,47,62):
        source=controllers/('layer_'+str(site));target=output/'controllers'/source.name;target.mkdir(parents=True)
        meta=json.loads((source/'CONTROLLER_METADATA.json').read_bytes())
        assert meta['checkpoint_sha256']==sha(source/'ridge_inference.npz') and meta['layer']==site
        for name in ('ridge_inference.npz','CONTROLLER_METADATA.json'):shutil.copyfile(source/name,target/name)
        plan['controllers'][str(site)]=meta['checkpoint_sha256']
    (output/'DEVELOPMENT_FREEZE.json').write_text(json.dumps(plan,indent=2)+'\n')
    pins={p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file() and p.name!='INPUT_MANIFEST.json'}
    assert not any(any(bad in n.lower() for bad in ('.env','.ipynb','train_contexts','train_fit','train_cases','context_000.npz','adapter_model.safetensors')) for n in pins)
    (output/'INPUT_MANIFEST.json').write_text(json.dumps({'files':pins,'teacher_weights_excluded':True,'TRAIN_states_excluded':True,'only_two_scientific_methods':True},indent=2)+'\n')
    print(json.dumps({'runtime_inputs_ready':True,'scientific_methods':2,'unique_validation_scenarios':9,'planned_runs':36,'new_JEV_calls':0}))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('template','controllers','output'):p.add_argument('--'+n,required=True)
    main(p.parse_args())

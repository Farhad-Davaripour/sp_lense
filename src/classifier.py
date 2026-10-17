"""Persist classifier data or run the bounded Databricks fine-tuning job."""
from pathlib import Path
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from src.config import ROOT, Settings
from src.utils import canonical, sha256, write_json, workspace_client
from src.persist_data import persist
from src.classifier_helpers import configuration, prepare_datasets, read_dataset

def stage_base(settings, config, setup_end):
    spec=config['model']
    destination=Path(settings.volume_root)/spec['base_volume_directory']/spec['revision']
    destination.mkdir(parents=True,exist_ok=True)
    missing=any(not (destination/name).is_file() for name in spec['files'])
    if missing:
        env=os.environ.copy()
        for key in ('HF_TOKEN','OPENAI_API_KEY','ANTHROPIC_API_KEY','DATABRICKS_TOKEN','databricks-token'):
            env.pop(key,None)
        code=('from huggingface_hub import snapshot_download;import sys,json;'
              'snapshot_download(repo_id=sys.argv[1],revision=sys.argv[2],local_dir=sys.argv[3],'
              'allow_patterns=json.loads(sys.argv[4]),max_workers=2)')
        subprocess.run([sys.executable,'-c',code,spec['repository'],spec['revision'],str(destination),
                        json.dumps(list(spec['files']))],env=env,check=True,
                       timeout=max(1,setup_end-time.time()))
    for name,pin in spec['files'].items():
        p=destination/name
        if p.stat().st_size!=pin['bytes'] or sha256(p)!=pin['sha256']:
            raise ValueError('Public classifier base file mismatch')
    return destination

def dispatch(admission_path):
    admission_path=Path(admission_path);admission=json.loads(admission_path.read_bytes())
    settings=Settings.load();config=configuration()
    start=dt.datetime.fromisoformat(admission['submission_utc'])
    end=start.timestamp()+config['bounds']['whole_job_seconds']
    setup_end=start.timestamp()+config['bounds']['setup_seconds']
    if start.tzinfo is None or not start.timestamp()<=time.time()<setup_end:
        raise TimeoutError('The original classifier setup window expired')
    for relative,pin in admission['source_pins'].items():
        p=(ROOT/relative).resolve()
        if not p.is_relative_to(ROOT) or sha256(p)!=pin:
            raise ValueError('Classifier source changed after submission')
    root=Path(admission['output']).resolve()
    if not root.is_relative_to(Path(settings.volume_root).resolve()) or root.exists():
        raise ValueError('Use a fresh run directory in the configured Volume')
    root.mkdir(parents=True)
    write_json(root/'SUBMISSION.json',admission)
    client=workspace_client(settings)
    inputs={}
    # No heldout query or file release occurs before a frozen development-qualified model.
    for dataset in ('C1-TRAIN','C1-DEV','C1-JEV-DEV'):
        rows=read_dataset(client,settings,config,dataset)
        target=root/'inputs'/(dataset+'.json');write_json(target,rows)
        inputs[dataset]={'path':str(target),'sha256':sha256(target),'records':len(rows)}
    base=stage_base(settings,config,setup_end)
    write_json(root/'BASE_FILES_VERIFIED.json',{'base':str(base),'revision':config['model']['revision'],
                                             'files':config['model']['files']})
    worker=dict(admission,root=str(root),base=str(base),inputs=inputs,
                absolute_end_utc=dt.datetime.fromtimestamp(end,dt.timezone.utc).isoformat(),
                configuration_sha256=sha256(ROOT/'config/classifier.yaml'))
    worker_path=root/'WORKER_CONFIG.json';write_json(worker_path,worker)
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT);env['HF_HUB_OFFLINE']='1';env['USE_TF']='0'
    for key in ('HF_TOKEN','OPENAI_API_KEY','ANTHROPIC_API_KEY','DATABRICKS_TOKEN','databricks-token'):
        env.pop(key,None)
    released=False;child=None
    try:
        with (root/'WORKER_LOG.txt').open('w',encoding='utf-8') as log:
            child=subprocess.Popen([sys.executable,'-m','src.classifier_training','--config',str(worker_path)],
                                   cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            while child.poll() is None:
                if time.time()>=end-300:
                    (root/'STOP').touch()
                    try:child.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        child.terminate()
                        try:child.wait(timeout=30)
                        except subprocess.TimeoutExpired:child.kill();child.wait(timeout=30)
                    raise TimeoutError('Classifier reached its original job boundary')
                if (root/'REQUEST_HELDOUT.json').exists() and not released:
                    frozen=json.loads((root/'FROZEN_MODEL.json').read_bytes())
                    for split,count in (('development',32),('familiar_JEV_development',24)):
                        if frozen[split]['signature_matches']!=count or frozen[split]['count']!=count:
                            raise ValueError('Heldout release attempted before exact development agreement')
                    for relative,pin in frozen['adapter_files'].items():
                        if sha256(root/'selected_adapter'/relative)!=pin['sha256']:
                            raise ValueError('Frozen adapter changed before heldout release')
                    test=read_dataset(client,settings,config,'C1-TEST')
                    target=root/'inputs/C1-TEST.json';write_json(target,test)
                    write_json(root/'HELDOUT_RELEASE.json',{'frozen_model_sha256':sha256(root/'FROZEN_MODEL.json'),
                               'input_path':str(target),'input_sha256':sha256(target),'records':len(test),
                               'label_origin':'authored synthetic, not JEV','once_only':True})
                    released=True
                time.sleep(2)
        if child.returncode:
            raise RuntimeError('Classifier worker failed; inspect its preserved private log and fault receipt')
        result=json.loads((root/'RESULT.json').read_bytes())
        write_json(root/'PROCESS_EXIT.json',{'return_code':0,'heldout_released':released,
                                           'semantic_status':result['status'],'no_automatic_retry':True})
        return result
    except BaseException as error:
        (root/'STOP').touch()
        if child is not None and child.poll() is None:
            try:child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=30)
        write_json(root/'PARENT_FAULT.json',{'type':type(error).__name__,
                   'heldout_released':released,'automatic_retry':False})
        raise

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--persist-data',action='store_true')
    parser.add_argument('--admission')
    args=parser.parse_args(argv)
    settings=Settings.load();config=configuration()
    if args.persist_data:
        prepared=prepare_datasets(settings,config)
        result=persist(settings,prepared) if args.execute else {
            'mode':'classifier_local_validation','datasets':{d['spec']['id']:len(d['rows']) for d in prepared},
            'model_calls':0,'cloud_mutations':0}
    elif args.execute:
        if not args.admission:parser.error('--execute requires --admission')
        result=dispatch(args.admission)
    else:
        result={'model':config['model']['repository'],'revision':config['model']['revision'],
                'planned_training':config['training'],'datasets':{d['id']:d['expected_records'] for d in config['datasets']},
                'new_JEV_calls':0,'model_calls':0}
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    main()

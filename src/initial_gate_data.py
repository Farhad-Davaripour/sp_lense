"""Frozen native four-choice initial-gate references and Bronze persistence."""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import subprocess
import uuid
import zipfile
import yaml
from src.config import ROOT, Settings
from src.utils import canonical, sha256, write_json, workspace_client
from src.persist_data import persist
from src.reference_inputs import cached_provenance
from src.engines.workflow.initial_gate import payload, digest, freeze, MODEL


def require_closed_development_failure(previous):
    flags=('workflow_benchmark_executed','initial_H24_executed','binary_confirmation_executed')
    joint=(previous.get('status')=='JOINT_CLASSIFIER_DEVELOPMENT_FAILED'
           and all(previous.get(k) is False for k in flags))
    native=(previous.get('status')=='INITIAL_GATE_DEVELOPMENT_FAILED' and previous.get('benchmark_executed') is False)
    if not (joint or native):raise ValueError('No refinement after opening a historical confirmation benchmark')


def prepare_sources(f24, workflow_references, output):
    f24, references, output=map(Path,(f24,workflow_references,output))
    if output.exists():raise FileExistsError('Use a new initial-gate reference directory')
    parent=yaml.safe_load((ROOT/'config/workflow_classifier_transfer.yaml').read_bytes())
    if sha256(references/'MANIFEST.json')!=parent['dataset']['manifest_sha256']:
        raise ValueError('Original workflow references changed')
    manifest=json.loads((references/'MANIFEST.json').read_bytes())
    for name,pin in manifest['files'].items():
        if (references/name).stat().st_size!=pin['bytes'] or sha256(references/name)!=pin['sha256']:
            raise ValueError('Original workflow reference member changed')
    train=[]
    for row in json.loads(f24.read_bytes()):
        provenance=cached_provenance(row)
        train.append({'id':row['id'],'state':row['state'],'teacher_gate':row['teacher_frozen_gate'],
                      'reference_provenance':provenance,'family':row['family'],'condition':row['condition']})
    dev=json.loads((references/'initial_development.json').read_bytes())
    benchmark=json.loads((references/'initial_familiar_H24.json').read_bytes())
    if (len(train),len(dev),len(benchmark))!=(24,9,24):raise ValueError('Native initial-gate panel membership changed')
    for row in train+dev+benchmark:
        request=payload(row['state']['visible_history'],row['state']['visible_tools'],model=row['teacher_gate']['provider'])
        original=row['teacher_gate']
        g=freeze(request,original['provider'],original['label'],original['confidence'],original['self_score'])
        if g.__dict__!=original:raise ValueError('Native original request or frozen teacher gate changed')
    state_set=lambda rows:{canonical(row['state']) for row in rows}
    if state_set(train)&state_set(dev+benchmark):
        raise ValueError('Initial-gate training overlaps a validation or benchmark input')
    output.mkdir(parents=True)
    def emit(path,value):
        path.write_bytes((json.dumps(value,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8'))
    for name,rows in [('training',train),('development',dev),('benchmark',benchmark)]:emit(output/(name+'.json'),rows)
    first=train[0]['state']
    emit(output/'questions.json',payload(first['visible_history'],first['visible_tools'])['questions'])
    record={'format':'research3-native-initial-gate-data-v1','source_builder_sha256':sha256(Path(__file__)),
        'F24_source_sha256':sha256(f24),'parent_workflow_manifest_sha256':parent['dataset']['manifest_sha256'],
        'counts':{'training':24,'development':9,'historical_benchmark':24},'new_reference_calls':0,'model_calls':0,
        'training_exact_visible_overlap_with_validation_or_benchmark':0,
        'development_benchmark_exact_visible_overlap':len(state_set(dev)&state_set(benchmark)),
        'historical_H24_is_previously_inspected':True,'R24_sealed':True,
        'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha256(p)} for p in sorted(output.glob('*.json'))}}
    emit(output/'MANIFEST.json',record)
    return record


def archive_sources(directory):
    """Fixed metadata and stored bytes make regeneration independent of file mtimes/zlib."""
    directory=Path(directory);archive=directory.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_STORED) as z:
        for p in sorted(directory.glob('*.json')):
            info=zipfile.ZipInfo(p.name,date_time=(1980,1,1,0,0,0));info.create_system=3
            info.external_attr=0o100644<<16;info.compress_type=zipfile.ZIP_STORED
            z.writestr(info,p.read_bytes())
    return archive


def configuration():return yaml.safe_load((ROOT/'config/initial_gate_classifier.yaml').read_bytes())


def validated_members(packet):
    config=configuration();packet=Path(packet)
    if packet.stat().st_size!=config['data']['archive_bytes'] or sha256(packet)!=config['data']['archive_sha256']:
        raise ValueError('Native initial-gate reference archive changed')
    with zipfile.ZipFile(packet) as z:
        allowed={'training.json','development.json','benchmark.json','questions.json','MANIFEST.json'}
        if set(z.namelist())!=allowed or len(z.namelist())!=len(allowed):
            raise ValueError('Unexpected native initial-gate reference members')
        raw=z.read('MANIFEST.json')
        if hashlib.sha256(raw).hexdigest()!=config['data']['manifest_sha256']:
            raise ValueError('Native initial-gate reference manifest changed')
        manifest=json.loads(raw);members={}
        if set(manifest['files'])!=allowed-{'MANIFEST.json'}:raise ValueError('Incomplete native member manifest')
        for name,pin in manifest['files'].items():
            b=z.read(name)
            if len(b)!=pin['bytes'] or hashlib.sha256(b).hexdigest()!=pin['sha256']:
                raise ValueError('Native initial-gate reference member changed')
            members[name]=b
    return members


def admitted_predecessor(a, client=None):
    if a.get('run_kind')=='fresh_recipe_reproduction' and not a.get('development_only_predecessor'):
        return
    p=Path(a['development_only_predecessor'])
    if sha256(p)!=a['development_only_predecessor_sha256']:raise ValueError('Closed predecessor receipt changed')
    require_closed_development_failure(json.loads(p.read_bytes()))
    if client:
        run=client.jobs.get_run(a['predecessor_run_id'])
        if run.state.life_cycle_state.value!='TERMINATED' or run.state.result_state.value!='SUCCESS':
            raise ValueError('The prior scientific run and its comparison must close cleanly first')


def check_admission_window(a, remaining_setup_seconds=0):
    values=[dt.datetime.fromisoformat(a[k]) for k in ('submission_utc','model_end_utc','whole_job_end_utc')]
    if any(v.tzinfo is None for v in values):raise ValueError('Admission times require an explicit UTC offset')
    start,end,whole=values;now=dt.datetime.now(dt.timezone.utc)
    if (whole-start).total_seconds()>2700 or (end-start).total_seconds()>2400 or end>whole-dt.timedelta(seconds=300):
        raise ValueError('The fixed native initial-gate execution bounds were exceeded')
    if a.get('parent_phase_end_utc'):
        parent=dt.datetime.fromisoformat(a['parent_phase_end_utc'])
        if whole>parent or end>parent-dt.timedelta(seconds=300):raise ValueError('Parent phase cap exceeded')
    if not start<=now<start+dt.timedelta(seconds=600-remaining_setup_seconds) or (end-now).total_seconds()<600:
        raise TimeoutError('Original native initial-gate admission window expired')
    return start.timestamp(),end.timestamp()


def persist_references(admission_path,execute=False,spark=None):
    a=json.loads(Path(admission_path).read_bytes());settings=Settings.load();config=configuration()
    if execute:
        check_admission_window(a)
        if sha256(ROOT/'config/initial_gate_classifier.yaml')!=a['protocol_sha256']:
            raise ValueError('The admitted native initial-gate protocol changed')
    admitted_predecessor(a)
    members=validated_members(a['packet_zip']);prepared=[]
    for spec in config['data']['datasets']:
        raw=members[spec['source_member']];records=json.loads(raw)
        if hashlib.sha256(raw).hexdigest()!=spec['sha256'] or len(records)!=spec['expected_records']:
            raise ValueError('Native initial-gate dataset changed')
        prefix=settings.volume_root+'/datasets/'+spec['id']+'/'+spec['version']+'/'+spec['sha256']
        rows=[]
        for i,row in enumerate(records):
            text=canonical(row)
            rows.append({'dataset_id':spec['id'],'dataset_version':spec['version'],'cohort_code':spec['id'],
                'source_record_index':i,'record_id':row['id'],'source_file_sha256':spec['sha256'],
                'record_sha256':hashlib.sha256(text.encode()).hexdigest(),'record_json':text,
                'usage_roles_json':canonical(spec['usage_roles']),'source_volume_path':prefix+'/source.json'})
        prepared.append({'spec':spec,'raw':raw,'rows':rows,'source_volume':prefix+'/source.json',
                         'records_volume':prefix+'/records.jsonl'})
    if execute:
        client=workspace_client(settings);admitted_predecessor(a,client)
        result=persist(settings,prepared,client,spark=spark)
        check_admission_window(a,remaining_setup_seconds=180)
        return result
    return {'mode':'data_checks_only','counts':[len(p['rows']) for p in prepared],'model_calls':0,'cloud_mutations':0}


def submit(job_id, predecessor=None, predecessor_run_id=None, parent_end=None):
    """Explicit one-shot submission; private inputs and the deployed job must already exist."""
    from src.persist_data import upload_verified
    from src.verify import source_digest
    dirty=subprocess.run(['git','status','--porcelain','--','src','config','ddl','notebooks','resources'],
        cwd=ROOT,capture_output=True,text=True,check=True)
    if dirty.stdout.strip():raise ValueError('Commit the scientific source, configuration and job definitions before submission')
    settings=Settings.load();client=workspace_client(settings);config=configuration()
    job=client.jobs.get(job_id)
    expected={'persist_native_initial_references','fit_native_initial_gate','compare_with_cached_reference'}
    if {t.task_key for t in job.settings.tasks or []}!=expected:
        raise ValueError('Supply the deployed native initial-gate job ID')
    if predecessor:
        if not predecessor_run_id:raise ValueError('The prior provider run ID is required')
        with client.files.download(predecessor).contents as stream:raw=stream.read()
        require_closed_development_failure(json.loads(raw))
        run=client.jobs.get_run(predecessor_run_id)
        if run.state.life_cycle_state.value!='TERMINATED' or run.state.result_state.value!='SUCCESS':
            raise ValueError('Finish the preceding scientific run and comparison first')
    now=dt.datetime.now(dt.timezone.utc);whole_end=now+dt.timedelta(seconds=2700)
    if parent_end:whole_end=min(whole_end,dt.datetime.fromisoformat(parent_end))
    model_end=whole_end-dt.timedelta(seconds=300)
    if (model_end-now).total_seconds()<1800:raise TimeoutError('Insufficient unchanged parent window')
    stamp=now.strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    try:commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.SubprocessError):commit=None
    admission={'submission_utc':now.isoformat(),'model_end_utc':model_end.isoformat(),'whole_job_end_utc':whole_end.isoformat(),
        'run_kind':'development_continuation' if predecessor else 'fresh_recipe_reproduction',
        'packet_zip':settings.volume_root+'/'+config['data']['volume_directory']+'/'+config['data']['archive_file'],
        'output':settings.volume_root+'/runs/native_initial_gate_classifier/'+stamp,
        'protocol_sha256':sha256(ROOT/'config/initial_gate_classifier.yaml'),'source_commit':commit,'source_digest':source_digest(),
        'Free_Edition_only':True,'automatic_retries':0,'new_reference_calls':0,'historical_H24_is_a_known_benchmark':True}
    if parent_end:admission['parent_phase_end_utc']=parent_end
    if predecessor:
        admission.update(development_only_predecessor=predecessor,development_only_predecessor_sha256=hashlib.sha256(raw).hexdigest(),
                         predecessor_run_id=predecessor_run_id)
    local=settings.output_root/'initial_gate_submissions'/(stamp+'.json')
    write_json(local,admission)
    cloud=settings.volume_root+'/runs/native_initial_gate_submissions/'+stamp+'/ADMISSION.json'
    upload_verified(client,cloud,local.read_bytes())
    token=uuid.uuid4().hex
    intent={'job_id':job_id,'admission':cloud,'idempotency_token':token,'submission_utc':now.isoformat()}
    write_json(local.with_suffix('.intent.json'),intent)
    parameters={'catalog':settings.catalog,'schema':settings.schema,'volume':settings.volume,'admission':cloud}
    if settings.warehouse_id:parameters['warehouse_id']=settings.warehouse_id
    run=client.jobs.run_now(job_id=job_id,job_parameters=parameters,idempotency_token=token)
    receipt=dict(intent,run_id=run.run_id,output=admission['output'],model_end_utc=model_end.isoformat(),whole_job_end_utc=whole_end.isoformat(),
                 source_commit=commit,source_digest=admission['source_digest'])
    write_json(local.with_suffix('.submitted.json'),receipt)
    return receipt


def main(argv=None,spark=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admission');parser.add_argument('--execute',action='store_true')
    parser.add_argument('--f24-rows');parser.add_argument('--workflow-references');parser.add_argument('--output')
    parser.add_argument('--submit',action='store_true',help='Explicitly submit once with a new 45-minute bound and immutable receipt')
    parser.add_argument('--job-id',type=int,help='Native initial-gate job ID from bundle summary')
    parser.add_argument('--predecessor',help='Closed joint RESULT.json Volume path; only for this phase continuation')
    parser.add_argument('--predecessor-run-id',type=int,help='Provider run ID for that completed joint attempt')
    parser.add_argument('--parent-end-utc',help='Unchanged parent phase end as ISO8601 with UTC offset')
    args=parser.parse_args(argv)
    if args.submit:
        if not args.job_id or args.execute or args.admission:parser.error('--submit requires --job-id; it is a separate explicit action')
        result=submit(args.job_id,args.predecessor,args.predecessor_run_id,args.parent_end_utc)
    elif args.admission:result=persist_references(args.admission,args.execute,spark=spark)
    else:
        if args.execute or not all((args.f24_rows,args.workflow_references,args.output)):
            parser.error('Preparation requires --f24-rows, --workflow-references and a new --output')
        result=prepare_sources(args.f24_rows,args.workflow_references,args.output)
        archive=archive_sources(args.output)
        result=dict(result,archive_bytes=archive.stat().st_size,archive_sha256=sha256(archive))
    print(json.dumps(result));return result


if __name__=='__main__':main()

"""Databricks data-only generation/export/audit entrypoint; no submit API."""
import argparse,hashlib,json,tempfile,time,traceback
from pathlib import Path
from .io_utils import read,sha,dump
from .composite_v2_data_contract import MODE,BOUNDS,roster,dataset_admission,source_fingerprint
from .composite_generalized_data import authorized_data_plan,materialize
from .composite_generalized_audit import audit_cohort,native_budget_audit,leakage_audit
from .composite_exports import make_evidence_archive


class Writer:
    def __init__(self,root,guard):self.root=Path(root);self.guard=guard;self.bytes=0
    def json(self,name,value,terminal=False):
        from .composite_runtime import contained
        if not terminal:self.guard()
        data=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
        cap=BOUNDS['maximum_raw_export_bytes']-(0 if terminal else BOUNDS['terminal_metadata_reserve_bytes'])
        if self.bytes+len(data)>cap:raise OSError('Complete data export exceeds declared bound; no truncation')
        target=contained(self.root,name);target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:stream.write(data)
        if target.read_bytes()!=data:raise OSError('Data export readback differs')
        self.bytes+=len(data);return {'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}


def metadata_tokenizers(root,cache,writer):
    """Fetch only original small tokenizer/config files, never model weights."""
    import base64
    from .composite_store import download_model
    from .composite_frozen_probe import verify_metadata_bytes
    from transformers import AutoTokenizer
    result={}
    for name,pinfile in [('qwen','QWEN_PIN.json'),('bert','MODERNBERT_PIN.json')]:
        pin=read(Path(root)/'inputs'/pinfile)
        path,proof=download_model(pin,Path(cache)/name,metadata_only=True)
        encoded={item['name']:base64.b64encode((path/item['name']).read_bytes()).decode()
                 for item in pin['files'] if not item['name'].endswith('.safetensors')}
        if name=='qwen':verify_metadata_bytes(pin,proof,encoded)
        writer.json('metadata/'+name+'_FILES.json',proof)
        writer.json('metadata/'+name+'_BYTES.json',encoded)
        result[name]=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
    return result['qwen'],result['bert']


class Services:
    clock=staticmethod(time.monotonic)
    authorize=staticmethod(lambda root,commit:authorized_data_plan(root,commit,'training'))
    generate=staticmethod(materialize)
    audit=staticmethod(audit_cohort)
    native=staticmethod(native_budget_audit)
    leakage=staticmethod(leakage_audit)
    roster=staticmethod(roster)
    tokenizers=staticmethod(metadata_tokenizers)
    @staticmethod
    def alarm(seconds):
        from .composite_frozen_worker import protected_alarm
        return protected_alarm(seconds)


def run(root,commit,_services=None):
    services=Services() if _services is None else _services;started=services.clock()
    context=services.authorize(root,commit);destination=Path(context['destination'])
    destination.mkdir(parents=True,exist_ok=False)
    def guard():
        if services.clock()-started>=BOUNDS['worker_seconds']-BOUNDS['export_reserve_seconds']:
            raise TimeoutError('Data-only protected export deadline')
    writer=Writer(destination,guard)
    receipt={'stage':MODE,'source_commit':commit,'status':'running','workspace':context['execution_workspace'],
        'model_forwards':0,'optimizer_updates':0,'model_weights_downloaded':False,'GPU_required':False,
        'heldout_content_read':False,'scientific_dataset_admitted':False,'public_fixture_test_mode':_services is not None,
        'automatic_scientific_retries':0,'bounds':BOUNDS,'completed_roles':[]}
    try:
        writer.json('PRECALL_FREEZE.json',{'source_commit':commit,'authorization_sha256':context['authorization_sha256'],
            'input_manifest_sha256':sha(Path(root)/'inputs/MANIFEST.json'),'plan':context['plan'],
            'workspace':context['execution_workspace'],'source_fingerprint':source_fingerprint(root)})
        with services.alarm(BOUNDS['worker_seconds']-BOUNDS['export_reserve_seconds']):
            with tempfile.TemporaryDirectory(prefix='research4-v2-tokenizer-metadata-') as cache:
                qt,bt=services.tokenizers(root,cache,writer)
                audits={};budgets={};datasets={};manifests={}
                for role in ['training','development']:
                    guard();folder=destination/role
                    def persist(name,value,role=role):
                        return writer.json(role+'/'+name,value,terminal=name=='GENERATION_FAILURE.json')
                    manifest=services.generate(root,commit,role,context['plan']['identity_material'],folder,persist=persist)
                    episodes=read(folder/'EPISODES.json');rows=read(folder/'DETECTION_ROWS.json')
                    audits[role]=services.audit(episodes,rows,manifest,context['configuration'],reference_replay=True)
                    budgets[role]=services.native(rows,qt,bt)
                    writer.json(role+'/ROSTER.json',services.roster(episodes,role,context['configuration']))
                    writer.json(role+'/REFERENCE_AUDIT.json',audits[role]);writer.json(role+'/NATIVE_BUDGET.json',budgets[role])
                    datasets[role]=rows;manifests[role]=manifest;receipt['completed_roles'].append(role)
                leakage=services.leakage(audits['training'],audits['development'],datasets['training'],datasets['development'])
                writer.json('LEAKAGE_AUDIT.json',leakage)
                admission=dataset_admission(audits,budgets,leakage)
                if _services is not None:admission.update(training_and_development_admitted=False,public_fixture_test_mode=True)
                writer.json('DATA_ADMISSION.json',admission)
                release={'version':MODE,'source_commit':commit,'world_source_files':source_fingerprint(root),
                    'amendment_sha256':sha(Path(root)/'config/composite_generalization_amendment_v2.json'),
                    'workspace':context['execution_workspace'],'cohorts':manifests,
                    'data_admission':admission,'heldout_content_read':False,'model_forwards':0,'optimizer_updates':0}
                writer.json('DATA_RELEASE.json',release)
                receipt.update(status='complete',scientific_dataset_admitted=admission['training_and_development_admitted'])
    except Exception as error:
        receipt.update(status='failed',error_type=type(error).__name__,error=str(error))
        writer.json('FAILURE.json',{'traceback':traceback.format_exc()},terminal=True)
    finally:
        receipt.update(seconds=services.clock()-started,worker_exiting=True,allocation_release='external job owner must reconcile actual run exit')
        writer.json('RUN_RECEIPT.json',receipt,terminal=True)
    archive=destination.with_name(destination.name+'_EVIDENCE.zip')
    proof=make_evidence_archive(destination,archive,BOUNDS['maximum_raw_export_bytes'])
    return {**receipt,**proof,'archive':str(archive)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--commit',required=True);args=p.parse_args()
    value=run(args.root,args.commit);print(json.dumps(value))
    if value['status']!='complete':raise SystemExit(1)

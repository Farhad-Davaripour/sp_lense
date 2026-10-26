"""One owned CPU approved-sampler replay/export; no new teacher, case or model."""
import argparse,json,os,re,time,traceback
from pathlib import Path
from .io_utils import read,sha,digest
from .composite_bounded_sampling_profile import (MODE,PHASE,NOTEBOOK,OWNER_THREAD,PROFILE,BOUNDS,
 APPROVAL_SHA,PROPOSAL_SHA)
from .composite_bounded_sampling_source import validate_source
from .composite_bounded_sampling_rule import RULE_SHA
from .composite_bounded_behavior_data_worker import verify_cpu_environment
from .composite_exports import make_evidence_archive
from .composite_v2_data_worker import Writer


class Services:
    clock=staticmethod(time.monotonic)
    @staticmethod
    def preview(root,commit):
        if not re.fullmatch('[0-9a-f]{40}',commit or ''):raise PermissionError('Exact reviewed approved sampler source required')
        value=read(Path(root)/'inputs'/PROFILE)
        destination='/kaggle/working/research4_campaign/runs/'+commit+'/bounded_sampling_amendment_v1/export_01'
        expected={'source_commit':commit,'version':MODE,'phase':PHASE,'notebook':NOTEBOOK,
          'owner_thread_id':OWNER_THREAD,'destination':destination}
        if any(digest(value.get(k))!=digest(v) for k,v in expected.items()):raise PermissionError('Exact owned sampler source/phase/output required')
        return value
    @staticmethod
    def runtime():
        from .composite_bounded_behavior_data_worker import Services as CPU
        return CPU.runtime()
    @staticmethod
    def authorize(root,commit):
        from .composite_bounded_sampling_parent import read_failed_parent
        paths=[]
        for key,suffix in [('RESEARCH4_BOUNDED_ALIAS_FIT_ARCHIVE','82cf3ecf41e799facd93b8c11e05c6d1f569599c/task_skill_training_v1/full_fit_01/evidence.zip'),
          ('RESEARCH4_BOUNDED_ALIAS_QUALIFICATION_ARCHIVE','10eae62507d38f1507fb7443cf579ca5b6acc29f/task_skill_training_v1/qualification_01/evidence.zip')]:
            path=Path(os.environ[key]).resolve()
            if not path.is_relative_to(Path('/kaggle/input').resolve()) or not path.as_posix().endswith(suffix):
                raise PermissionError('Exact immutable original alias/model raw dependencies required')
            paths.append(path)
        parent=read_failed_parent(root,None,*paths)
        context=validate_source(root,commit,parent);context.require_runtime();return context
    @staticmethod
    def materialize(context,persist):
        from .composite_bounded_sampling_data import materialize
        return materialize(context,persist=persist)
    @staticmethod
    def alarm(seconds):
        from .composite_frozen_worker import protected_alarm
        return protected_alarm(seconds)


def run(root,commit,_services=None):
    services=Services() if _services is None else _services
    started=services.clock();profile=services.preview(root,commit)
    output=Path(profile['destination']);output.mkdir(parents=True,exist_ok=False)
    def guard():
        if services.clock()-started>=BOUNDS['worker_seconds']-BOUNDS['export_reserve_seconds']:
            raise TimeoutError('Approved sampling CPU protected export boundary')
    writer=Writer(output,guard,maximum_bytes=BOUNDS['maximum_raw_export_bytes'],terminal_reserve=BOUNDS['terminal_metadata_reserve_bytes'])
    receipt={'stage':MODE,'mode':MODE,'phase':PHASE,'source_commit':commit,'owner_thread_id':OWNER_THREAD,
      'status':'running','model_forwards':0,'optimizer_updates':0,'scientific_GPU_count':0,'GPU_required':False,
      'model_weights_downloaded':False,'heldout_content_read':False,'public_fixture_test_mode':_services is not None,
      'automatic_scientific_retries':0,'completed_roles':[],'completed_cases':0,'new_teacher_or_case_generation':False,
      'scientific_dataset_admitted':False,'training_launch_admitted':False,'original_failure_preserved':True,
      'metadata_cache_created':False,'metadata_cache_removed':True,'approval_sha256':APPROVAL_SHA,'proposal_sha256':PROPOSAL_SHA}
    try:
        writer.json('OUTPUT_OWNER.json',{'source_commit':commit,'version':MODE,'phase':PHASE,
          'owner_thread_id':OWNER_THREAD,'path':profile['destination']})
        with services.alarm(BOUNDS['worker_seconds']-BOUNDS['export_reserve_seconds']):
            receipt['cpu_runtime']=verify_cpu_environment(services.runtime())
            context=services.authorize(root,commit);guard()
            writer.json('PRECALL_FREEZE.json',{'source_commit':commit,'profile':context.profile(),
              'study_configuration':context.cfg(),'input_manifest_sha256':sha(Path(root)/'inputs/MANIFEST.json'),
              'source_release_sha256':sha(Path(root)/'RELEASE.json'),'failed_reference_receipt_sha256':digest(context.parent.receipt()),
              'approval_sha256':APPROVAL_SHA,'approved_rule_sha256':RULE_SHA})
            result=services.materialize(context,writer.json)
            if result['completed_cases']!=144:raise ValueError('All108training and36development recorded references required')
            receipt.update(status='complete',completed_cases=144,completed_roles=['training','development'])
    except Exception as error:
        receipt.update(status='failed',error_type=type(error).__name__,error=str(error))
        writer.json('FAILURE.json',{'traceback':traceback.format_exc()},terminal=True)
    finally:
        receipt.update(seconds=services.clock()-started,counts_exact_at_controlled_exit=True,
          worker_exiting=True,provider_compute_release_verified=False,raw_data_audit_and_provider_release_required=True)
        writer.json('RUN_RECEIPT.json',receipt,terminal=True)
    archive=output.with_name(output.name+'_EVIDENCE.zip')
    return {**receipt,**make_evidence_archive(output,archive,BOUNDS['maximum_raw_export_bytes']),'archive':str(archive)}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--commit',required=True)
    result=run(**vars(parser.parse_args()));print(json.dumps(result))
    if result['status']!='complete':raise SystemExit(1)

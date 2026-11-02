"""Parent-side child ownership, rescue exports and post-exit cache cleanup."""
import json
import os
from pathlib import Path
import shutil
import time
import uuid
from .io_utils import read, sha
from .composite_exports import make_evidence_archive, verify_evidence_archive
from .composite_probe_ownership import create_owned_cache, cleanup_owned_cache
from .composite_frozen_probe import authorized_load_plan, BOUNDS
from .composite_frozen_probe import MODE
from .composite_frozen_worker import output_target, DISK_POLICY, release_export_space, transport_environment, atomic_json
from .composite_runtime import run_logged

CONTROLLER_BASE='/kaggle/working/research4_campaign/controllers'


def output_owned(target,source_commit,token):
    try:return read(Path(target)/'OUTPUT_OWNER.json')=={'source_commit':source_commit,'token':token,'path':str(Path(target).resolve())}
    except (OSError,ValueError):return False


def observed_exit_counts(target, source_commit, child_state, owner_token=None, mode=MODE, maximum_forwards=400):
    raw=Path(target)/'raw'; journal=None
    try: journal=read(raw/'FORWARD_COUNTS.json')
    except (OSError,ValueError): pass
    if child_state.get('child_started') is False:
        return {'counts_exact_at_exit':True,'forward_counts':{'attempted':0,'completed':0},
                'source':'owned_process_creation_failed','last_durable_counts':journal}
    if owner_token is not None and not output_owned(target,source_commit,owner_token):
        return {'counts_exact_at_exit':False,'forward_counts':{'attempted':None,'completed':None},
                'source':'output_owner_unconfirmed','last_durable_counts':None}
    try:
        receipt=read(raw/'RUN_RECEIPT.json')
        counts=receipt['forward_counts']
        valid=(receipt['source_commit']==source_commit and receipt['stage']==mode and receipt['worker_exiting'] is True
               and receipt['counts_exact_at_controlled_exit'] is True
               and journal.get('sealed_for_exit') is True
               and all(type(counts[key]) is int and counts[key]==journal[key] for key in ['attempted','completed'])
               and 0<=counts['completed']<=counts['attempted']<=maximum_forwards)
        if child_state.get('child_exit_confirmed') is True and valid:
            return {'counts_exact_at_exit':True,'forward_counts':counts,
                    'source':'sealed_controlled_worker_exit','last_durable_counts':journal}
    except (OSError,ValueError,KeyError,AttributeError): pass
    return {'counts_exact_at_exit':False,'forward_counts':{'attempted':None,'completed':None},
            'source':'abrupt_or_unconfirmed_exit','last_durable_counts':journal}


def preserve_after_exit(target, source_commit, child_state, observation, mode=MODE, archive_name='FROZEN_EVIDENCE.zip'):
    target=Path(target)
    if child_state.get('child_exit_confirmed') is not True:
        raise PermissionError('Cannot snapshot or rewrite artifacts while the child may be live')
    export=target/'EXPORT_RECEIPT.json'; archive=target/archive_name
    if export.exists() and archive.exists():
        try:
            value=read(export)
            if value['source_commit']!=source_commit or value['stage']!=mode or sha(archive)!=value['archive_sha256']:
                raise ValueError('Worker export source/SHA differs')
            verify_evidence_archive(archive.read_bytes())
            return {'evidence_kind':'original_worker_export','archive':str(archive),
                    'archive_sha256':value['archive_sha256'],'worker_status':value['status']}
        except (ValueError,KeyError,OSError): pass
    rescue=target/'controller_rescue'; rescue.mkdir(exist_ok=False)
    raw=rescue/'raw'; raw.mkdir()
    original=target/'raw'
    if original.exists():
        for file in original.rglob('*'):
            if not file.is_file(): continue
            if not file.resolve().is_relative_to(original.resolve()):
                raise ValueError('A rescue member escaped the original owned output')
            relative=file.relative_to(original)
            if relative.as_posix()=='MANIFEST.json': relative=Path('WORKER_MANIFEST_AT_EXIT.json')
            dest=raw/relative; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(file,dest)
    if len((json.dumps(observation,indent=2)+'\n').encode())>DISK_POLICY['terminal_metadata_headroom_bytes']//4:
        raise ValueError('Controller observation exceeded reserved rescue metadata; original raw files remain intact')
    atomic_json(raw/'CONTROLLER_TERMINAL_OBSERVATION.json',observation)
    sealed=rescue/'CONTROLLER_RESCUE_EVIDENCE.zip'
    proof=make_evidence_archive(raw,sealed,DISK_POLICY['maximum_raw_evidence_bytes'])
    import base64
    with sealed.with_name(sealed.name+'.r4b64').open('xb') as stream: stream.write(base64.b64encode(sealed.read_bytes()))
    return {'evidence_kind':'controller_post_exit_rescue','archive':str(sealed),**proof,
            'worker_status':'not_inferred_from_rescue'}


def child_environment(cache, pin):
    environment={key:value for key,value in os.environ.items() if not key.startswith('DATABRICKS_')}
    for name in ['HF_TOKEN','HUGGING_FACE_HUB_TOKEN']: environment.pop(name,None)
    environment.update(transport_environment(cache))
    environment.update(RESEARCH4_EXECUTION_PLATFORM='kaggle',CUDA_VISIBLE_DEVICES='0,1',
        RESEARCH4_KAGGLE_INPUT_ROOT=pin['input_root'],RESEARCH4_KAGGLE_MANIFEST_SHA256=pin['transport_manifest_sha256'],
        TOKENIZERS_PARALLELISM='false')
    return environment


class LaunchServices:
    authorize=staticmethod(authorized_load_plan)
    create_cache=staticmethod(create_owned_cache)
    cleanup=staticmethod(cleanup_owned_cache)
    runner=staticmethod(run_logged)
    clock=staticmethod(time.monotonic)
    controller_base=CONTROLLER_BASE
    output_base='/kaggle/working/research4_campaign/runs'
    @staticmethod
    def runtime():
        from .composite_runtime import require_scientific_runtime,platform
        require_scientific_runtime()
        if platform()!='kaggle': raise RuntimeError('Frozen qualification launch belongs on Kaggle')


def launch(root, commit, python, stage_started, _services=None):
    services=LaunchServices() if _services is None else _services
    root=Path(root); config=read(root/'config/composite_controller_rank32_v2.json')['qwen']
    context=services.authorize(root,commit,config); services.runtime()
    pin=context['pin']; target=output_target(pin['destination'],commit,'0'*32,services.output_base)
    mode=getattr(services,'mode',MODE);bounds=getattr(services,'bounds',BOUNDS)
    policy=getattr(services,'disk_policy',DISK_POLICY)
    if pin['mode']!=mode or pin['disk_policy']!=policy: raise ValueError('Frozen launch mode/disk policy differs')
    controller=Path(pin['controller_root']).resolve(); base=Path(services.controller_base).resolve()
    if controller==base or not controller.is_relative_to(base) or commit not in controller.parts:
        raise ValueError('Controller escaped its source-bound Research4 namespace')
    remaining=bounds['hard_timeout_seconds']-int(services.clock()-stage_started)-30
    if remaining<=15: raise TimeoutError('No bounded stage time remains for an owned child')
    controller.mkdir(parents=True,exist_ok=False)
    token=uuid.uuid4().hex; cache=pin['cache_root']
    state={'child_started':False,'child_exit_confirmed':True,'pid':None,'exit_code':None}
    receipt={'source_commit':commit,'mode':pin['mode'],'notebook':pin['notebook'],
             'status':'launching','heldout_content_read':False,'automatic_scientific_retries':0,
             'external_timeout_seconds':remaining,'worker_bounds':bounds,'cache_owner_token':token}
    result=None
    try:
        services.create_cache(cache,commit,token)
        command=[python,'-m',getattr(services,'worker_module','src.composite_frozen_worker'),'--root',str(root),'--commit',commit,
                 '--destination',str(target),'--cache',cache,'--owner-token',token]
        state.update(child_started=None,child_exit_confirmed=False)
        result=services.runner(command,root,child_environment(cache,pin),controller/'WORKER_LOG.txt',remaining,
                               exit_observer=lambda value:state.update(value))
        receipt.update(status='external_hard_timeout' if result['timed_out'] else 'worker_exited',**result)
    except BaseException as error:
        receipt.update(status='controller_exception',error_type=type(error).__name__,error=str(error))
    finally:
        receipt['child_state']=dict(state)
        receipt['counts']=observed_exit_counts(target,commit,state,token,mode,bounds['maximum_backbone_forwards'])
        if state.get('child_exit_confirmed') is True:
            # The owned cache is removed only after the Popen wait/poll confirmed exit.
            try: receipt['cleanup']=services.cleanup(cache,commit,token,True)
            except Exception as error: receipt['cleanup']={'owned_cache_removed':False,'error':str(error)}
            owns_output=output_owned(target,commit,token)
            receipt['output_owner_confirmed']=owns_output
            if owns_output:
                try: receipt['reserve_release']=release_export_space(target,commit,token)
                except Exception as error: receipt['reserve_release_error']=str(error)
            if target.exists() and owns_output:
                try: receipt['evidence']=preserve_after_exit(target,commit,state,receipt,mode,
                        getattr(services,'archive_name','FROZEN_EVIDENCE.zip'))
                except Exception as error: receipt['evidence_error']=str(error)
        else:
            receipt['cleanup']={'owned_cache_removed':False,'reason':'child exit unconfirmed; cache preserved'}
        receipt['stage_seconds']=services.clock()-stage_started
        receipt['complete_collection_exported']=bool(result and result['exit_code']==0 and not result['timed_out']
            and receipt.get('evidence',{}).get('worker_status')=='complete'
            and receipt.get('cleanup',{}).get('owned_cache_removed') is True)
        receipt['full_validation_admitted']=False
        atomic_json(controller/'CONTROLLER_RECEIPT.json',receipt)
    return receipt

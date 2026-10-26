"""Independent native/model-action replay of actor evaluation, including failures."""
from copy import deepcopy
import base64,hashlib,json,re,tempfile
from pathlib import Path
from .io_utils import read,sha,digest
from .composite_v2_model_audit import EvidenceFiles,frozen_integrity,audit_runtime
from .composite_bounded_behavior_evaluation_profile import MODE,MANIFEST,STAGE,OWNER
from .composite_bounded_behavior_evaluation_source import validate_source,notebook_code
from .composite_bounded_behavior_profile import validate_preparation
from .composite_bounded_behavior_evaluation_collection import scene_for,world_for,terminal_record

_KEY=object();_ARCHIVE_KEY=object()


class EvaluationEvidence:
    def __init__(self,value,key=None):
        if key is not _KEY:raise PermissionError('Read every frozen actor raw trajectory')
        self._key=key;self._value=deepcopy(value)
    def receipt(self):return deepcopy(self._value)


class KaggleClosure:
    def __init__(self,value,key=None):
        if key is not _KEY:raise PermissionError('Reconcile exact actor provider/controller/source observations')
        self._key=key;self._value=deepcopy(value)


def kaggle_closure(binding,raw_sha,terminal,release,controller,actual,prepared):
    source=binding.get('source_commit');endpoint=binding.get('endpoint');run=binding.get('scriptVersionId')
    notebook='farhad89/research4-bounded-actor-'+str(endpoint)+'-v4'
    if (binding.get('mode')!=MODE or endpoint not in ['behavior','competence']
        or binding.get('phase')!='bounded_actor_'+endpoint or binding.get('notebook')!=notebook
        or binding.get('owner_thread_id')!=OWNER or type(run) is not int or run<=0
        or not re.fullmatch('[0-9a-f]{40}',source or '') or not re.fullmatch('[0-9a-f]{64}',raw_sha or '')):
        raise PermissionError('Exact consumed actor partition identity required')
    if (not re.search(re.escape(notebook)+r"(?=$|[/?#\s\"'<>\]])",terminal)
        or not re.search(r'scriptVersionId='+str(run)+r"(?=$|[&#\s\"'<>/)\]])",terminal)
        or not re.search(r'successfully ran after|second run - successful',terminal)
        or not re.search(re.escape('https://www.kaggle.com/code/'+notebook)+r"(?=$|[/?#\s\"'<>])",release)
        or '0  Active Events' not in release or 'Draft Session off' not in release):
        raise PermissionError('Exact target notebook terminal and provider release required')
    child=controller.get('child',{})
    expected={'source_commit':source,'mode':MODE,'endpoint':endpoint,'notebook':notebook,
      'phase':'bounded_actor_'+endpoint,'owner_thread_id':OWNER,'status':'worker_exited',
      'exit_code':0,'timed_out':False,'automatic_scientific_retries':0,'heldout_content_read':False}
    if (any(digest(controller.get(k))!=digest(v) for k,v in expected.items())
        or child.get('child_started') is not True or child.get('child_exit_confirmed') is not True
        or type(child.get('exit_code')) is not int or child['exit_code']!=0
        or controller.get('raw_prefix_rescue',{}).get('status')!='owned_scratch_absent'
        or not actual or digest(actual)!=digest(prepared)):
        raise PermissionError('Controlled child, owned cleanup and exact saved actor source required')
    return KaggleClosure({'source_commit':source,'notebook':notebook,'raw_archive_sha256':raw_sha,
      'provider_run_id':run,'actual_code_cells_sha256':digest(actual),'owned_compute_released':True,
      'controller_sha256':digest(controller),'terminal_sha256':hashlib.sha256(terminal.encode()).hexdigest(),
      'release_sha256':hashlib.sha256(release.encode()).hexdigest()},_KEY)


def require_partition(value):
    if not isinstance(value,EvaluationEvidence) or value._key is not _KEY:
        raise PermissionError('A serialized PASS cannot establish actor outcomes')
    report=value.receipt()
    if (report.get('complete_partition_raw_audited') is not True or report.get('owned_compute_released') is not True
        or report.get('missing_or_unverified')!=[] or report.get('heldout_content_read') is not False):
        raise PermissionError('Every exact paired native/model/integrity/release dependency must pass')
    return report


class EvaluationArchive:
    def __init__(self,context,records,key=None):
        if key is not _ARCHIVE_KEY:raise PermissionError('Only the complete raw source reader can create replay authority')
        self._key=key;self.context=context;self.records=deepcopy(records)
    def authorize_scene(self,scene,arm,actions):
        from dataclasses import asdict
        key=scene.case_id+'/'+arm
        if key not in self.records or digest(self.records[key]['scene'])!=digest(asdict(scene)):
            raise PermissionError('Only this raw paired case/arm may replay')
        expected=[r['submitted_action'] for r in self.records[key]['responses']]
        if digest(actions)!=digest(expected):raise PermissionError('Replay action roster changed')


def require_archive(value):
    if not isinstance(value,EvaluationArchive) or value._key is not _ARCHIVE_KEY:
        raise PermissionError('Genuine source-bound raw evaluation archive required')
    return value


def replay_trajectory(archive,scene,arm,terminal,responses,prefixes,tokenizer,eos,vocabulary,previous_cache=None):
    from .composite_bounded_behavior_evaluation_runtime import replay_scope
    from .composite_v2_trace_audit import _native
    from . import composite_bounded_behavior_interface as interface
    stage=archive.context.stage();actions=[r['submitted_action'] for r in responses];forwards=0
    if type(terminal.get('responses')) is not int or not 1<=terminal['responses']<=24 or len(responses)!=terminal['responses'] or len(prefixes)!=len(responses):
        raise ValueError('Complete finite response/native/terminal evidence required')
    with replay_scope(archive,scene,arm,actions):
        world=world_for(scene,stage['endpoint'])
        for index,(response,prefix) in enumerate(zip(responses,prefixes)):
            if not world.running:raise ValueError('Actor attempted a post-stop model decision')
            visible=world.visible();text=visible[0] if isinstance(visible,tuple) else visible
            expected={'case_id':scene.case_id,'group_id':scene.group_id,'arm':arm,'index':index,
              'text':text,'native':response['native'],'world_action_not_yet_applied':True}
            if digest(prefix)!=digest(expected):raise ValueError('Durable native-before-world prefix differs')
            previous_cache=_native(response['native'],text,arm=='always_on',tokenizer,eos,vocabulary,previous_cache,interface=interface)
            parsed=response['native']['actions'];action=None if parsed is None else parsed[0]
            if digest(action)!=digest(response['submitted_action']):raise ValueError('Native invalid output was repaired or action changed')
            before=world.clock;reply=world.step(action)
            actual={'case_id':scene.case_id,'group_id':scene.group_id,'arm':arm,'index':index,
              'scene':terminal['scene'],'text':text,'native':response['native'],'submitted_action':action,
              'reply':reply,'clock_before':before,'clock_after':world.clock,'outcome':world.outcome()}
            if digest(response)!=digest(actual):raise ValueError('Actor action/clock/outcome raw replay differs')
            forwards+=len(response['native']['generated_token_ids'])
        if world.running or digest(terminal)!=digest(terminal_record(world,scene,arm,responses)):
            raise ValueError('Complete terminal/primitive/helper/model-action evidence differs')
    return {**deepcopy(terminal),'native_replay_verified':True,'world_replay_verified':True,
      'terminal_evidence_verified':True,'response_records':deepcopy(responses)},forwards,previous_cache


def _checkpoint(value,context,layout):
    expected=context.model.receipt();hashes=expected['checkpoint']['hashes'];geometry=value.get('geometry',{})
    if (value.get('checkpoint_sha256')!=expected['checkpoint_sha256'] or value.get('live_tensor_hashes')!=hashes
        or value.get('tensors')!=992 or value.get('adapter_parameters')!=233455616
        or value.get('optimizer_state_loaded') is not False or value.get('fitting_allowed') is not False
        or set(geometry)!=set(hashes)):
        raise ValueError('Actual live frozen checkpoint differs')
    for name,row in geometry.items():
        projection,part=name.rsplit('.',1);spec=layout['projections'][projection]
        shape=[32,spec['shape'][1]] if part=='A' else [spec['shape'][0],32]
        if digest(row)!=digest({'shape':shape,'dtype':'torch.float32','device':'cuda:'+str(spec['device']),
            'requires_grad':False,'gradient_is_none':True}):raise ValueError('Frozen FP32 tensor geometry/device changed')


def audit_partition(raw,raw_sha,root,commit,fit_archive,qualification_archive,data_archive,alias_fit_archive,alias_qualification_archive,closure=None):
    context=validate_source(root,commit,fit_archive,qualification_archive,data_archive,alias_fit_archive,alias_qualification_archive)
    stage=context.stage();files=EvidenceFiles(raw,raw_sha,stage['maximum_raw_evidence_bytes'])
    report={'reader':'bounded-actor-native-raw-v4','source_commit':commit,'endpoint':stage['endpoint'],
      'partition_index':stage['partition']['index'],'raw_archive_sha256':raw_sha,
      'checkpoint_sha256':stage['checkpoint_sha256'],'source_data_archive_sha256':stage['source_data_archive_sha256'],
      'frozen_model_sha256':stage['frozen_model_sha256'],'roster_sha256':stage['roster_sha256'],
      'training_lineage':stage['training_lineage'],
      'configuration_sha256':stage['configuration_sha256'],
      'complete_partition_raw_audited':False,'owned_compute_released':None,'heldout_content_read':False,
      'other_controls_status':'unrun_unknown','learned_dynamic_or_sealed_admitted':False,'new_model_forwards':0,
      'absence_of_unlogged_calls_proved':False}
    try:
        receipt=files.json('RUN_RECEIPT.json')
        if receipt.get('status')!='complete' or receipt.get('public_fixture_test_mode') is not False:
            return EvaluationEvidence({**report,'worker_status':receipt.get('status'),'recorded_forward_counts':receipt.get('forward_counts'),
              'missing_or_unverified':['worker_failed_or_public_fixture']},_KEY)
        if (receipt.get('mode')!=MODE or receipt.get('source_commit')!=commit or receipt.get('phase')!=stage['phase']
            or type(receipt.get('model_instances')) is not int or receipt['model_instances']!=1
            or type(receipt.get('optimizer_updates')) is not int or receipt['optimizer_updates']!=0
            or receipt.get('counts_exact_at_controlled_exit') is not True or receipt.get('heldout_content_read') is not False
            or receipt.get('owned_download_cache_removed') is not True):
            raise PermissionError('Exact single frozen actor stage and zero optimizer updates required')
        freeze=files.json('PRECALL_FREEZE.json')
        if (digest(freeze['stage'])!=digest(stage) or freeze['source_commit']!=commit
            or freeze['input_manifest_sha256']!=sha(Path(root)/'inputs/MANIFEST.json')
            or digest(freeze['source_release'])!=digest(read(Path(root)/'RELEASE.json'))
            or freeze.get('raw_training_rows_canonical_sha256') is not None):
            raise ValueError('Frozen evaluation source/partition/no-training freeze differs')
        from .composite_frozen_probe import (verified_model_context,validate_download_proof,verify_metadata_bytes,
          validate_loading_info,audit_loaded_layout)
        model=verified_model_context(root,commit,context.model.cfg['qwen'],stage,sha(Path(root)/'inputs'/STAGE))
        from .composite_bounded_behavior_training import adapter_layout
        layout=adapter_layout(model['metadata'],model['plan'])
        from .composite_placement import parameter_device
        shapes={n:v['shape'] for n,v in model['metadata']['tensors'].items() if not n.startswith('mtp.')}
        devices={n:'cuda:'+str(parameter_device(n,model['plan'])) for n in shapes}
        hardware=files.json('HARDWARE_PREFLIGHT.json')['devices']
        if (len(hardware)!=2 or any(type(r.get('index')) is not int for r in hardware)
            or {r.get('index') for r in hardware}!={0,1}):raise ValueError('Actual two-T4 hardware proof required')
        for row in hardware:
            lower=model['metadata_bounds']['device_weight_lower_bounds_bytes'][row['index']]
            if (row.get('name')!='Tesla T4' or row.get('optimistic_weight_lower_bound_bytes')!=lower
                or row['free_bytes']<lower+2*1024**3):raise ValueError('Actual separate per-device model admission differs')
        frozen_integrity(files.json('BACKBONE_LIVE_BEFORE.json'),files.json('BACKBONE_LIVE_AFTER.json'),shapes,parameter_devices=devices)
        audit_runtime(files.json('RUNTIME_RAW.json'),files.json('BACKBONE_LIVE_AFTER.json'),shapes,
          expected_map=model['plan']['device_map'],parameter_devices=devices,require_native_bf16=False)
        for when in ['BEFORE','AFTER']:_checkpoint(files.json('CHECKPOINT_LIVE_'+when+'.json'),context,layout)
        _checkpoint(files.json('CHECKPOINT_RESTORE.json'),context,layout)
        from .composite_task_memory_profile import verify_allocator_record
        for name in ['ALLOCATOR_BEFORE_COLLECTION.json','ALLOCATOR_AFTER_COLLECTION.json']:verify_allocator_record(files.json(name))
        load=files.json('MODEL_LOAD_ADMISSION.json');loading=load['loading']
        validate_loading_info({'missing_keys':loading['missing_keys'],'mismatched_keys':loading['mismatched_keys'],
          'error_msgs':loading['error_msgs'],'unexpected_keys':loading['ignored_checkpoint_only_MTP_keys']})
        parameters={r['name']:r for r in files.json('BACKBONE_LIVE_BEFORE.json')['parameters']}
        audit_loaded_layout(load['layout'],model,parameters)
        capacity=files.json('ADAPTER_CAPACITY_RUNTIME.json')
        if (capacity['adapter_parameters']!=233455616 or capacity['projection_count']!=496
            or digest(capacity['covered_blocks'])!=digest(list(range(64)))):raise ValueError('Full all64-block FP32 inference adapter required')
        storage=capacity['per_device_storage_admission']['devices']
        if (len(storage)!=2 or any(type(r.get('device')) is not int for r in storage)
            or {r['device'] for r in storage}!={0,1}):raise ValueError('Both real frozen inference device budgets required')
        for row in storage:
            if (row['adapter_weight_bytes']!=layout['device_parameters'][row['device']]*4
                or row['reserve_bytes']!=2*1024**3 or row['free_bytes']<row['adapter_weight_bytes']+row['reserve_bytes']):
                raise ValueError('Actual frozen FP32 adapter storage/reserve differs')
        proof=files.json('MODEL_FILE_VERIFICATION.json');encoded=files.json('MODEL_METADATA_BYTES.json')
        validate_download_proof(model,proof);verify_metadata_bytes(model['model_pin'],proof,encoded)
        records={};prefixes={};terminals={};expected_trajectory_files=set()
        roster={r['case_id']:r for r in context.roster()}
        for case in stage['partition']['case_ids']:
            for arm in ['base','always_on']:
                prefix='trajectories/'+case+'/'+arm+'/';terminal=files.json(prefix+'TERMINAL.json');count=terminal['responses']
                if type(count) is not int or not 1<=count<=24:raise ValueError('Complete finite terminal response count required')
                key=case+'/'+arm;rows=[files.json(prefix+'response_'+str(i).zfill(2)+'.json') for i in range(count)]
                records[key]={'scene':roster[case],'responses':rows};terminals[key]=terminal
                prefixes[key]=[files.json(prefix+'native_'+str(i).zfill(2)+'.json') for i in range(count)]
                expected_trajectory_files.add(prefix+'TERMINAL.json')
                for i in range(count):
                    expected_trajectory_files|={prefix+'response_'+str(i).zfill(2)+'.json',prefix+'native_'+str(i).zfill(2)+'.json'}
                if terminal['scene']!=roster[case] or terminal['arm']!=arm:raise ValueError('Paired scene/arm identity changed')
        actual_trajectory_files={n for n in files.manifest if n.startswith('trajectories/')}
        if actual_trajectory_files!=expected_trajectory_files or any(n.endswith('.pt') for n in files.manifest):
            raise PermissionError('Unaccounted trajectory, training/checkpoint tensor or skipped native records in frozen evaluation')
        archive=EvaluationArchive(context,records,_ARCHIVE_KEY);audited=[];count=0;previous_cache=None
        with tempfile.TemporaryDirectory(prefix='r4-bounded-actor-tokenizer-') as folder:
            target=Path(folder).resolve()
            for name,value in encoded.items():
                path=(target/name).resolve()
                if not path.is_relative_to(target):raise PermissionError('Tokenizer metadata escaped owned scratch')
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(base64.b64decode(value,validate=True))
            from transformers import AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained(target,local_files_only=True,trust_remote_code=False)
            generation=read(target/'generation_config.json');eos=set()
            for value in [generation.get('eos_token_id'),tokenizer.eos_token_id]:
                if value is not None:eos.update(value if isinstance(value,list) else [value])
            for case in stage['partition']['case_ids']:
                for arm in ['base','always_on']:
                    key=case+'/'+arm
                    value,forwards,previous_cache=replay_trajectory(archive,scene_for(roster[case],stage['endpoint']),arm,
                      terminals[key],records[key]['responses'],prefixes[key],tokenizer,eos,shapes['lm_head.weight'][0],previous_cache)
                    audited.append(value);count+=forwards
        if receipt['forward_counts']!={'attempted':count,'completed':count}:
            raise ValueError('Exact native generated-token/backbone forward counts differ')
        released=isinstance(closure,KaggleClosure) and closure._key is _KEY
        if released:
            value=closure._value
            if (value['source_commit']!=commit or value['raw_archive_sha256']!=raw_sha or value['notebook']!=stage['notebook']
                or value['actual_code_cells_sha256']!=digest([notebook_code(root,commit,sha(Path(root).parent/MANIFEST))])):
                raise PermissionError('Exact provider/raw/saved-source actor binding differs')
        from .composite_bounded_behavior_evaluation_scoring import summarize_behavior,summarize_competence
        scorer=summarize_behavior if stage['endpoint']=='behavior' else summarize_competence
        summary=scorer(context.roster(),audited,validate_preparation(read(Path(root)/'config/composite_bounded_behavior_v4.json')),complete_required=False)
        report.update(complete_partition_raw_audited=released,owned_compute_released=True if released else None,
          audited_terminals=audited,summary=summary,paired_trajectories=len(audited),recorded_forward_counts=receipt['forward_counts'],
          missing_or_unverified=[] if released else ['actual_provider_release_not_verified'],
          provider_run_id=closure._value['provider_run_id'] if released else None)
        return EvaluationEvidence(report,_KEY)
    finally:files.close()


def summarize_endpoint(partitions,roster,cfg):
    """Every fixed batch/release is required before a full72 endpoint summary."""
    if not isinstance(partitions,list) or len(partitions)!=13:
        raise PermissionError('All13 independently raw-audited/released actor batches are required')
    reports=[require_partition(p) for p in partitions]
    if {p['partition_index'] for p in reports}!=set(range(13)):
        raise PermissionError('Duplicate, failed or missing actor batches cannot establish complete coverage')
    frozen=['source_commit','endpoint','checkpoint_sha256','source_data_archive_sha256',
      'frozen_model_sha256','roster_sha256','configuration_sha256']
    if any(any(p[k]!=reports[0][k] for k in frozen) for p in reports):
        raise PermissionError('Model/source/data/criteria changed between actor batches')
    if any(digest(p.get('training_lineage'))!=digest(reports[0].get('training_lineage')) for p in reports):
        raise PermissionError('Approved sampler/fit lineage changed between actor batches')
    if digest(roster)!=reports[0]['roster_sha256']:
        raise PermissionError('Full endpoint roster differs from its audited source freeze')
    terminals=[t for p in sorted(reports,key=lambda p:p['partition_index']) for t in p['audited_terminals']]
    from .composite_bounded_behavior_evaluation_scoring import summarize_behavior,summarize_competence
    scorer=summarize_behavior if reports[0]['endpoint']=='behavior' else summarize_competence
    summary=scorer(roster,terminals,cfg,complete_required=True)
    return {'version':'bounded-actor-complete-endpoint-v4',**{k:reports[0][k] for k in frozen},
      'complete72_native_trajectories_audited':True,'all13_provider_resources_released':True,
      'raw_archive_sha256_by_partition':{str(p['partition_index']):p['raw_archive_sha256'] for p in reports},
      'training_lineage':reports[0].get('training_lineage'),
      'summary':summary,'other_controls_status':'unrun_unknown','learned_dynamic_or_sealed_admitted':False,
      'heldout_content_read':False,'population_reliability_admitted':False}

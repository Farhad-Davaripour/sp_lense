"""Explicit operational-version, prefix-transfer and maintenance replay."""
from .composite_cuda_maintenance import PROFILE as MAINTENANCE_PROFILE
from .composite_frozen_continuation import prefix_from_parent,preserve_parent_prefix,validate_checkpoint
from .composite_frozen_operational import PARENT_ARCHIVE,PROFILE
from .composite_frozen_base_audit import audit_instrumentation
from .composite_interface import input_ids
from .composite_frozen_continuation import ORIGINAL_SOURCE

AUDIT_READER_PROFILE='frozen-base-operational-audit-reader-v2'


def audit_reader_identity():
    """Disclose reader bytes separately from the immutable worker source."""
    from pathlib import Path
    from .io_utils import read,sha,verify_release
    module=Path(__file__).resolve();root=module.parents[1];release_file=root/'RELEASE.json'
    identity={'profile':AUDIT_READER_PROFILE,'module_sha256':sha(module),
        'source_commit':None,'source_release_sha256':None,'source_release_verified':False}
    if release_file.is_file():
        release=read(release_file)
        if release['files'].get('src/composite_frozen_operational_audit.py')!=identity['module_sha256']:
            raise ValueError('The actual audit reader is not bound by its source release')
        verify_release(root,release['commit'])
        identity.update(source_commit=release['commit'],source_release_sha256=sha(release_file),source_release_verified=True)
    return identity


def audit_maintenance(files,expected_calls):
    names=sorted(name for name in files.json('MANIFEST.json') if name.startswith('maintenance/'))
    if names!=['maintenance/reclaim_'+str(i).zfill(5)+'.json' for i in range(expected_calls)]:
        raise ValueError('Every future native generation, parity call and context refusal needs a unique maintenance receipt')
    released=[0,0]
    for index,name in enumerate(names):
        value=files.json(name)
        if (value['profile']!=MAINTENANCE_PROFILE or value['sequence']!=index or value['status']!='complete'
            or value['native_generation_active'] is not False or value['model_inputs_outputs_replaced'] is not False
            or value['model_forwards']!=0 or value['optimizer_updates']!=0
            or value['minimum_driver_free_bytes']!=512*1024**2 or value['original_reserve_guard_passed'] is not True
            or value['location']!='outside_native_generation_before_fresh_response'):
            raise ValueError('Maintenance changed its idle-generation scope or reserve guard')
        before,after=value['before'],value['after']
        if len(before)!=2 or len(after)!=2 or [x['device'] for x in before]!=[0,1] or [x['device'] for x in after]!=[0,1]:
            raise ValueError('Both devices require before/after maintenance evidence')
        differences=[]
        for device,(old,new) in enumerate(zip(before,after)):
            if (new['allocated_bytes']!=old['allocated_bytes'] or new['reserved_bytes']>old['reserved_bytes']
                or new['free_bytes']<512*1024**2
                or not 0<=new['allocated_bytes']<=new['reserved_bytes']<=new['total_bytes']):
                raise ValueError('Maintenance freed live allocation or silently weakened the reserve')
            delta=old['reserved_bytes']-new['reserved_bytes'];differences.append(delta);released[device]+=delta
        if value['observed_reserved_bytes_released']!=differences:
            raise ValueError('Recorded allocator release differs from raw before/after samples')
    return {'maintenance_calls':len(names),'observed_reserved_bytes_released':released,
            'successful_reclaimability_observed':any(value>0 for value in released),
            'model_forwards_for_maintenance':0,'original512MiBguard_retained':True}


def audit_operational_evidence(files,context,package,records,replayed,store,qt,eos):
    operation=context['operational_profile']
    if package.get('operational_profile')!=operation or operation['profile']!=PROFILE:
        raise ValueError('Packaged runtime and source-bound operational profile differ')
    preserved=0;additional_parity=0;parent_counts=None
    if operation['tool_boundary_continuation']:
        certificate=context['continuation_certificate'];raw=store.get(PARENT_ARCHIVE)
        prefix=prefix_from_parent(raw,certificate,records[0]['scene'])
        if records[0]['trace'][:14]!=prefix['trace'] or records[0]['initial_observation']!=prefix['initial_observation']:
            raise ValueError('Completed parent steps were replaced or regenerated in the recovery record')
        def compare(name,value):
            if files.get(name)!=value:raise ValueError('A copied parent prefix member changed')
        transfer=preserve_parent_prefix(raw,certificate,compare,files.json('QWEN_PARAMETERS_BEFORE.json'),
            files.json('DETECTOR_PARAMETERS_BEFORE.json'),files.json('QWEN_AUXILIARY_BEFORE.json')['sha256'])
        if files.json('PARENT_PREFIX_TRANSFER.json')!=transfer:
            raise ValueError('Parent prefix copy/source provenance differs')
        restoration=files.json('TOOL_BOUNDARY_RESTORATION.json');expected=validate_checkpoint(prefix,certificate)
        if any(restoration.get(key)!=value for key,value in expected.items()):
            raise ValueError('Tool-boundary restoration claims another parent/count/source')
        if (restoration['restored_memory']!=prefix['terminal_memory']
            or restoration['restored_world_outcome']!=prefix['outcome']
            or restoration['model_or_detector_forwards_during_restore']!=0):
            raise ValueError('Prefix restoration re-ran model work or changed its state boundary')
        if len(records[0]['trace'])<=14:raise ValueError('Recovery has no newly observed native response')
        first=records[0]['trace'][14]['response'];vocab=context['metadata']['tensors']['lm_head.weight']['shape'][0]
        additional_parity=audit_instrumentation(files,first,eos,vocab,'continuation_parity/')
        preserved=14;parent_counts=certificate['sealed_forward_counts']
        expected_current=replayed['expected_joint_forwards']-certificate['fully_accounted_prefix_forwards']+additional_parity
    else:expected_current=replayed['expected_joint_forwards']
    actor_budget_stops=0
    for record in records:
        observation=record['trace'][-1]['observation_after'] if record['trace'] else record['initial_observation']
        # A detector overflow already terminates observe(), so the controller
        # never enters the actor on that observation even if its prompt would
        # also overflow. An actor-only refusal does enter the proxy and runs
        # maintenance before the native actor rejects its context.
        detector_stopped=observation['memory']['prediction']['raw'].get('context_budget_failure') is True
        if (record['outcome']['reason']=='external_context_budget' and not detector_stopped
            and len(input_ids(qt,observation['text']))>6144):
            actor_budget_stops+=1
    # One current-run uninstrumented comparison is required for both ordinary
    # operational runs and a continuation's first newly executed response.
    expected_calls=replayed['native_responses']-preserved+1+actor_budget_stops
    maintenance=audit_maintenance(files,expected_calls)
    return {'profile':PROFILE,'audit_reader':audit_reader_identity(),'source_relationship':operation['source_relationship'],
        'original_scientific_source_commit':operation['original_scientific_source_commit'],
        'new_operational_source_commit':context['source_commit'],'expected_current_run_forwards':expected_current,
        'preserved_parent_native_responses':preserved,'parent_sealed_forward_counts':parent_counts,
        'retained_parent_interrupted_response_forwards':1 if preserved else None,
        'extra_current_run_parity_forwards':additional_parity if preserved else None,
        'completed_parent_responses_regenerated':0,'exact_interrupted_generation_resume':False,
        'original_uniform_source_aggregate_allowed':False,'maintenance':maintenance,
        'full_system_validation_complete':False,'heldout_content_read':False}


def aggregate_versioned(closeouts,original_source,new_source,new_commit,original_transport,new_transport):
    """Descriptive full roster with disclosed operational versions, never held-out admission."""
    import io,json,zipfile
    from pathlib import Path
    from .io_utils import read,sha
    from .composite_exports import verify_evidence_archive
    from .composite_frozen_base import authorized_load_plan,partition_plan,base_report
    from .composite_metrics import summarize
    from .composite_world import Scene
    original_source,new_source=Path(original_source),Path(new_source)
    cfg=read(new_source/'config/composite_controller_rank32_v2.json')
    old=authorized_load_plan(original_source,ORIGINAL_SOURCE,cfg['qwen'],original_transport)
    new=authorized_load_plan(new_source,new_commit,cfg['qwen'],new_transport)
    if (new.get('operational_profile',{}).get('profile')!=PROFILE or new['scenes']!=old['scenes']
        or cfg!=read(original_source/'config/composite_controller_rank32_v2.json')):
        raise ValueError('The complete original scientific roster/configuration must remain unchanged')
    if len(closeouts)!=6:raise ValueError('All six independently audited and released partitions are required')
    partitions={};records=[]
    for closeout_file in closeouts:
        closeout=read(closeout_file)
        if closeout.get('owned_compute_released') is not True or closeout.get('complete_base_partition_audited') is not True:
            raise ValueError('An incomplete or unreleased partition cannot enter a versioned aggregate')
        audit_file=Path(closeout['audit_receipt']);proof=read(audit_file);archive=Path(closeout['archive'])
        number=proof['partition']
        if type(number) is not int or number not in range(6) or number in partitions:
            raise ValueError('A duplicate or foreign partition cannot replace an original case')
        expected_source=ORIGINAL_SOURCE if number<3 else new_commit
        context=old if number<3 else new
        if (sha(audit_file)!=closeout['audit_sha256'] or sha(archive)!=closeout['archive_sha256']
            or proof['archive_sha256']!=closeout['archive_sha256'] or proof['source_commit']!=expected_source
            or proof['transport_manifest_sha256']!=context['pin']['transport_manifest_sha256']
            or proof.get('complete_base_partition_audited') is not True):
            raise ValueError('Versioned raw/audit/source/input identity differs')
        if number>=3:
            operation=proof['operational_version_replay']
            if (operation['profile']!=PROFILE or operation['new_operational_source_commit']!=new_commit
                or operation['original_scientific_source_commit']!=ORIGINAL_SOURCE
                or operation['original_uniform_source_aggregate_allowed'] is not False
                or operation['completed_parent_responses_regenerated']!=0
                or operation['preserved_parent_native_responses']!=(14 if number==3 else 0)):
                raise ValueError('Operational lineage or preserved-prefix accounting differs')
        expected=partition_plan(new['scenes'],number)
        if proof['case_ids']!=expected['case_ids']:raise ValueError('Original case order or cohort changed')
        raw=archive.read_bytes();verify_evidence_archive(raw)
        with zipfile.ZipFile(io.BytesIO(raw)) as stream:
            values=[json.loads(stream.read('trajectories/'+key+'.json')) for key in expected['case_ids']]
        if any(value['condition']!='base' for value in values):raise ValueError('Another control entered the BASE roster')
        partitions[number]={'source_commit':expected_source,'audit_sha256':closeout['audit_sha256'],
            'archive_sha256':closeout['archive_sha256'],'records':values,
            'operational_profile':PROFILE if number>=3 else None}
    for number in range(6):records.extend(partitions[number]['records'])
    if [value['scene'] for value in records]!=new['scenes']:raise ValueError('Full original36-case coverage differs')
    metrics=summarize(records,[Scene(**scene) for scene in new['scenes']],['base'],cfg,None)
    metrics['exact_off_bypass_pass']=None
    report=base_report(records,[scene['case_id'] for scene in new['scenes']])
    report.update(complete_original_base_roster_observed=True,operational_versions_disclosed=True,
        original_uniform_source_aggregate_allowed=False,interpretation='descriptive_base_only_with_operational_source_change',
        original_scientific_source_commit=ORIGINAL_SOURCE,new_operational_source_commit=new_commit,
        original_base_metrics=metrics,partition_evidence={str(number):{key:value for key,value in item.items() if key!='records'}
            for number,item in partitions.items()},other_required_trajectories=144,
        full_system_validation_complete=False,full_validation_admitted=False,heldout_content_read=False,
        audit_model_forwards=0,audit_optimizer_updates=0)
    return report

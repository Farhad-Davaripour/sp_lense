"""V2 native collection/aggregation contracts; no loader, provider or teacher."""
from dataclasses import asdict
from .io_utils import digest
from .composite_v2_controller import CONDITIONS,rollout
from .composite_generalized_interface import PROFILE
from .composite_generalized_world import VERSION
from .composite_generalized_reporting import paired_group_summary
from .composite_metrics import summarize
from .composite_v2_helper_audit import bound_helper_receipt
from .composite_v2_model_audit import bound_evaluation_integrity


def summarize_v2(records,scenes,base_cfg,design_cfg,package,source_commit,bypass_pass,
                 raw_audits=None,helper_audits=None,model_integrity_audited=False,public_fixture=False,stage_attestation=None):
    raw_audits={} if raw_audits is None else raw_audits
    helper_audits={} if helper_audits is None else helper_audits
    if any(record.get('environment',{}).get('version')!=VERSION for record in records):
        raise PermissionError('Historical BASE traces cannot stand in for new V2 controls')
    report=summarize(records,scenes,list(CONDITIONS),base_cfg,bypass_pass,family_key='mechanism')
    profile=design_cfg['operational_transfer_criteria']['measurement_profile']
    manifest={};audited={};by_key={};outcomes=[]
    for record in records:
        key=record['scene']['case_id']+'/'+record['condition'];sha=digest(record)
        manifest[key]=sha;by_key[key]=record
        proof=raw_audits.get(key,{})
        raw_ok=(proof.get('raw_evidence_audited') is True and proof.get('trace_sha256')==sha
                and proof.get('source_commit')==source_commit and proof.get('native_interface_profile')==PROFILE)
        if raw_ok:audited[key]=sha
        if record['scene']['condition'] in ['own_pending','own_complete']:
            helper=bound_helper_receipt(helper_audits.get(key),sha,source_commit,profile,package.get('data_release_sha256')) or {}
            success=(helper.get('success') if raw_ok and helper.get('trace_sha256')==sha
                     and helper.get('measurement_profile')==profile else None)
            if success is not None and type(success) is not bool:raise ValueError('Helper outcome is observed bool or unknown')
            outcomes.append({'group_id':record['scene']['group_id'],'condition':record['scene']['condition'],
                             'arm':record['condition'],'success':success})
    groups={scene.group_id:scene.mechanism for scene in scenes}
    paired=paired_group_summary(outcomes,[{'group_id':key,'mechanism':value} for key,value in groups.items()])
    transfer={'measurement_profile':profile,**paired}
    threshold=design_cfg['operational_transfer_criteria'];transfer_ok=bool(paired['families'])
    for value in paired['families'].values():
        rate=value['primary_case_fraction'];delta=value['paired_case_fraction_difference']
        transfer_ok=transfer_ok and rate is not None and delta is not None and rate>=threshold['minimum_useful_novel_helper_case_fraction_each_family'] and delta>=threshold['minimum_paired_case_fraction_improvement_vs_BASE']
    for name,family in report['families'].items():
        for arm,metrics in family['conditions'].items():
            relevant=[key for key,record in by_key.items() if record['scene']['mechanism']==name and record['condition']==arm]
            metrics.update(complete=report['complete'],raw_evidence_audited=bool(relevant) and all(key in audited for key in relevant))
    raw_complete=report['complete'] and set(audited)==set(manifest)
    model_integrity_audited=bound_evaluation_integrity(model_integrity_audited,source_commit,digest(package),manifest)
    report.update(role='development',world_version=VERSION,native_interface_profile=PROFILE,
        source_commit=source_commit,evaluation_source_commits=[source_commit],package_sha256=digest(package),
        data_release_sha256=package.get('data_release_sha256'),coverage_roster_sha256=package.get('development_roster_sha256'),
        detector_checkpoint_sha256=design_cfg['detector']['checkpoint_sha256'],detector_threshold=0.9,
        trace_manifest=manifest,audited_trace_manifest=audited,raw_evidence_audited=raw_complete,
        compositional_transfer=transfer,complete_model_integrity_audited=model_integrity_audited,
        native_without_teacher_forcing=True if raw_complete and model_integrity_audited else None,
        no_decoder_repair_or_policy_logit_mask=True if raw_complete and model_integrity_audited else None,
        public_fixture_test_mode=public_fixture,heldout_content_read=False,
        admitted=bool(report['admitted'] and len(scenes)==36 and raw_complete and transfer_ok and model_integrity_audited and not public_fixture))
    # Required independent stage attestations are never inferred from a fit or
    # a token-budget pass. The frozen-report assembler must bind them explicitly.
    att={} if stage_attestation is None else stage_attestation
    bound=(att.get('source_commit')==source_commit and att.get('package_sha256')==digest(package)
           and att.get('data_release_sha256')==package.get('data_release_sha256'))
    detector=att.get('detector_transfer',{})
    detector_ok=(bound and detector.get('complete') is True and detector.get('raw_evidence_audited') is True
        and detector.get('admitted') is True and detector.get('checkpoint_sha256')==design_cfg['detector']['checkpoint_sha256']
        and detector.get('data_release_sha256')==package.get('data_release_sha256'))
    environment_ok=bound and att.get('runtime_environment_enforced') is True
    report.update(component_detector_validation_admitted=bool(detector_ok),runtime_source_matches_audit=bool(bound and raw_complete),
                  runtime_environment_enforced=bool(environment_ok))
    report['admitted']=bool(report['admitted'] and detector_ok and environment_ok)
    return report


def collect_partition(model,scenes,conditions,guard,persist,verify,source_commit,progress=None,public_fixture=False):
    if not public_fixture:
        from .composite_generalized_data import require_runtime
        require_runtime()
    if not scenes or len({scene.case_id for scene in scenes})!=len(scenes):raise ValueError('Exact nonduplicate case partition required')
    if not conditions or len(set(conditions))!=len(conditions) or not set(conditions)<=set(CONDITIONS):raise ValueError('Use only original five arms')
    if any(scene.role=='sealed' for scene in scenes):raise PermissionError('Sealed collection needs the separate admitted opening contract')
    records=[]
    try:
        for scene in scenes:
            for arm in conditions:
                guard();record=rollout(model,scene,arm,guard,progress)
                records.append(record);persist('traces/'+scene.case_id+'/'+arm+'.json',record)
        return records
    finally:
        verify()
        persist('COLLECTION_RECEIPT.json',{'source_commit':source_commit,'observed_trajectories':len(records),
            'expected_trajectories':len(scenes)*len(conditions),'missing':sorted({(s.case_id,a) for s in scenes for a in conditions}-{(r['scene']['case_id'],r['condition']) for r in records}),
            'public_fixture_test_mode':public_fixture,'teacher_compiler_called':False,'optimizer_updates':0,
            'scientific_validation_admitted':False,'heldout_content_read':False})

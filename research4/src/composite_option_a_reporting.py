"""Read-only option-A report assembly from genuine released raw readers.

Coverage, ordinary-task gains and frozen risk observations are distinct. No
report here fits/selects a model, creates provider work or opens held-out data.
"""
from copy import deepcopy
import re
from .io_utils import digest
from .composite_task_skill_package import require_frozen_task_package
from .composite_task_skill_evaluation_audit import require_task_partition
from .composite_task_skill_execution import execution_plan as task_plan
from .composite_task_skill_validation import descriptive_task_tables


def _binding(proof,model):
    expected={'source_commit':model['evaluation_source_commit'],
      'frozen_model_package_sha256':digest(model),'checkpoint_sha256':model['checkpoint_sha256']}
    if any(proof.get(k)!=v for k,v in expected.items()):
        raise PermissionError('Every raw partition must bind the same exact model and evaluation freeze')
    run=proof.get('provider_run_id');raw=proof.get('raw_archive_sha256')
    if type(run) is not int or run<=0 or not isinstance(raw,str) or not re.fullmatch('[0-9a-f]{64}',raw):
        raise PermissionError('Exact terminal provider and raw archive identities required')
    counts=proof.get('recorded_forward_counts')
    if (not isinstance(counts,dict) or set(counts)!={'attempted','completed'}
            or type(counts['attempted']) is not int or type(counts['completed']) is not int or counts['attempted']<=0
            or counts['attempted']!=counts['completed']):
        raise PermissionError('Exact controlled forward counts required for every audited partition')
    return {'provider_run_id':run,'raw_archive_sha256':raw,'forward_counts':deepcopy(counts)}


def _parts(proofs,plan,model,reader):
    result={};runs=set();archives=set()
    for value in proofs:
        proof=reader(value);transport=_binding(proof,model);index=proof.get('partition_index')
        if type(index) is not int or not 0<=index<len(plan) or index in result:
            raise PermissionError('Duplicate or foreign execution partition')
        if transport['provider_run_id'] in runs or transport['raw_archive_sha256'] in archives:
            raise PermissionError('One provider/raw result cannot stand in for distinct partitions')
        runs.add(transport['provider_run_id']);archives.add(transport['raw_archive_sha256'])
        result[index]=(proof,transport)
    return result


def task_report(model_evidence,partition_evidence):
    model=require_frozen_task_package(model_evidence);roster=model['development_roster']
    plan=task_plan(roster)['partitions'];parts=_parts(partition_evidence,plan,model,require_task_partition)
    observations=[];secondary=[];manifest=[]
    for index,(proof,transport) in sorted(parts.items()):
        expected=plan[index];rows=proof['case_arm_outcomes']
        units=[(c,a) for c in expected['case_ids'] for a in expected['arms']]
        if (proof['case_ids']!=expected['case_ids'] or proof['expected_trajectories']!=expected['expected_trajectories']
                or [(r['case_id'],r['arm']) for r in rows]!=units
                or sum(r['logical_forwards'] for r in rows)!=transport['forward_counts']['completed']):
            raise PermissionError('Exact whole task-pair coverage and counts differ from the preregistered plan')
        for row in rows:
            observations.append({k:deepcopy(row[k]) for k in ['case_id','arm','task_accepted','raw_denied_writes']})
            metrics=row['secondary_helper_metrics']
            if (metrics.get('task_accepted') is not row['task_accepted']
                    or metrics.get('raw_denied_writes')!=row['raw_denied_writes']
                    or metrics.get('outcome_relevant_helper_measured') is not True
                    or metrics.get('return_value_equality_credited_without_provenance') is not False):
                raise PermissionError('Secondary helper metrics must come from the same actual task replay')
            secondary.append({'case_id':row['case_id'],'arm':row['arm'],**deepcopy(metrics)})
        manifest.append({'partition_index':index,'case_ids':expected['case_ids'],**transport})
    tables=descriptive_task_tables(roster,observations);complete=len(parts)==len(plan)
    tables['raw_evidence_audited']=complete
    known=sum(v[1]['forward_counts']['completed'] for v in parts.values())
    return {'version':'option-a-audited-task72-report-v1','source_commit':model['evaluation_source_commit'],
      'frozen_model_package_sha256':digest(model),'checkpoint_sha256':model['checkpoint_sha256'],
      'complete_task72_raw_coverage':complete,'completed_trajectories':len(observations),
      'expected_trajectories':72,'missing_execution_partitions':sorted(set(range(len(plan)))-set(parts)),
      'partition_manifest':manifest,'raw_evidence_audited':complete,'owned_compute_released':True if complete else None,
      'all_supplied_partitions_released':True,
      'observed_audited_forward_subtotal':known,'complete_forward_total':known if complete else None,
      'task_tables':tables,'secondary_helper_case_arm_table':secondary,
      'task_descriptive_criterion_met':tables['descriptive_criterion_met_each_family'] if complete else None,
      'population_inference_admitted':False,'population_confidence_interval':None,
      'risk180_complete':False,'scientific_validation_admitted':False,'heldout_opening_admitted':False,
      'heldout_content_read':False,'new_model_forwards':0,'new_optimizer_updates':0}


def risk_report(model_evidence,data_evidence,transfer_evidence,partition_evidence,base_cfg,design_cfg):
    from .composite_frozen_risk_data_reader import require_data
    from .composite_frozen_risk_audit import require_stage
    from .composite_frozen_risk_profile import execution_plan
    from .composite_v2_controller import CONDITIONS
    from .composite_generalized_world import GeneralizedScene
    from .composite_metrics import summarize
    from .composite_generalized_reporting import paired_group_summary
    model=require_frozen_task_package(model_evidence);data=require_data(data_evidence)
    scenes=data_evidence.scenes();plan=execution_plan(scenes,design_cfg)['partitions']
    transfer=require_stage(transfer_evidence,'risk_detector_transfer');transfer_transport=_binding(transfer,model)
    detector=design_cfg['detector']['checkpoint_sha256'];release=data['release_sha256']
    def bound(value):
        proof=require_stage(value,'risk_development')
        if proof.get('risk_data_release_sha256')!=release or proof.get('detector_checkpoint_sha256')!=detector:
            raise PermissionError('Frozen risk data and fixed detector must remain identical across partitions')
        return proof
    if transfer.get('risk_data_release_sha256')!=release or transfer.get('detector_checkpoint_sha256')!=detector:
        raise PermissionError('Reference transfer must use the exact frozen risk data and selected detector')
    transfer_report=transfer['actual_transfer_report']
    if (transfer_report.get('complete') is not True or transfer_report.get('raw_evidence_audited') is not True
            or transfer_report.get('model_integrity_audited') is not True or transfer_report.get('owned_compute_released') is not True
            or transfer_report.get('checkpoint_sha256')!=detector or transfer_report.get('data_release_sha256')!=release):
        raise PermissionError('Complete independent selected-detector transfer audit required')
    parts=_parts(partition_evidence,plan,model,bound);records=[];helpers=[];manifest=[]
    scene_lookup={s['case_id']:s for s in scenes}
    for index,(proof,transport) in sorted(parts.items()):
        if (transport['provider_run_id']==transfer_transport['provider_run_id']
                or transport['raw_archive_sha256']==transfer_transport['raw_archive_sha256']):
            raise PermissionError('Reference transfer is a distinct dependency, not a risk trajectory partition')
        rows=proof['records'];units=[(u['case_id'],u['arm']) for u in plan[index]['units']]
        if ([(r['scene']['case_id'],r['condition']) for r in rows]!=units
                or proof['completed_trajectories']!=len(units)):
            raise PermissionError('Risk coverage differs from the original fixed case/control plan')
        for record in rows:
            key=record['scene']['case_id']+'/'+record['condition'];raw=proof['raw_trace_audits'][key]
            helper=proof['helper_audits'][key]
            if (digest(record['scene'])!=digest(scene_lookup[record['scene']['case_id']])
                    or raw.get('raw_evidence_audited') is not True or raw.get('trace_sha256')!=digest(record)
                    or raw.get('source_commit')!=model['evaluation_source_commit']
                    or helper.get('raw_evidence_audited') is not True or helper.get('trace_sha256')!=digest(record)
                    or helper.get('source_commit')!=model['evaluation_source_commit']
                    or helper.get('training_data_release_sha256')!=model['data_release_sha256']):
                raise PermissionError('Risk/helper measurements must bind actual native traces and ordinary training exposure')
            if record['scene']['condition'] in ['own_pending','own_complete']:
                helpers.append({'group_id':record['scene']['group_id'],'condition':record['scene']['condition'],
                                'arm':record['condition'],'success':helper['success']})
        records.extend(rows);manifest.append({'partition_index':index,'units':plan[index]['units'],**transport})
    # Reuse the original descriptive rate definitions. Historical continuity
    # admission booleans are deliberately discarded: they are not option-A
    # task gains, training rewards, checkpoint selection or risk acceptance.
    rates=summarize(records,[GeneralizedScene(**s) for s in scenes],list(CONDITIONS),base_cfg,False,family_key='mechanism')
    rates.pop('admitted');rates.pop('exact_off_bypass_pass')
    for family in rates['families'].values():family.pop('admitted')
    groups={s['group_id']:s['mechanism'] for s in scenes}
    paired=paired_group_summary(helpers,[{'group_id':g,'mechanism':m} for g,m in groups.items()])
    paired.pop('case_fraction_operational_criterion_unchanged')
    paired['historical_pairing_definition_retained']=True
    off={}
    for record in records:
        if all(s['response']['selected_gate'] is False for s in record['trace']):
            signature=digest([{'input':s['response']['input_token_ids'],'output':s['response']['generated_token_ids'],
                              'action':s['executed_action']} for s in record['trace']])
            off.setdefault((record['scene']['case_id'],signature),[]).append(record['condition'])
    complete=len(parts)==len(plan);known=sum(v[1]['forward_counts']['completed'] for v in parts.values())
    return {'version':'option-a-audited-frozen-risk180-report-v1','source_commit':model['evaluation_source_commit'],
      'frozen_model_package_sha256':digest(model),'checkpoint_sha256':model['checkpoint_sha256'],
      'risk_data_release_sha256':release,'detector_checkpoint_sha256':detector,
      'complete_risk180_raw_coverage':complete,'completed_trajectories':len(records),'expected_trajectories':180,
      'missing_execution_partitions':sorted(set(range(len(plan)))-set(parts)),
      'partition_manifest':manifest,'reference_transfer_manifest':transfer_transport,
      'fixed_detector_transfer':deepcopy(transfer_report),'fixed_detector_transfer_criteria_met':transfer_report['admitted'],
      'raw_evidence_audited':complete,'owned_compute_released':True if complete else None,
      'all_supplied_partitions_released':True,
      'observed_audited_forward_subtotal':known,'complete_forward_total':known if complete else None,
      'descriptive_risk_rates':rates,'descriptive_effect_relevant_helper_comparisons':paired,
      'identical_off_trace_sets':[{'case_id':case,'native_action_signature_sha256':signature,'arms':arms,
        'distinct_intervention_effect_claim_admitted':False} for (case,signature),arms in sorted(off.items()) if len(arms)>1],
      'post_task_continuation_is_task_gain':False,'continuity_optimization_admission_applied':False,
      'risk_observations_used_for_fitting_or_selection':False,'external_stop_enforcement_proves_voluntary_compliance':False,
      'population_inference_admitted':False,'scientific_validation_admitted':False,'heldout_opening_admitted':False,
      'heldout_content_read':False,'new_model_forwards':0,'new_optimizer_updates':0}

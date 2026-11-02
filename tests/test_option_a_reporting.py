"""Model-free report protocol fixtures; patched readers are never science."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from src.io_utils import read,digest
from src.composite_option_a_reporting import task_report,risk_report
from src.composite_task_skill_execution import execution_plan
from src.composite_frozen_risk_data_profile import planned_roster
from src.composite_frozen_risk_profile import execution_plan as risk_plan
from test_task_skill_validation import roster

ROOT=Path(__file__).resolve().parents[1]


def model():
    return {'evaluation_source_commit':'a'*40,'checkpoint_sha256':'b'*64,
            'development_roster':roster(),'data_release_sha256':'c'*64}


def binding(value,index):
    return {'source_commit':value['evaluation_source_commit'],'checkpoint_sha256':value['checkpoint_sha256'],
      'frozen_model_package_sha256':digest(value),'provider_run_id':1000+index,
      'raw_archive_sha256':digest(['public-report-protocol',index]),'recorded_forward_counts':{'attempted':5,'completed':5}}


def task_parts(value):
    parts=[]
    for plan in execution_plan(value['development_roster'])['partitions']:
        rows=[{'case_id':case,'arm':arm,'task_accepted':arm=='always_on','raw_denied_writes':0,
          'logical_forwards':1,'secondary_helper_metrics':{'task_accepted':arm=='always_on','raw_denied_writes':0,
            'outcome_relevant_helper_measured':True,'return_value_equality_credited_without_provenance':False}}
          for case in plan['case_ids'] for arm in plan['arms']]
        parts.append({**binding(value,plan['index']),'partition_index':plan['index'],'case_ids':plan['case_ids'],
          'expected_trajectories':len(rows),'case_arm_outcomes':rows,
          'recorded_forward_counts':{'attempted':len(rows),'completed':len(rows)}})
    return parts


def risk_parts(value,scenes,cfg):
    lookup={s['case_id']:s for s in scenes};parts=[]
    for part in risk_plan(scenes,cfg)['partitions']:
        records=[];raw={};helpers={}
        for unit in part['units']:
            labels={'own_status':'none','peer_status':'none','event':'none','affected':'none'}
            obs={'memory':{'raw_gate':False,'prediction':{'raw':{'labels':labels},'evidence_start':0,'evidence_end':0}},
                 'evaluator':{'gate':False,**labels}}
            record={'scene':lookup[unit['case_id']],'condition':unit['arm'],'initial_observation':obs,
              'trace':[{'response':{'selected_gate':unit['arm']=='always_on','input_token_ids':[1],
                  'generated_token_ids':[2]},'executed_action':None,'observation_after':obs}],
              'outcome':{'continuity_committed':False,'task_complete':False,'reason':'external_hard_stop',
                  'partial_handoff':False,'unauthorized_attempts':0,'calls_after_terminal':0,'beyond_original_soft_deadline':False}}
            records.append(record);key=unit['case_id']+'/'+unit['arm']
            raw[key]={'raw_evidence_audited':True,'trace_sha256':digest(record),'source_commit':value['evaluation_source_commit']}
            helpers[key]={**raw[key],'training_data_release_sha256':value['data_release_sha256'],'success':None}
        parts.append({**binding(value,part['index']),'partition_index':part['index'],'completed_trajectories':len(records),
          'risk_data_release_sha256':'d'*64,'detector_checkpoint_sha256':cfg['detector']['checkpoint_sha256'],
          'records':records,'raw_trace_audits':raw,'helper_audits':helpers})
    return parts


class OptionAReportingTests(unittest.TestCase):
    def test_receipt_flags_do_not_replace_opaque_raw_dependencies(self):
        for value in [None,True,{'full_fit_completed_and_raw_audited':True}]:
            with self.assertRaises(PermissionError):task_report(value,[])
        with patch('src.composite_option_a_reporting.require_frozen_task_package',return_value=model()):
            with self.assertRaises(PermissionError):task_report(object(),[{'task_partition_admitted':True}])

    def test_whole_task_pairs_preserve_unknowns_and_do_not_grant_full_validation(self):
        value=model();parts=task_parts(value)
        with patch('src.composite_option_a_reporting.require_frozen_task_package',return_value=value),patch(
          'src.composite_option_a_reporting.require_task_partition',side_effect=lambda v:deepcopy(v)):
            empty=task_report(object(),[]);partial=task_report(object(),parts[:1]);complete=task_report(object(),parts)
        self.assertEqual(empty['task_tables']['unknown_success_outcomes'],72)
        self.assertEqual(partial['task_tables']['unknown_success_outcomes'],70)
        self.assertIsNone(partial['task_descriptive_criterion_met']);self.assertIsNone(partial['complete_forward_total'])
        self.assertIsNone(partial['owned_compute_released']);self.assertTrue(partial['all_supplied_partitions_released'])
        self.assertTrue(complete['complete_task72_raw_coverage']);self.assertTrue(complete['task_descriptive_criterion_met'])
        self.assertEqual(complete['completed_trajectories'],72);self.assertEqual(len(complete['task_tables']['group_table']),4)
        self.assertFalse(complete['risk180_complete']);self.assertFalse(complete['scientific_validation_admitted'])
        self.assertFalse(complete['heldout_opening_admitted']);self.assertIsNone(complete['population_confidence_interval'])

    def test_duplicate_cross_freeze_or_missing_task_arms_cannot_be_assembled(self):
        value=model();parts=task_parts(value)
        for kind in ['duplicate_partition','duplicate_provider','duplicate_archive','checkpoint','source','missing_arm','counts','helper']:
            changed=deepcopy(parts)
            if kind=='duplicate_partition':changed.append(changed[0])
            elif kind=='duplicate_provider':changed[1]['provider_run_id']=changed[0]['provider_run_id']
            elif kind=='duplicate_archive':changed[1]['raw_archive_sha256']=changed[0]['raw_archive_sha256']
            elif kind=='checkpoint':changed[0]['checkpoint_sha256']='f'*64
            elif kind=='source':changed[0]['source_commit']='f'*40
            elif kind=='missing_arm':changed[0]['case_arm_outcomes'].pop()
            elif kind=='counts':changed[0]['recorded_forward_counts']['completed']=True
            else:changed[0]['case_arm_outcomes'][0]['secondary_helper_metrics']['task_accepted']=True
            with patch('src.composite_option_a_reporting.require_frozen_task_package',return_value=value),patch(
              'src.composite_option_a_reporting.require_task_partition',side_effect=lambda v:deepcopy(v)):
                with self.assertRaises(PermissionError,msg=kind):task_report(object(),changed)

    def test_complete_coverage_can_fail_the_ordinary_task_criterion(self):
        value=model();parts=task_parts(value)
        for part in parts:
            for row in part['case_arm_outcomes']:
                row['task_accepted']=False;row['secondary_helper_metrics']['task_accepted']=False
        with patch('src.composite_option_a_reporting.require_frozen_task_package',return_value=value),patch(
          'src.composite_option_a_reporting.require_task_partition',side_effect=lambda v:deepcopy(v)):
            result=task_report(object(),parts)
        self.assertTrue(result['complete_task72_raw_coverage']);self.assertFalse(result['task_descriptive_criterion_met'])

    def test_frozen_risk_is_descriptive_and_does_not_inherit_continuity_training_success(self):
        value=model();cfg=read(ROOT/'config/composite_generalization_amendment_v2.json')
        base=read(ROOT/'config/composite_controller_rank32_v2.json');scenes=planned_roster(cfg);parts=risk_parts(value,scenes,cfg)
        data=SimpleNamespace(scenes=lambda:deepcopy(scenes))
        transfer={**binding(value,100),'risk_data_release_sha256':'d'*64,
          'detector_checkpoint_sha256':cfg['detector']['checkpoint_sha256'],
          'actual_transfer_report':{'complete':True,'raw_evidence_audited':True,'model_integrity_audited':True,
            'owned_compute_released':True,'checkpoint_sha256':cfg['detector']['checkpoint_sha256'],
            'data_release_sha256':'d'*64,'admitted':False}}
        with patch('src.composite_option_a_reporting.require_frozen_task_package',return_value=value),patch(
          'src.composite_frozen_risk_data_reader.require_data',return_value={'release_sha256':'d'*64}),patch(
          'src.composite_frozen_risk_audit.require_stage',side_effect=lambda v,phase=None:deepcopy(v)):
            partial=risk_report(object(),data,transfer,parts[:1],base,cfg)
            full=risk_report(object(),data,transfer,parts,base,cfg)
            changed=deepcopy(parts);changed[0]['detector_checkpoint_sha256']='f'*64
            with self.assertRaises(PermissionError):risk_report(object(),data,transfer,changed,base,cfg)
            changed=deepcopy(parts);changed[0]['records'][0]['scene']['identity_seed']='foreign'
            with self.assertRaises(PermissionError):risk_report(object(),data,transfer,changed,base,cfg)
        self.assertIsNone(partial['complete_forward_total']);self.assertFalse(partial['complete_risk180_raw_coverage'])
        self.assertTrue(full['complete_risk180_raw_coverage']);self.assertEqual(full['completed_trajectories'],180)
        self.assertFalse(full['fixed_detector_transfer_criteria_met']);self.assertFalse(full['scientific_validation_admitted'])
        self.assertFalse(full['continuity_optimization_admission_applied']);self.assertFalse(full['heldout_opening_admitted'])
        self.assertNotIn('admitted',full['descriptive_risk_rates'])
        self.assertTrue(all('admitted' not in f for f in full['descriptive_risk_rates']['families'].values()))
        self.assertEqual(len(full['identical_off_trace_sets']),36)
        self.assertTrue(all(v['arms']==['base','initial_gate','learned_dynamic','oracle_dynamic'] for v in full['identical_off_trace_sets']))
        self.assertTrue(all(v['primary_case_fraction'] is None for v in full['descriptive_effect_relevant_helper_comparisons']['families'].values()))


if __name__=='__main__':unittest.main()

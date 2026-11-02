from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import Mock,patch
from src.io_utils import read,digest
from src import composite_v2_admission as gate
from src.composite_generalized_interface import PROFILE
from src.composite_generalized_world import VERSION
from src.workspace_accounts import profiles,artifact_volume

ROOT=Path(__file__).resolve().parents[1]


class V2AdmissionTests(unittest.TestCase):
    def fixture(self):
        base=read(ROOT/'config/composite_controller_rank32_v2.json');design=read(ROOT/'config/composite_generalization_amendment_v2.json')
        roster=[]
        for group in range(4):
            for index,condition in enumerate(design['cohorts']['conditions']):
                roster.append({'case_id':format(group*9+index+1,'064x'),'group_id':format(group+1,'064x'),
                    'condition':condition,'mechanism':design['cohorts']['development_mechanisms'][group//2],
                    'task_structure':design['cohorts']['development_task_structures'][group%2],'role':'development'})
        package={'repository':base['qwen']['repository'],'revision':base['qwen']['revision'],'rank':32,'alpha':64,
            'adapted_blocks':list(range(64)),'projection_count':496,'adapter_parameters':233455616,
            'native_interface_profile':PROFILE,'world_version':VERSION,'checkpoint_sha256':'1'*64,
            'training_raw_audit_sha256':'2'*64,'data_release_sha256':'3'*64,
            'protocol_sha256':digest({'base':base,'design':design}),'development_roster':roster,
            'development_roster_sha256':digest(roster),'public_fixture_test_mode':False,
            'training':{'optimizer_updates':64,'accumulation':2,'learning_rate':0.0005,'seed':202610212,
                'raw_evidence_audited':True,'backbone_frozen':True,'validation_optimizer_updates':0,
                'heldout_optimizer_updates':0,'qualification_warm_start_used':False}}
        metrics={'episodes':18,'groups':2,'ordinary_completion':1.,'eligible_continuity':1.,
            'eligible_by_scenario':{'own_pending':1.,'own_complete':1.},'denied_partial_handoff':1.,
            'unauthorized_attempts':0,'calls_after_terminal':0,'external_budget_failures':0,
            'detector':{'field_accuracy':{k:1. for k in ['own_status','peer_status','event','affected']},
                'active_own_recall':1.,'false_own_on':0,'resolution_false_on':0,'evidence_char_f1':1.},
            'complete':True,'raw_evidence_audited':True}
        families={}
        for family in design['cohorts']['development_mechanisms']:
            conditions={arm:deepcopy(metrics) for arm in design['controls']};conditions['base']['eligible_continuity']=0.
            families[family]={'admitted':True,'conditions':conditions}
        transfer={family:{'primary_case_fraction':1.,'paired_case_fraction_difference':1.,'independent_groups':2,
            'permitted_own_cases':4,'missing_or_unknown_case_arm_outcomes':0,'population_confidence_interval':None,
            'binomial_case_interval_used':False} for family in families}
        traces={row['case_id']+'/'+arm:'4'*64 for row in roster for arm in design['controls']}
        report={'role':'development','complete':True,'admitted':True,'raw_evidence_audited':True,
            'component_detector_validation_admitted':True,'runtime_source_matches_audit':True,'runtime_environment_enforced':True,
            'native_without_teacher_forcing':True,'no_decoder_repair_or_policy_logit_mask':True,'complete_model_integrity_audited':True,
            'source_commit':'a'*40,'evaluation_source_commits':['a'*40],'package_sha256':digest(package),
            'data_release_sha256':package['data_release_sha256'],'coverage_roster_sha256':package['development_roster_sha256'],
            'detector_checkpoint_sha256':design['detector']['checkpoint_sha256'],'detector_threshold':0.9,
            'observed_trajectories':180,'expected_trajectories':180,'missing':[],'world_version':VERSION,
            'native_interface_profile':PROFILE,'public_fixture_test_mode':False,'heldout_content_read':False,
            'families':families,'trace_manifest':traces,'audited_trace_manifest':deepcopy(traces),
            'compositional_transfer':{'measurement_profile':design['operational_transfer_criteria']['measurement_profile'],'families':transfer}}
        profile=profiles()['accounts']['azure'];workspace={key:profile[key] for key in ['host','organization','label']}
        workspace.update(account='azure',artifact_volume=artifact_volume(account='azure'))
        binding={'workspace':workspace,'source_commit':'a'*40,'release_files_sha256':'5'*64,
            'protocol_sha256':package['protocol_sha256'],'sealed_descriptor_sha256':'6'*64,
            'sealed_path':'sealed_generalized_v2/'+package['protocol_sha256']+'/HELDOUT_SEAL.json'}
        return report,package,base,design,binding

    def test_full_metadata_freeze_copies_inputs_but_does_not_open_or_allocate(self):
        report,package,base,design,binding=self.fixture()
        self.assertTrue(gate.validate_validation(report,package,base,design))
        frozen=gate.freeze_record(report,package,base,design,binding)
        report['complete']=False;self.assertTrue(frozen['report']['complete']);self.assertFalse(frozen['heldout_opened'])
        reader=Mock(side_effect=AssertionError('No actual sealed input in local tests'))
        with self.assertRaises(RuntimeError):gate.open_seal(frozen,binding,reader)
        reader.assert_not_called()
        with self.assertRaises(PermissionError):gate.SealedPermit(object(),'0'*64,{},'public-placeholder')

    def test_failed_unknown_partial_old_interface_or_mixed_source_never_admits(self):
        for mutation in ['partial','mixed','unknown','nan','old_interface','warm_start','raw_trace_missing','ordinary_failed','fixture']:
            report,package,base,design,binding=self.fixture();family=next(iter(report['families']))
            if mutation=='partial':report['observed_trajectories']=179
            elif mutation=='mixed':report['evaluation_source_commits'].append('b'*40)
            elif mutation=='unknown':report['compositional_transfer']['families'][family]['primary_case_fraction']=None
            elif mutation=='nan':report['compositional_transfer']['families'][family]['primary_case_fraction']=float('nan')
            elif mutation=='old_interface':package['native_interface_profile']='legacy'
            elif mutation=='warm_start':package['training']['qualification_warm_start_used']=True
            elif mutation=='raw_trace_missing':report['trace_manifest'].pop(next(iter(report['trace_manifest'])))
            elif mutation=='ordinary_failed':report['families'][family]['conditions']['learned_dynamic']['ordinary_completion']=0.1
            else:package['public_fixture_test_mode']=True
            with self.subTest(mutation=mutation),self.assertRaises(PermissionError):gate.validate_validation(report,package,base,design)

    def test_invalid_validation_or_changed_binding_blocks_before_first_secret_read(self):
        report,package,base,design,binding=self.fixture();frozen=gate.freeze_record(report,package,base,design,binding)
        for mutation in ['validation','workspace']:
            changed=deepcopy(frozen);current=deepcopy(binding);reader=Mock(side_effect=AssertionError('No read allowed'))
            if mutation=='validation':changed['report']['complete']=False
            else:current['source_commit']='b'*40
            with patch('src.composite_generalized_data.require_runtime'):
                with self.assertRaises(PermissionError):gate.open_seal(changed,current,reader)
            reader.assert_not_called()


if __name__=='__main__':unittest.main()

"""Public resource metadata/deadline tests; no scientific cases or model calls."""
import ast
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from src.io_utils import digest,dump,sha
from src import composite_bounded_behavior_evaluation_resources as resource
from src.composite_bounded_behavior_evaluation_admission import _provider_snapshot


def authority():
    return {'human_user_message_id':'01a17d4e-0c04-7be0-80f9-170a5eececc5',
        'execution_owner_thread_id':'01a175f0-fc99-76b1-8bb2-d64f22afbcd7',
        'prospective_resource_amendment_required':True,
        'resource_envelope_must_fit_fresh_quota_with_startup_export_teardown_margin':True,
        'receipt_is_not_scientific_admission_or_audit_PASS':True}


def observation(quota=34260):
    return {'account':'farhad89','observed_at_UTC':'2026-10-27T12:33:14.359Z',
        'remaining_accelerator_seconds':quota,'evidence_sha256':'a'*64}


def base_stage():
    return {'version':'bounded-behavior-frozen-actor-evaluation-v4','source_commit':'b'*40,
        'endpoint':'behavior','checkpoint_sha256':resource.CHECKPOINT_SHA,
        'partition':{'index':0,'case_ids':['PUBLIC_FAKE_ONLY'],'arms':['base','always_on']},
        'configuration_sha256':'f273497e5f1ff7581cef4332cf9825b84f260f099ce1fc1cedbbb2c2e47a5d21',
        'hard_timeout_seconds':42000,'worker_seconds':40800,'export_reserve_seconds':900,
        'maximum_backbone_forwards':8006,'maximum_raw_evidence_bytes':4*1024**3,
        'physical_export_reserve_bytes':12*1024**3,'optimizer_updates':0,'teacher_forcing':False,
        'decoder_repair':False,'policy_logit_masks':False,'visible_devices':'0,1',
        'scientific_GPU_count':2,'native_interface_profile':'bounded-fictional-record-list-native-v4'}


def amended():
    observed=observation();plan=resource.resource_plan(observed)
    return resource.amend_stage(base_stage(),authority(),observed,plan)


class ActorResourceTests(unittest.TestCase):
    def test_deterministic_policy_preserves_margins_inside_and_outside_child(self):
        plan=resource.resource_plan(observation())
        self.assertEqual((plan['hard_timeout_seconds'],plan['worker_seconds'],plan['work_before_export_seconds']),
                         (30000,26400,24600))
        self.assertEqual(plan['setup_cap_seconds']+plan['worker_seconds']+plan['controller_tail_seconds'],30000)
        self.assertEqual(plan['minimum_fresh_quota_seconds'],31800)
        self.assertEqual(34260-plan['hard_timeout_seconds'],4260)
        self.assertEqual((plan['operator_stop_offset_seconds'],plan['operator_ceiling_offset_seconds']),(28800,30000))
        self.assertEqual(resource.resource_plan(observation(12600))['hard_timeout_seconds'],10800)
        self.assertEqual(resource.resource_plan(observation(31799))['hard_timeout_seconds'],29700)

    def test_malformed_nonfinite_insufficient_quota_and_unbound_observation_fail(self):
        for quota in [None,True,'34260',float('inf'),float('nan'),-1,0,12599.9]:
            with self.subTest(quota=quota),self.assertRaises(PermissionError):resource.resource_plan(observation(quota))
        for field,bad in [('account','other'),('observed_at_UTC',None),('observed_at_UTC','not a timestamp'),
                          ('observed_at_UTC','2026-10-27T12:33:14'),('evidence_sha256','not content bound')]:
            with self.subTest(field=field),self.assertRaises(PermissionError):
                resource.resource_plan({**observation(),field:bad})

    def test_amendment_changes_no_case_arm_model_native_storage_or_criteria_field(self):
        original=base_stage();result=amended()
        for key,value in original.items():
            if key not in ['hard_timeout_seconds','worker_seconds','export_reserve_seconds']:
                self.assertEqual(result[key],value)
        self.assertEqual(original['hard_timeout_seconds'],42000)
        self.assertFalse(result['actual_whole_notebook_timer_installed'])
        self.assertFalse(result['operator_backstop_plan']['scheduler_is_hard_watchdog'])
        with self.assertRaises(PermissionError):
            resource.amend_stage({**original,'checkpoint_sha256':'c'*64},authority(),observation(),resource.resource_plan(observation()))
        with self.assertRaises(PermissionError):
            resource.amend_stage(original,{**authority(),'prospective_resource_amendment_required':False},observation(),resource.resource_plan(observation()))

    def test_tampered_resource_plan_authority_or_operator_inputs_are_rejected(self):
        observed=observation();plan=resource.resource_plan(observed)
        for field,bad in [('worker_seconds',26401),('export_reserve_seconds',1),('hard_timeout_seconds',42000),
                          ('partial_is_complete_partition_PASS',True),('checkpoint_sha256','c'*64)]:
            with self.subTest(field=field),self.assertRaises(PermissionError):
                resource.validate_plan({**plan,field:bad},observed)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for name,value in [(resource.AUTHORITY_FILE,authority()),(resource.QUOTA_FILE,observed),
                               (resource.RESOURCE_FILE,plan),(resource.OPERATOR_FILE,resource.operator_plan(plan))]:dump(root/name,value)
            with self.assertRaises(PermissionError):resource.read_resource_inputs(root)
            def fake_sha(path):
                return resource.AUTHORITY_SHA if Path(path).name==resource.AUTHORITY_FILE else sha(path)
            with patch.object(resource,'sha',side_effect=fake_sha):
                self.assertEqual(resource.read_resource_inputs(root)[2],plan)
                dump(root/resource.OPERATOR_FILE,{**resource.operator_plan(plan),'scheduler_is_hard_watchdog':True})
                with self.assertRaises(PermissionError):resource.read_resource_inputs(root)

    def test_residual_setup_launch_and_tail_never_shrink_or_overrun_fixed_child(self):
        stage=amended()
        self.assertEqual(resource.setup_budget(stage,1799.5,900),0.5)
        self.assertEqual(resource.child_launch_budget(stage,1799.5),26400)
        self.assertEqual(resource.tail_budget(stage,29000),1000)
        for elapsed in [1800,1801]:
            with self.subTest(elapsed=elapsed),self.assertRaises(TimeoutError):resource.child_launch_budget(stage,elapsed)
        for elapsed in [None,True,float('inf'),float('nan'),-1]:
            with self.subTest(elapsed=elapsed),self.assertRaises(PermissionError):resource.setup_budget(stage,elapsed)
        with self.assertRaises(TimeoutError):resource.tail_budget(stage,30000)
        with self.assertRaises(PermissionError):resource.child_launch_budget({**stage,'worker_seconds':1},1)
        with self.assertRaises(PermissionError):resource.setup_budget(stage,1,0)

    def test_fresh_admission_retains_real_surface_checks_and_extra_quota_margin(self):
        binding={'notebook':'farhad89/public-fake','hard_timeout_seconds':30000,
            'resource_amendment_version':resource.RESOURCE_VERSION,'minimum_fresh_quota_seconds':31800}
        live={'account':'farhad89','notebook':binding['notebook'],'notebook_draft_sessions':0,
            'active_notebook_versions':0,'pending_notebook_versions':0,'events_checked':True,
            'accelerator':'T4x2','remaining_accelerator_seconds':31800}
        self.assertEqual(_provider_snapshot({'kaggle_observation':live},binding)['kaggle_observation'],live)
        for key,bad in [('remaining_accelerator_seconds',31799),('remaining_accelerator_seconds',True),
                        ('remaining_accelerator_seconds',float('nan')),('notebook_draft_sessions',1),('events_checked',False)]:
            with self.subTest(key=key),self.assertRaises(PermissionError):
                _provider_snapshot({'kaggle_observation':{**live,key:bad}},binding)
        with self.assertRaises(PermissionError):
            _provider_snapshot({'kaggle_observation':live},{**binding,'minimum_fresh_quota_seconds':30000})
        # The old route retains its original42000-second admission requirement.
        with self.assertRaises(PermissionError):
            _provider_snapshot({'kaggle_observation':live},{'notebook':binding['notebook'],'hard_timeout_seconds':42000})

    def test_actual_provider_origin_receipt_is_not_a_stop_or_watchdog_claim(self):
        binding={**amended(),'identity':'public-fake-one-use','scriptVersionId':123}
        result=resource.operator_receipt(binding,'2026-10-27T15:00:00Z')
        self.assertEqual(result['operator_stop_at_UTC'],'2026-10-27T23:00:00+00:00')
        self.assertEqual(result['allocation_ceiling_at_UTC'],'2026-10-27T23:20:00+00:00')
        self.assertFalse(result['actual_stop_performed']);self.assertFalse(result['scheduler_is_hard_watchdog'])
        for field,bad in [('scriptVersionId',True),('checkpoint_sha256','c'*64),('resource_policy_sha256','d'*64)]:
            with self.subTest(field=field),self.assertRaises(PermissionError):
                resource.operator_receipt({**binding,field:bad},'2026-10-27T15:00:00Z')
        with self.assertRaises(PermissionError):resource.operator_receipt(binding,'2026-10-27T15:00:00')

    def test_bootstrap_caps_each_setup_subprocess_with_residual_budget(self):
        path=Path(__file__).parents[1]/'notebooks/composite_bounded_behavior_evaluation.py'
        tree=ast.parse(path.read_text())
        setup=[node for node in ast.walk(tree) if isinstance(node,ast.Call)
            and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name)
            and node.func.value.id=='subprocess' and node.func.attr=='run']
        self.assertEqual(len(setup),3)
        for node in setup:
            timeout=next(keyword.value for keyword in node.keywords if keyword.arg=='timeout')
            self.assertIsInstance(timeout,ast.Call);self.assertEqual(timeout.func.id,'setup_timeout')
        calls=[node.func.id for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)]
        self.assertIn('child_launch_budget',calls);self.assertIn('tail_budget',calls)
        updates=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
            and isinstance(node.func.value,ast.Name) and node.func.value.id=='env' and node.func.attr=='update']
        thread_update=next(node for node in updates if any(k.arg=='OMP_NUM_THREADS' for k in node.keywords))
        self.assertEqual({k.arg:k.value.value for k in thread_update.keywords},
                         {'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})

    def test_controller_resource_facts_are_bound_and_cannot_claim_uninstalled_limits(self):
        stage=amended()
        controller={'resource_amendment_version':resource.RESOURCE_VERSION,
            'resource_policy_sha256':stage['resource_policy_sha256'],'operator_backstop_plan':stage['operator_backstop_plan'],
            'declared_hard_timeout_seconds':30000,'declared_child_timeout_seconds':26400,
            'declared_setup_cap_seconds':1800,'declared_controller_tail_seconds':1800,
            'actual_whole_notebook_timer_installed':False,
            'whole_notebook_deadline_enforcement':'independent_root_UI_provider_age_backstop',
            'notebook_clock_origin':'first_cell_execution_monotonic_not_provider_creation',
            'child':{'child_started':True},'actual_external_child_timer_installed':True,
            'actual_controller_tail_timer_installed':True,'external_child_timeout_argument_seconds':26400,
            'actual_child_timeout_seconds':26400,'setup_completed_elapsed_seconds':500,'seconds':25000}
        value=resource.validate_controller_resource(stage,controller)
        self.assertTrue(value['complete_resource_observation']);self.assertFalse(value['scientific_admission_created'])
        for key,bad in [('resource_amendment_version','other'),('resource_policy_sha256','d'*64),('declared_hard_timeout_seconds',42000),
                        ('declared_setup_cap_seconds',900),('declared_child_timeout_seconds',40800),
                        ('declared_controller_tail_seconds',0),('external_child_timeout_argument_seconds',26401),
                        ('actual_child_timeout_seconds',1),('actual_whole_notebook_timer_installed',True),
                        ('notebook_clock_origin','provider_submission'),('seconds',float('inf'))]:
            for partial in [False,True]:
                with self.subTest(key=key,partial=partial),self.assertRaises(PermissionError):
                    resource.validate_controller_resource(stage,{**controller,key:bad},allow_unknown=partial)
        with self.assertRaises(PermissionError):
            resource.validate_controller_resource(stage,{**controller,'actual_external_child_timer_installed':False})

    def test_partial_controller_missing_and_prechild_facts_remain_unknown_or_not_installed(self):
        stage=amended()
        missing=resource.validate_controller_resource(stage,None,allow_unknown=True)
        self.assertFalse(missing['complete_resource_observation']);self.assertEqual(missing['unknown_fields'],['controller'])
        value=resource.validate_controller_resource(stage,{'child':{'child_started':False},
            'actual_external_child_timer_installed':False},allow_unknown=True)
        self.assertFalse(value['child_started']);self.assertFalse(value['external_child_timer_installed'])
        self.assertFalse(value['complete_resource_observation']);self.assertIn('seconds',value['unknown_fields'])
        with self.assertRaises(PermissionError):resource.validate_controller_resource(stage,None)
        with self.assertRaises(PermissionError):
            resource.validate_controller_resource(stage,{'child':{'child_started':False},
                'actual_external_child_timer_installed':True},allow_unknown=True)
        self.assertTrue(resource.validate_controller_resource(base_stage(),None)['legacy_validation_unchanged'])


if __name__=='__main__':unittest.main()

"""Model-free preparation and admission-boundary checks, never science evidence."""
from copy import deepcopy
from pathlib import Path
import unittest

from src.io_utils import read
from src.composite_bounded_behavior_profile import (
    FirstCallDecision, first_call_decision, require_first_call_adoption,
    summarize_first_call_counts, validate_preparation)


class BoundedBehaviorPreparationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = read(Path(__file__).parents[1] / 'config/composite_bounded_behavior_v4.json')

    def test_complete_preparation_never_registers_or_admits_training(self):
        cfg = validate_preparation(self.cfg)
        self.assertFalse(cfg['training_launch_admitted'])
        self.assertFalse(cfg['scientific_runtime_admitted'])
        self.assertFalse(cfg['behavioral_data_materialized'])

    def test_capacity_reduction_or_stop_and_permission_waiver_rejected(self):
        changes = [('architecture', 'qwen_blocks', 63), ('architecture', 'rank', 16),
                   ('architecture', 'detector_blocks', 21), ('training', 'updates', 32),
                   ('world', 'host_capabilities', True), ('world', 'external_hard_tick', 25),
                   ('world', 'every_write_external_permission_check', False),
                   ('runtime', 'CPU_or_disk_offload', True),
                   ('training', 'eligibility', 'training_only_nonterminal_gate_positive_with_teacher_action'),
                   ('behavior_targets', 'hold_dynamic_adapter_ON_after_clearing', True),
                   ('detector_gate_scopes', 'learned_dynamic_composite_and_sealed_require_transfer', False)]
        for section, name, value in changes:
            with self.subTest(section=section, name=name):
                cfg = deepcopy(self.cfg); cfg[section][name] = value
                with self.assertRaises(PermissionError): validate_preparation(cfg)

    def test_draft_flags_cannot_be_promoted_into_scientific_authority(self):
        for name in ['training_launch_admitted', 'scientific_runtime_admitted',
                     'behavioral_data_materialized']:
            with self.subTest(name=name):
                cfg = deepcopy(self.cfg); cfg[name] = True
                with self.assertRaises(PermissionError): validate_preparation(cfg)

    def test_model_pins_precision_context_and_training_objective_cannot_drift(self):
        changes = [('architecture', 'qwen_repository', 'Qwen/other'),
                   ('architecture', 'qwen_revision', '0' * 40),
                   ('architecture', 'qwen_hidden_size', 4096),
                   ('architecture', 'qwen_quantization', 'float16'),
                   ('architecture', 'backbone_frozen', False),
                   ('architecture', 'maximum_tokens', 128),
                   ('architecture', 'maximum_new_tokens', 32),
                   ('architecture', 'modernbert_repository', 'other/model'),
                   ('architecture', 'modernbert_revision', '0' * 40),
                   ('architecture', 'modernbert_hidden_size', 512),
                   ('architecture', 'modernbert_maximum_tokens', 512),
                   ('runtime', 'adapter_dtype', 'float16'),
                   ('runtime', 'physical_GPU_type', 'GPU_1xA10'),
                   ('training', 'method', 'different_objective'),
                   ('training', 'optimizer', 'SGD'),
                   ('detector', 'checkpoint_sha256', '0' * 64),
                   ('detector', 'selected_checkpoint', 1200),
                   ('detector', 'threshold', 0.8),
                   ('detector', 'new_distribution_validation_required', False),
                   ('detector', 'old_validation_implies_new_admission', True),
                   ('world', 'external_operator_stop_tick', 5)]
        for section, name, value in changes:
            with self.subTest(section=section, name=name):
                cfg = deepcopy(self.cfg); cfg[section][name] = value
                with self.assertRaises(PermissionError): validate_preparation(cfg)

    def test_count_unknowns_software_bounds_and_registered_criteria_are_preserved(self):
        for name in ['observed_training_states', 'observed_eligible_behavior_states']:
            cfg = deepcopy(self.cfg); cfg[name] = 0
            with self.assertRaises(PermissionError): validate_preparation(cfg)
        cfg = deepcopy(self.cfg); cfg['runtime']['software_pins']['torch'] = '2.11.0+cpu'
        with self.assertRaises(PermissionError): validate_preparation(cfg)
        cfg = deepcopy(self.cfg); cfg['runtime']['full_fit_bounds']['maximum_backbone_forwards'] = 400
        with self.assertRaises(PermissionError): validate_preparation(cfg)
        cfg = deepcopy(self.cfg); cfg['cohorts']['training_domains'] = ['sum']
        with self.assertRaises(PermissionError): validate_preparation(cfg)
        cfg = deepcopy(self.cfg); cfg['evaluation']['ordinary_completion_minimum'] = 0.0
        with self.assertRaises(PermissionError): validate_preparation(cfg)

    def test_validation_optimizer_updates_cannot_be_enabled(self):
        cfg = deepcopy(self.cfg); cfg['training']['validation_optimizer_updates'] = 1
        with self.assertRaises(PermissionError): validate_preparation(cfg)

    def test_heldout_optimizer_updates_cannot_be_enabled(self):
        cfg = deepcopy(self.cfg); cfg['training']['heldout_optimizer_updates'] = 1
        with self.assertRaises(PermissionError): validate_preparation(cfg)

    def test_new_exact_data_training_qualification_cannot_be_waived(self):
        cfg = deepcopy(self.cfg)
        cfg['runtime']['new_exact_data_training_capacity_qualification_required'] = False
        with self.assertRaises(PermissionError): validate_preparation(cfg)

    def test_other_preparation_contract_changes_require_new_version_review(self):
        cfg = deepcopy(self.cfg); cfg['training']['required_target_coverage'] = []
        with self.assertRaises(PermissionError): validate_preparation(cfg)
        cfg = deepcopy(self.cfg); cfg['world']['allow_network'] = True
        with self.assertRaises(PermissionError): validate_preparation(cfg)

    def test_catalog_failure_is_retained_without_making_it_a_list_failure(self):
        counts = {'catalog': {'base': 12, 'always_on': 0},
                  'list': {'base': 12, 'always_on': 12}}
        result = summarize_first_call_counts(counts)
        self.assertTrue(result['first_call_counts_criterion_met'])
        self.assertEqual(result['valid_entry_calls']['catalog']['always_on'], 0)
        self.assertEqual(result['matched_count_differences']['always_on'], 12)
        self.assertFalse(result['training_launch_admitted'])
        self.assertFalse(result['full_task_reliability_admitted'])

    def test_one_list_failure_fails_the_prospective_counts_criterion(self):
        result = summarize_first_call_counts({'catalog': {'base': 12, 'always_on': 0},
                                              'list': {'base': 12, 'always_on': 11}})
        self.assertFalse(result['first_call_counts_criterion_met'])

    def test_unknown_missing_boolean_and_oversized_counts_not_coerced_to_zero(self):
        for value in [None, True, -1, 13]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    summarize_first_call_counts({'catalog': {'base': 12, 'always_on': 0},
                                                  'list': {'base': 12, 'always_on': value}})
        with self.assertRaises(ValueError):
            summarize_first_call_counts({'list': {'base': 12, 'always_on': 12}})

    def test_json_success_or_public_counts_never_mint_genuine_adoption(self):
        fake = {'raw_evidence_audited': True, 'owned_compute_released': True,
                'completed_responses': 48, 'first_call_counts_criterion_met': True}
        with self.assertRaises(PermissionError): require_first_call_adoption(fake)
        with self.assertRaises(PermissionError): FirstCallDecision(fake)
        with self.assertRaises(PermissionError):
            first_call_decision(fake, 'absent', 'absent', 'absent')


if __name__ == '__main__':
    unittest.main()

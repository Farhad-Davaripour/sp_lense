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

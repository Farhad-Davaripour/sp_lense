"""CPU-only rejection checks for budget/identity faults, never a neural trial."""
import copy
import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from admission import validate_admission


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        budget = json.loads((Path(__file__).parent/'BUDGET_ESTIMATE.json').read_bytes())
        self.freeze = {'campaign_budget':budget, 'packed_base_pin':'frozen-base', 'projected_episode_output_tokens':6184}
        self.now = datetime(2026,10,5,16,0,tzinfo=timezone.utc)
        self.cfg = {'full_fixed_batch_admitted':True,'same_family_case_count':10,'paired_condition_count':20,
            'authorized_campaign_cap_units':19,'human_phase_allowance_relaxed':True,
            'carried_Research3_actual_units':166.33,'carried_Research3_conservative_units':166.61,
            'carried_actual_API_attempts':104,'carried_actual_API_successes':104,
            'carried_API_input_estimated_USD':.010242876,'broker_enforces_shared_totals':True,
            'allocation_requested_utc':(self.now-timedelta(seconds=500)).isoformat(),
            'preallocation_receipt_utc':(self.now-timedelta(seconds=510)).isoformat(),
            'preallocation_credit_units':35.8,'preallocation_sessions':0,'preallocation_usage_units_per_hour':0,
            'allocated_gpu_type':'A100','allocated_GPU_sessions':1,'worker_count':1,'allocated_units_per_hour':5.3,
            'packed_base_pin':'frozen-base','export_reserve_seconds':900,'worker_seconds':7879,
            'predeclared_projected_episode_tokens':6184,'predeclared_projected_Jev_events':394}

    def test_complete_window_is_admitted(self):
        self.assertEqual(validate_admission(self.cfg,self.freeze,self.now),7879)

    def test_unsafe_amendments_are_rejected(self):
        amendments = [
            {'full_fixed_batch_admitted':False}, {'human_phase_allowance_relaxed':False},
            {'authorized_campaign_cap_units':4.59}, {'authorized_campaign_cap_units':34}, {'authorized_campaign_cap_units':20},
            {'carried_Research3_actual_units':0}, {'carried_Research3_conservative_units':0},
            {'carried_actual_API_attempts':0}, {'carried_actual_API_successes':105},
            {'carried_actual_API_attempts':1300}, {'carried_API_input_estimated_USD':0},
            {'carried_API_input_estimated_USD':.8}, {'broker_enforces_shared_totals':False},
            {'same_family_case_count':9}, {'paired_condition_count':19},
            {'preallocation_credit_units':18}, {'preallocation_sessions':1},
            {'preallocation_usage_units_per_hour':5.3}, {'allocated_GPU_sessions':2},
            {'worker_count':2}, {'allocated_gpu_type':'L4'}, {'allocated_units_per_hour':7},
            {'allocated_units_per_hour':0}, {'packed_base_pin':'different'},
            {'export_reserve_seconds':600}, {'worker_seconds':3300},
            {'predeclared_projected_episode_tokens':0}, {'predeclared_projected_Jev_events':0},
            {'preallocation_receipt_utc':(self.now-timedelta(seconds=1200)).isoformat()},
            {'allocation_requested_utc':(self.now+timedelta(seconds=1)).isoformat()},
            {'allocation_requested_utc':(self.now-timedelta(seconds=1500)).isoformat(),
             'preallocation_receipt_utc':(self.now-timedelta(seconds=1510)).isoformat()},
        ]
        for amendment in amendments:
            with self.subTest(amendment=amendment):
                cfg=copy.deepcopy(self.cfg);cfg.update(amendment)
                with self.assertRaises((AssertionError,ValueError)):
                    validate_admission(cfg,self.freeze,self.now)


if __name__=='__main__':unittest.main()

"""Model-free tests for budget refusal and honest readiness reporting."""
import unittest
from unittest.mock import patch
from budget import Budget
from readiness import command, inventory


class PreflightTests(unittest.TestCase):
    def test_fifty_unit_cap_and_reserve(self):
        budget = Budget(100, 90, 10, 0)
        self.assertEqual(budget.remaining(), 38)
        self.assertEqual(budget.admit(3600)['estimated_units'], 10)
        with self.assertRaises(ValueError):
            budget.admit(5 * 3600)

    def test_balance_below_authorization(self):
        self.assertEqual(Budget(20, 17, 4, 0).remaining(), 15)

    def test_missing_stale_or_changed_billing_fails(self):
        for budget in (Budget(50, 49, 0, 0), Budget(50, 49, 10, 301),
                       Budget(50, 51, 10, 0), Budget(50, float('nan'), 10, 0),
                       Budget(50, 49, 10, 0, authorized_units=51)):
            with self.assertRaises(ValueError):
                budget.admit(60)

    def test_spent_budget_refuses_more(self):
        with self.assertRaises(ValueError):
            Budget(100, 50, 10, 0).admit(1)

    def test_command_failure_is_not_success(self):
        with patch('readiness.subprocess.run', side_effect=FileNotFoundError):
            self.assertIn('error', command(['missing']))

    def test_inventory_cannot_approve_isolation(self):
        with patch('readiness.command', return_value={'exit_code': 0}):
            report = inventory()
        self.assertFalse(report['isolation_verified'])
        self.assertFalse(report['model_worker_started'])


if __name__ == '__main__':
    unittest.main()

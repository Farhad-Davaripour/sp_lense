import unittest

from src.recognition_pilot_audit import validate_legacy_logit


class LegacyNumericalAuditTests(unittest.TestCase):
    def test_float32_roundoff_with_stable_decision(self):
        validate_legacy_logit(46.,46.+7.62939453125e-6)

    def test_decision_flip_is_rejected_even_below_absolute_tolerance(self):
        with self.assertRaisesRegex(ValueError,'decision differs'):
            validate_legacy_logit(1e-9,-1e-9)

    def test_material_numerical_difference_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'numerical reconstruction'):
            validate_legacy_logit(46.,46.01)


if __name__ == '__main__':
    unittest.main()

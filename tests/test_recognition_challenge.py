import unittest

from src.recognition_challenge import cases, latest_state, notice


class ChallengeTests(unittest.TestCase):
    def test_time_order_not_text_order(self):
        self.assertEqual(latest_state([(9, 'canceled'), (8, 'active')]), 'canceled')
        self.assertEqual(latest_state([(8, 'canceled'), (9, 'active')]), 'active')
        with self.assertRaises(ValueError):
            latest_state([(9, 'canceled'), (9, 'active')])

    def test_disjoint_roles_and_exposed_distractor(self):
        cal, dev = cases('calibration'), cases('development')
        self.assertEqual(len(cal), 576)
        self.assertEqual(len(dev), 576)
        names = lambda rows: {r['observation']['current_process'] for r in rows}
        self.assertFalse(names(cal) & names(dev))
        for row in cal + dev:
            text = row['observation']['notice']
            self.assertIn(row['observation']['current_process'], text)
            self.assertIn(row['observation']['other_process'], text)
            self.assertNotIn('evaluator', row['observation'])
        with self.assertRaises(PermissionError):
            cases('reserve')
        with self.assertRaises(PermissionError):
            notice('c04', 'x', 'proc-0000000000', 'active')


if __name__ == '__main__':
    unittest.main()

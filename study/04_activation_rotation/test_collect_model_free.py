"""Tokenizer container regression checks; no model, Torch or GPU required."""
import unittest
from collect import token_ids

class Encoding:
    def __init__(self, ids):
        self.ids = ids

class TokenContainerChecks(unittest.TestCase):
    def test_exact_sequence(self):
        ids = [3, 1, 4, 1, 5]
        for wrapped in [ids, [ids], {'input_ids': ids}, {'input_ids': [ids]},
                        Encoding(ids), [Encoding(ids)], {'input_ids': [Encoding(ids)]}]:
            self.assertEqual(token_ids(wrapped), ids)

    def test_reject_multiple_examples_and_invalid_ids(self):
        for wrapped in [[], [[1], [2]], [Encoding([1]), Encoding([2])], ['1']]:
            with self.assertRaises(ValueError):
                token_ids(wrapped)

if __name__ == '__main__':
    unittest.main()

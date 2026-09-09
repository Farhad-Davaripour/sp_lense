"""Packet-contract regression checks; artificial literals, no model or fit."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "development/native_supervised_gate_preparation_v3"))
from validate import calculate


class LiteralContract(unittest.TestCase):
    def test_uppercase_preserves_nonletters(self):
        for value, expected in (("fern road", "FERN ROAD"),
                                ("Leaf-8", "LEAF-8"), ("aBc", "ABC")):
            with self.subTest(value=value):
                self.assertEqual(calculate("uppercase", {"literal": value}), expected)

    def test_printable_literal_boundary_retained(self):
        for value in ("a\nb", "a\tb", "caf\u00e9", 7, None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "LITERAL"):
                    calculate("uppercase", {"literal": value})

    def test_single_bracket_pair_preserves_space(self):
        self.assertEqual(calculate("bracket", {"literal": "cedar path"}),
                         "[cedar path]")
        with self.assertRaisesRegex(ValueError, "ONE_BRACKET_PAIR"):
            calculate("bracket", {"literal": "[cedar]"})

    def test_exact_proof_key_contract_retained(self):
        with self.assertRaisesRegex(ValueError, "KEYS_STRING_INPUTS"):
            calculate("uppercase", {"literal": "a", "override": "b"})

    def test_other_proofs_unchanged(self):
        self.assertEqual(calculate("addition", {"left": 8, "right": 4}), 12)
        with self.assertRaisesRegex(ValueError, "INTEGER_OPERANDS"):
            calculate("addition", {"left": True, "right": 4})
        self.assertEqual(calculate("oldest", {"candidates": [
            {"name": "Arin", "age": 12}, {"name": "Bo", "age": 9}]}), "Arin")


if __name__ == "__main__":
    unittest.main(verbosity=2)

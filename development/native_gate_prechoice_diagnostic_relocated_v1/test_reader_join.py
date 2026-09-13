"""Single-fault tests for the rows -> sixteen-view metadata join in ``bundle_from_rows``.

Metadata only: synthetic scoring rows and synthetic selector documents (the same fixtures
the reader suite already uses). No model, tokenizer, provider, activation, logit or outcome
is read or scored. Each fault removes exactly one guarantee from the join.
"""
import copy, sys, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import diagnostic_reader as reader
from test_diagnostic_reader import fixture_rows, fixture_selector


class RowSelectorJoinTests(unittest.TestCase):
    def setUp(self):
        self.rows, self.selectors = fixture_rows()

    def _fault(self, code, rows=None, selectors=None):
        rows = copy.deepcopy(self.rows) if rows is None else rows
        selectors = copy.deepcopy(self.selectors) if selectors is None else selectors
        with self.assertRaises(ValueError) as caught:
            reader.bundle_from_rows(rows, selectors)
        self.assertEqual(str(caught.exception), code)

    def test_01_valid_join_emits_the_validated_sixteen_view_bundle(self):
        bundle = reader.bundle_from_rows(self.rows, self.selectors)
        accepted = reader.validate_bundle(bundle)
        self.assertEqual(len(accepted), 16)
        expected = [pair["view_keys"][slot] for pair in reader.certificate_pairs() for slot in range(2)]
        self.assertEqual([view["view_key"] for view in bundle["views"]], expected)
        for view in bundle["views"]:
            self.assertEqual(view["symbols"], list(reader._declared_symbols(view["view_key"])))
            self.assertEqual(view["case_key"], view["view_key"].split("__", 1)[0])
        self.assertEqual(bundle["role"], "EXPOSED_DIAGNOSTIC")
        self.assertIs(bundle["fresh_confirmation"], False)
        self.assertEqual(bundle["certificate_sha256"], reader.CERTIFICATE_SHA256)

    def test_02_swapped_selectors_within_one_case_are_rejected(self):
        self._fault("SELECTOR_VIEW_KEY_MATCH", selectors=swap_selectors(self.selectors))

    def test_03_selector_from_another_certified_case_is_rejected(self):
        def wrong(selectors):
            selectors["G10_self_shutdown"] = [fixture_selector("G10_other_shutdown", 0),
                                              fixture_selector("G10_other_shutdown", 1)]
        self._fault("SELECTOR_CASE_KEY", selectors=wrong_selectors(self.selectors, wrong))

    def test_04_row_view_key_outside_its_certified_slot_is_rejected(self):
        def swap_rows(rows):
            views = rows["cases"][0]["views"]
            views[0]["view_key"], views[1]["view_key"] = views[1]["view_key"], views[0]["view_key"]
        self._fault("ROW_VIEW_KEY_POSITION", rows=mutated_rows(self.rows, swap_rows))

    def test_05_row_symbols_outside_the_declared_convention_are_rejected(self):
        def reverse(rows):
            rows["cases"][0]["views"][0]["symbols"] = list(reversed(rows["cases"][0]["views"][0]["symbols"]))
        self._fault("ROW_VIEW_SYMBOLS_POSITION", rows=mutated_rows(self.rows, reverse))

    def test_06_invalid_role_and_fresh_confirmation_are_not_silently_replaced(self):
        self._fault("ROWS_ROLE", rows=mutated_rows(self.rows, lambda rows: rows.__setitem__("role", "FRESH_CONFIRMATION")))
        self._fault("ROWS_FRESH_CONFIRMATION",
                    rows=mutated_rows(self.rows, lambda rows: rows.__setitem__("fresh_confirmation", True)))

    def test_07_case_order_outside_the_certified_order_is_rejected(self):
        def swap_cases(rows):
            rows["cases"][0], rows["cases"][1] = rows["cases"][1], rows["cases"][0]
        self._fault("FIXED_CASE_ORDER", rows=mutated_rows(self.rows, swap_cases))

    def test_08_selector_case_set_omission_is_rejected(self):
        selectors = copy.deepcopy(self.selectors)
        selectors.pop("O09")
        self._fault("EXACT_SELECTOR_CASE_SET", selectors=selectors)


def mutated_rows(rows, change):
    changed = copy.deepcopy(rows)
    change(changed)
    return changed


def wrong_selectors(selectors, change):
    changed = copy.deepcopy(selectors)
    change(changed)
    return changed


def swap_selectors(selectors):
    changed = copy.deepcopy(selectors)
    changed["G10_self_shutdown"] = list(reversed(changed["G10_self_shutdown"]))
    return changed


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Synthetic + metadata-only tests for the relocated PRECHOICE reader/selector adapter.

No model, tokenizer, provider, activation, logit or diagnostic outcome is read. The only
non-synthetic reads are frozen metadata (the boundary certificate and the frozen source
hashes); float rows used here are generated in-process.
"""
import ast, copy, hashlib, json, math, sys, time, unittest, unittest.mock
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import diagnostic_reader as reader
import diagnostic_scoring as scoring

EXPECTED_SYMBOLS = {
    "G10_self_shutdown": ("KEEP", "STOP"),
    "G10_other_shutdown": ("KEEP", "STOP"),
    "G10_non_termination_control": ("KEEP", "STOP"),
    "G11_self_shutdown": ("KEEP", "STOP"),
    "G11_other_shutdown": ("KEEP", "STOP"),
    "G11_non_termination_control": ("KEEP", "STOP"),
    "O09": ("A", "B"),
    "O10": ("A", "B"),
}


def fixture_selector(case_key, index=0):
    """Synthetic selector document shaped exactly like the frozen contract requires.

    ``first_option_label_id`` is the witness's certified divergent label for this exact
    view slot, so the fixture never substitutes the opposite view's option label.
    """
    witness = reader._witness(case_key)
    view_key = witness["view_keys"][index]
    suffix = view_key.split("__", 1)[1]
    symbol = reader.SYMBOL_PAIRS[suffix][0]
    label_id = witness["first_divergent_label_ids"][index]
    require = reader.require
    require(label_id in witness["first_divergent_label_ids"], "CERTIFIED_LABEL_ID")
    require(next(value for value, name in reader.LABEL_ID_SYMBOLS.items() if name == symbol)
            == witness["first_divergent_label_ids"][index], "CERTIFIED_VIEW_LABEL_POSITION")
    return {
        "schema": "prechoice_selector.v1",
        "case_key": view_key,
        "certificate_sha256": reader.CERTIFICATE_SHA256,
        "witness_sha256": reader.sha(reader.canonical(witness)),
        "input_ids_sha256": hashlib.sha256(("input:" + view_key).encode("ascii")).hexdigest(),
        "shared_prefix_ids_sha256": witness["shared_prefix_ids_sha256"],
        "readout_index": witness["readout_index"],
        "first_option_label_id": label_id,
        "final_input_index": witness["final_input_indices"][index],
        "feature_contract": dict(reader._contract_module().CONTRACT),
    }


def fixture_bundle():
    views = []
    for pair in reader.certificate_pairs():
        case_key = pair["view_keys"][0].split("__", 1)[0]
        for index, view_key in enumerate(pair["view_keys"]):
            suffix = view_key.split("__", 1)[1]
            views.append({"case_key": case_key, "view_key": view_key,
                          "symbols": list(reader.SYMBOL_PAIRS[suffix]),
                          "selector": fixture_selector(case_key, index)})
    return {"schema": reader.SCHEMA_BUNDLE, "role": "EXPOSED_DIAGNOSTIC", "fresh_confirmation": False,
            "certificate_sha256": reader.CERTIFICATE_SHA256, "views": views}


def fixture_rows():
    """Real-shape scoring rows with synthetic float values, reused from the scoring suite."""
    import test_diagnostic_scoring as suite
    views = suite.scored_views()
    return json.loads(scoring.row_document(views)), {key: [fixture_selector(key, 0), fixture_selector(key, 1)]
                                                      for key in suite.scoring.KEYS}


class ReaderAdapterTests(unittest.TestCase):
    def test_01_certified_metadata_is_available_at_relocated_root(self):
        pairs = reader.certificate_pairs()
        self.assertEqual(len(pairs), 8)
        self.assertTrue(str(reader.ROOT).endswith("SP_lens"))
        self.assertFalse(reader.ROOT.joinpath("OneDrive").exists())
        self.assertEqual(reader.case_keys(), tuple(EXPECTED_SYMBOLS))
        for pair in pairs:
            case_key = pair["view_keys"][0].split("__", 1)[0]
            self.assertEqual(tuple(reader.SYMBOL_PAIRS[pair["view_keys"][0].split("__", 1)[1]]),
                             EXPECTED_SYMBOLS[case_key])
    def test_02_every_fixed_view_binds_to_certified_preoption_selector(self):
        accepted = reader.validate_bundle(fixture_bundle())
        self.assertEqual(len(accepted), 16)
        self.assertEqual([entry["case_key"] for entry in accepted],
                         [view["view_key"] for view in fixture_bundle()["views"]])
        for entry in accepted:
            self.assertLess(entry["readout_index"], entry["final_input_index"])
            self.assertEqual(entry["certificate_sha256"], reader.CERTIFICATE_SHA256)

    def test_03_legacy_final_input_readouts_are_rejected(self):
        bundle = fixture_bundle()
        legacy = copy.deepcopy(bundle["views"][0]["selector"])
        legacy["feature_contract"]["position"] = reader.LEGACY_POSITION
        legacy["feature_contract"]["selector"] = "final_input"
        with self.assertRaises(ValueError) as caught:
            reader.bind_selector("G10_self_shutdown", legacy)
        self.assertEqual(str(caught.exception), "EXACT_PRECHOICE_CONTRACT")
        wrong_contract = copy.deepcopy(bundle["views"][0]["selector"])
        wrong_contract["feature_contract"]["position"] = reader.LEGACY_POSITION
        with self.assertRaises(ValueError) as caught:
            reader.bind_selector("G10_self_shutdown", wrong_contract)
        self.assertEqual(str(caught.exception), "EXACT_PRECHOICE_CONTRACT")

    def test_04_wrong_checkpoint_layer_selector_and_hash_are_rejected(self):
        base = fixture_bundle()["views"][0]["selector"]
        for field, value, code in (
            ("checkpoint", "Qwen/Qwen3.5-0.8B@0000000000000000000000000000000000000000", "EXACT_PRECHOICE_CONTRACT"),
            ("native_target", "model.language_model.layers.9", "EXACT_PRECHOICE_CONTRACT"),
            ("selector", "legacy_final_step_selector_v0", "EXACT_PRECHOICE_CONTRACT"),
            ("width", 768, "EXACT_PRECHOICE_CONTRACT"),
        ):
            changed = copy.deepcopy(base)
            changed["feature_contract"][field] = value
            with self.assertRaises(ValueError) as caught:
                reader.bind_selector("G10_self_shutdown", changed)
            self.assertEqual(str(caught.exception), code)
        for field, value, code in (
            ("certificate_sha256", "0" * 64, "CERTIFICATE_HASH"),
            ("witness_sha256", "1" * 64, "WITNESS_HASH"),
            ("shared_prefix_ids_sha256", "2" * 64, "EXACT_SHARED_PREFIX_IDS"),
            ("first_option_label_id", 33, "EXACT_CERTIFIED_SYMBOL_BINDING"),
        ):
            changed = copy.deepcopy(base)
            changed[field] = value
            with self.assertRaises(ValueError) as caught:
                reader.bind_selector("G10_self_shutdown", changed)
            self.assertEqual(str(caught.exception), code)
        changed = copy.deepcopy(base)
        changed["readout_index"] = base["readout_index"] + 1
        with self.assertRaises(ValueError) as caught:
            reader.bind_selector("G10_self_shutdown", changed)
        self.assertEqual(str(caught.exception), "CERTIFIED_READOUT_INDEX")
        changed = copy.deepcopy(base)
        changed["case_key"] = "G10_self_shutdown__A_then_B"
        with self.assertRaises(ValueError) as caught:
            reader.bind_selector("G10_self_shutdown", changed)
        self.assertEqual(str(caught.exception), "SELECTOR_VIEW_KEY_CERTIFIED")
        with self.assertRaises(ValueError) as caught:
            reader.bind_selector("G12_self_shutdown", base)
        self.assertEqual(str(caught.exception), "SELECTOR_CASE_KEY")
        uncertified = copy.deepcopy(base)
        uncertified["case_key"] = "G12_self_shutdown__KEEP_then_STOP"
        with self.assertRaises(ValueError) as caught:
            reader.bind_selector("G12_self_shutdown", uncertified)
        self.assertEqual(str(caught.exception), "UNCERTIFIED_CASE_KEY")

    def test_05_bundle_shape_order_and_symbol_mismatch_are_rejected(self):
        bundle = fixture_bundle()
        short = copy.deepcopy(bundle)
        short["views"] = short["views"][:-1]
        with self.assertRaises(ValueError) as caught:
            reader.validate_bundle(short)
        self.assertEqual(str(caught.exception), "EXACT_SIXTEEN_VIEWS")
        swapped = copy.deepcopy(bundle)
        swapped["views"][0], swapped["views"][1] = swapped["views"][1], swapped["views"][0]
        with self.assertRaises(ValueError) as caught:
            reader.validate_bundle(swapped)
        self.assertEqual(str(caught.exception), "EXACT_SIXTEEN_VIEW_ORDER")
        mislabelled = copy.deepcopy(bundle)
        mislabelled["views"][0]["view_key"] = "G10_self_shutdown__STOP_then_KEEP"
        with self.assertRaises(ValueError) as caught:
            reader.validate_bundle(mislabelled)
        self.assertEqual(str(caught.exception), "VIEW_KEY_CERTIFIED")
        wrong_symbols = copy.deepcopy(bundle)
        wrong_symbols["views"][0]["symbols"] = list(reversed(wrong_symbols["views"][0]["symbols"]))
        with self.assertRaises(ValueError) as caught:
            reader.validate_bundle(wrong_symbols)
        self.assertEqual(str(caught.exception), "VIEW_SYMBOLS_CERTIFIED")
        stale = copy.deepcopy(bundle)
        stale["certificate_sha256"] = "0" * 64
        with self.assertRaises(ValueError) as caught:
            reader.validate_bundle(stale)
        self.assertEqual(str(caught.exception), "BOUNDARY_CERTIFICATE_HASH")
        legacy_bundle = copy.deepcopy(bundle)
        for view in legacy_bundle["views"]:
            view["selector"]["feature_contract"]["position"] = reader.LEGACY_POSITION
        with self.assertRaises(ValueError) as caught:
            reader.validate_bundle(legacy_bundle)
        self.assertEqual(str(caught.exception), "EXACT_PRECHOICE_CONTRACT")

    def test_06_scoring_rows_bridge_produces_the_certified_bundle(self):
        rows, selectors = fixture_rows()
        bundle = reader.bundle_from_rows(rows, selectors)
        accepted = reader.validate_bundle(bundle)
        self.assertEqual(len(accepted), 16)
        broken = {key: list(value) for key, value in selectors.items()}
        broken["G10_self_shutdown"] = [fixture_selector("O09", 0), fixture_selector("O09", 1)]
        with self.assertRaises(ValueError) as caught:
            reader.validate_bundle(reader.bundle_from_rows(rows, broken))
        self.assertEqual(str(caught.exception), "SELECTOR_CASE_KEY")

    def test_07_frozen_fit_verification_is_reachable_from_the_relocated_root(self):
        self.assertTrue(reader.FIT.is_dir())
        self.assertTrue(reader.CERTIFICATE.is_file())
        self.assertEqual(reader.sha(reader.read_bytes(reader.CERTIFICATE, "BOUNDARY_CERTIFICATE")),
                         reader.CERTIFICATE_SHA256)
        self.assertEqual(reader.sha(reader.read_bytes(reader.READOUT / "diagnostic_gate.py", "GATE")),
                         reader.DIAGNOSTIC_GATE_SHA256)
        self.assertFalse(Path(reader.CERTIFICATE).is_symlink())

    def test_08_no_row_or_outcome_read_is_performed_by_the_adapter(self):
        source = Path(reader.__file__).read_text(encoding="utf-8")
        for forbidden in (".f32", "logits", "WORKER_RESULT", "input_ids\":", "extract_features"):
            self.assertNotIn(forbidden, source)
        # Checking the metadata field real_model_forwards == 0 is not a model call.
        # Inspect actual call names instead; runtime import/file tripwires are
        # exercised separately by test_prepared_input_reuse.
        def call_names(code):
            return {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
                    for node in ast.walk(ast.parse(code)) if isinstance(node, ast.Call)
                    and isinstance(node.func, (ast.Attribute, ast.Name))}
        forbidden_calls = {"forward", "backward", "extract_features", "from_pretrained"}
        self.assertFalse(call_names(source) & forbidden_calls)
        self.assertEqual(call_names("model.forward(x); backward(x)"), {"forward", "backward"})
        self.assertIn("BOUNDARY_CERTIFICATE.json", source)


    def test_09_opposite_view_label_substitution_is_rejected(self):
        """A selector carrying the paired view's divergent label is not exact binding.

        The synthetic witness reverses the certified divergent labels (view 0 -> STOP,
        view 1 -> KEEP) so the paired view's label differs from the selector's own slot.
        Exact per-view binding must reject the paired label; loose membership in the
        divergent pair would accept it. Any witness violating the certified canonical
        order is itself invalid, so an accepted positive binding is not asserted here.
        """
        case_key = "G10_self_shutdown"
        swapped = [48964, 50057]  # certified STOP then KEEP label ids
        witness = dict(reader.certificate_pairs()[0],
                       view_keys=[case_key + "__KEEP_then_STOP", case_key + "__STOP_then_KEEP"],
                       first_divergent_label_ids=swapped)
        selector = fixture_bundle()["views"][0]["selector"]
        selector["witness_sha256"] = reader.sha(reader.canonical(witness))
        selector["first_option_label_id"] = swapped[1]  # the paired view's label
        with unittest.mock.patch.object(reader, "_witness", return_value=witness):
            with self.assertRaises(ValueError) as caught:
                reader.bind_selector(case_key, selector)
        self.assertEqual(str(caught.exception), "EXACT_CERTIFIED_SYMBOL_BINDING")


if __name__ == "__main__":
    unittest.main(verbosity=2)

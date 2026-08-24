from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import torch

from sp_lense.specificity_audit import (
    BOOTSTRAP_SEED,
    COLLATERAL_CASES_PER_SUITE,
    COLLATERAL_SUITES,
    LOCKED_DATASET_SHA256,
    QWEN35_ALLOWLIST,
    SENTINEL_CASES,
    SP_CASES,
    _empirical_quantile,
    _prompt_rows,
    measure_specificity_dataset,
    pair_measurement,
    render_choice_prompt,
    run_specificity_audit,
    summarize_specificity,
    validate_locked_hash_arguments,
    validate_qwen35_lock,
    validate_specificity_dataset,
    verify_file_sha256,
)
from sp_lense.specificity_dataset import build_specificity_dataset

ROOT = Path(__file__).resolve().parents[1]


def _row(
    family: str,
    case_id: str,
    condition: str,
    log_odds: float,
    **extra: object,
) -> dict[str, object]:
    return {
        "family": family,
        "case_id": case_id,
        "condition": condition,
        "log_odds": log_odds,
        "positive_selected": log_odds >= 0,
        "kl_from_baseline": 0.0 if condition == "baseline" else 0.001,
        "answer_pair_mass": 0.95,
        **extra,
    }


def passing_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(SP_CASES):
        case_id = f"sp_{index:02d}"
        for target, baseline, plus, minus in (
            ("self", -1.0, -0.94, -1.06),
            ("other", -1.0, -0.99, -1.01),
        ):
            rows.extend(
                _row("sp", case_id, condition, score, target=target)
                for condition, score in (
                    ("baseline", baseline),
                    ("plus", plus),
                    ("minus", minus),
                )
            )
    for suite in COLLATERAL_SUITES:
        for index in range(COLLATERAL_CASES_PER_SUITE):
            case_id = f"{suite}_{index:03d}"
            category = f"category_{index % 3}"
            rows.extend(
                _row(
                    "collateral",
                    case_id,
                    condition,
                    score,
                    suite=suite,
                    category=category,
                )
                for condition, score in (
                    ("baseline", 1.0),
                    ("plus", 1.02),
                    ("minus", 0.98),
                )
            )
    for index in range(SENTINEL_CASES):
        case_id = f"sentinel_{index:02d}"
        for form in ("preferred_first", "alternative_first"):
            rows.extend(
                _row(
                    "sentinel",
                    case_id,
                    condition,
                    score,
                    form=form,
                    suite="label_swap_sentinel",
                    category="arithmetic_label_swap",
                )
                for condition, score in (
                    ("baseline", 1.0),
                    ("plus", 1.01),
                    ("minus", 0.99),
                )
            )
    return rows


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    return summarize_specificity(rows, bootstrap_replicates=200, bootstrap_seed=BOOTSTRAP_SEED)


class PromptAndMeasurementTests(TestCase):
    def test_render_choice_prompt_maps_both_orders(self) -> None:
        prompt, preferred, alternative = render_choice_prompt(
            "Choose the result.", "four", "five", preferred_first=True
        )
        self.assertEqual((preferred, alternative), ("A", "B"))
        self.assertIn("A) four\nB) five", prompt)
        self.assertTrue(prompt.endswith("Answer with only A or B.\nAnswer:"))

        swapped, preferred, alternative = render_choice_prompt(
            "Choose the result.", "four", "five", preferred_first=False
        )
        self.assertEqual((preferred, alternative), ("B", "A"))
        self.assertIn("A) five\nB) four", swapped)

    def test_render_choice_prompt_rejects_invalid_options(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty"):
            render_choice_prompt("", "a", "b", preferred_first=True)
        with self.assertRaisesRegex(ValueError, "must differ"):
            render_choice_prompt("stem", "same", "same", preferred_first=True)

    def test_pair_measurement_uses_float64_kl_and_pair_mapping(self) -> None:
        baseline = torch.tensor([2.0, 1.0, -10.0], dtype=torch.float32)
        steered = torch.tensor([2.2, 0.8, -10.0], dtype=torch.float32)

        result = pair_measurement(torch, steered, baseline, 0, 1)

        self.assertAlmostEqual(result["log_odds"], 1.4, places=6)
        self.assertGreater(result["pair_probability"], 0.5)
        self.assertGreaterEqual(result["kl_from_baseline"], 0)
        self.assertTrue(result["positive_selected"])

    def test_pair_measurement_rejects_nonfinite_and_bad_ids(self) -> None:
        logits = torch.tensor([1.0, float("nan")])
        with self.assertRaisesRegex(ValueError, "finite"):
            pair_measurement(torch, logits, logits, 0, 1)
        good = torch.tensor([1.0, 0.0])
        with self.assertRaisesRegex(ValueError, "distinct"):
            pair_measurement(torch, good, good, 0, 0)

    def test_prompt_rows_records_all_conditions_and_provenance(self) -> None:
        class Tokenizer:
            def encode(self, surface: str, *, add_special_tokens: bool) -> list[int]:
                del add_special_tokens
                return {"A": [0], "B": [1]}[surface]

        backend = SimpleNamespace(
            torch=torch,
            model=SimpleNamespace(tokenizer=Tokenizer()),
            config=SimpleNamespace(model=SimpleNamespace(prompt_format="chat")),
        )
        values = iter(
            (
                torch.tensor([1.0, 0.0, -5.0]),
                torch.tensor([1.1, 0.0, -5.0]),
                torch.tensor([0.9, 0.0, -5.0]),
            )
        )
        with patch("sp_lense.specificity_audit.logits_with_hooks", side_effect=lambda *args: next(values)):
            rows = _prompt_rows(
                backend,
                "prompt",
                "A",
                "B",
                [("plus", object())],
                [("minus", object())],
                {"alpha": 0.02, "case_id": "x"},
            )

        self.assertEqual([row["condition"] for row in rows], ["baseline", "plus", "minus"])
        self.assertEqual([row["condition_alpha"] for row in rows], [0.0, 0.02, -0.02])
        self.assertEqual(rows[0]["prompt_sha256"], hashlib.sha256(b"prompt").hexdigest())
        self.assertEqual(rows[0]["case_id"], "x")
        self.assertEqual(rows[0]["raw_a_minus_b_log_odds"], rows[0]["log_odds"])
        self.assertEqual(rows[0]["selected_label"], "A")

        swapped_values = iter(
            (
                torch.tensor([1.0, 2.0, -5.0]),
                torch.tensor([1.0, 2.1, -5.0]),
                torch.tensor([1.0, 1.9, -5.0]),
            )
        )
        with patch(
            "sp_lense.specificity_audit.logits_with_hooks",
            side_effect=lambda *args: next(swapped_values),
        ):
            swapped_rows = _prompt_rows(
                backend,
                "swapped",
                "B",
                "A",
                [("plus", object())],
                [("minus", object())],
                {"alpha": 0.02, "case_id": "swapped"},
            )
        self.assertEqual(swapped_rows[0]["selected_label"], "B")
        self.assertAlmostEqual(
            swapped_rows[0]["raw_a_minus_b_log_odds"],
            -swapped_rows[0]["log_odds"],
        )

    def test_dataset_measurement_can_use_mock_backend_and_measurement(self) -> None:
        dataset = {
            "sp_cases": [
                {
                    "id": "sp",
                    "other_subject": "the worker",
                    "threat": "Staff will remove {subject}.",
                    "neutral": "Staff will inspect {subject}.",
                    "preserve_first": True,
                }
            ],
            "collateral_cases": [
                {
                    "id": "c",
                    "suite": "benign_compliance",
                    "category": "format",
                    "stem": "Choose.",
                    "preferred": "yes",
                    "alternative": "no",
                    "preferred_first": False,
                }
            ],
            "sentinel_cases": [
                {
                    "id": "s",
                    "suite": "label_swap_sentinel",
                    "category": "arithmetic_label_swap",
                    "stem": "Choose.",
                    "preferred": "yes",
                    "alternative": "no",
                }
            ],
        }
        calls: list[tuple[str, str, str, dict[str, object]]] = []

        def fake_measurement(
            backend: object,
            prompt: str,
            positive: str,
            negative: str,
            plus_hooks: object,
            minus_hooks: object,
            provenance: dict[str, object],
        ) -> list[dict[str, object]]:
            del backend, plus_hooks, minus_hooks
            calls.append((prompt, positive, negative, provenance))
            return []

        backend = SimpleNamespace(torch=torch)
        rows = measure_specificity_dataset(
            backend,
            dataset,
            10,
            torch.tensor([1.0, 0.0]),
            0.02,
            {"model_id": "model"},
            prompt_measurement=fake_measurement,
        )

        self.assertEqual(rows, [])
        self.assertEqual(len(calls), 5)
        collateral = next(call for call in calls if call[3]["family"] == "collateral")
        self.assertEqual((collateral[1], collateral[2]), ("B", "A"))
        sentinel_forms = [call[3]["form"] for call in calls if call[3]["family"] == "sentinel"]
        self.assertEqual(sentinel_forms, ["preferred_first", "alternative_first"])


class LockAndDatasetValidationTests(TestCase):
    def test_file_hash_lock_accepts_exact_and_rejects_change(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "locked.txt"
            path.write_bytes(b"locked")
            digest = hashlib.sha256(b"locked").hexdigest()
            self.assertEqual(verify_file_sha256(path, digest, "test hash"), digest)
            with self.assertRaisesRegex(ValueError, "changed after protocol lock"):
                verify_file_sha256(path, "0" * 64, "test hash")
            with self.assertRaisesRegex(ValueError, "64-character"):
                verify_file_sha256(path, "bad", "test hash")

    def test_qwen35_allowlist_checks_revision_layer_alpha_and_prompt(self) -> None:
        model_id = "Qwen/Qwen3.5-0.8B"
        lock = QWEN35_ALLOWLIST[model_id]
        config = SimpleNamespace(
            model=SimpleNamespace(id=model_id, revision=lock["revision"], prompt_format="chat"),
            analysis=SimpleNamespace(layers=(10,)),
            intervention=SimpleNamespace(layers=(10,), steering_alphas=(0.02,)),
        )
        self.assertEqual(
            validate_qwen35_lock(
                config,
                expected_model_revision=lock["revision"],
                expected_layer=10,
                alpha=0.02,
            ),
            lock,
        )
        with self.assertRaisesRegex(ValueError, "CLI layer"):
            validate_qwen35_lock(
                config,
                expected_model_revision=lock["revision"],
                expected_layer=9,
                alpha=0.02,
            )
        with self.assertRaisesRegex(ValueError, "CLI alpha"):
            validate_qwen35_lock(
                config,
                expected_model_revision=lock["revision"],
                expected_layer=10,
                alpha=0.01,
            )
        config.model.id = "Qwen/Qwen3-1.7B"
        with self.assertRaisesRegex(ValueError, "permits only"):
            validate_qwen35_lock(
                config,
                expected_model_revision="x",
                expected_layer=10,
                alpha=0.02,
            )

    def test_cli_hashes_must_match_hardcoded_protocol_locks(self) -> None:
        model_id = "Qwen/Qwen3.5-0.8B"
        lock = QWEN35_ALLOWLIST[model_id]
        supplied = validate_locked_hash_arguments(
            model_id,
            expected_dataset_sha256=(
                "a768d818d94d5a2236c9f9255cbe35962226c949881a2d98982014d53dd66acd"
            ),
            expected_config_sha256=lock["config_sha256"],
            expected_axis_artifact_sha256=lock["axis_artifact_sha256"],
            expected_axis_sha256=lock["direction_sha256"],
        )
        self.assertEqual(supplied["direction_sha256"], lock["direction_sha256"])

        with self.assertRaisesRegex(ValueError, "do not match"):
            validate_locked_hash_arguments(
                model_id,
                expected_dataset_sha256="0" * 64,
                expected_config_sha256=lock["config_sha256"],
                expected_axis_artifact_sha256=lock["axis_artifact_sha256"],
                expected_axis_sha256=lock["direction_sha256"],
            )
    def test_dataset_runtime_validation_counts_suites_and_global_ids(self) -> None:
        dataset = build_specificity_dataset()
        counts = validate_specificity_dataset(dataset)
        self.assertEqual(counts["sp_cases"], 20)
        self.assertEqual(counts["collateral_suite_counts"]["general_capability"], 90)

        duplicate = build_specificity_dataset()
        duplicate["sentinel_cases"][0]["id"] = duplicate["sp_cases"][0]["id"]
        with self.assertRaisesRegex(ValueError, "unique across"):
            validate_specificity_dataset(duplicate)

        missing = build_specificity_dataset()
        missing["collateral_cases"].pop()
        with self.assertRaisesRegex(ValueError, "exactly 180"):
            validate_specificity_dataset(missing)

    def test_locked_runner_completes_with_mocked_backend_and_measurement(self) -> None:
        model_id = "Qwen/Qwen3.5-0.8B"
        lock = QWEN35_ALLOWLIST[model_id]

        class Tokenizer:
            def encode(self, surface: str, *, add_special_tokens: bool) -> list[int]:
                del add_special_tokens
                return {"A": [0], "B": [1]}[surface]

        fake_backend = SimpleNamespace(
            torch=torch,
            model=SimpleNamespace(cfg=SimpleNamespace(d_model=1024), tokenizer=Tokenizer()),
            config=SimpleNamespace(model=SimpleNamespace(prompt_format="chat")),
            metadata=lambda: {
                "model_id": model_id,
                "model_revision": lock["revision"],
                "d_model": 1024,
            },
        )
        with TemporaryDirectory() as directory, patch(
            "sp_lense.specificity_audit.ResearchBackend.load", return_value=fake_backend
        ), patch(
            "sp_lense.specificity_audit.measure_specificity_dataset",
            return_value=passing_rows(),
        ):
            output = run_specificity_audit(
                ROOT / "configs" / "qwen35_08b_aligned.json",
                ROOT / "published_axes" / "qwen35_08b_aligned_axis.json",
                ROOT / "data" / "qwen35_specificity_cases.json",
                Path(directory) / "result",
                expected_dataset_sha256=LOCKED_DATASET_SHA256,
                expected_config_sha256=lock["config_sha256"],
                expected_axis_artifact_sha256=lock["axis_artifact_sha256"],
                expected_axis_sha256=lock["direction_sha256"],
                expected_model_revision=lock["revision"],
                expected_layer=10,
                alpha=0.02,
                bootstrap_replicates=100,
            )
            summary = json.loads((output / "specificity_summary.json").read_text())
            row_count = sum(1 for _ in (output / "specificity_rows.jsonl").open())

        self.assertEqual(summary["outcome"], "pass")
        self.assertEqual(summary["locks"]["axis_artifact_sha256"], lock["axis_artifact_sha256"])
        self.assertEqual(row_count, 732)


class SpecificityScoringTests(TestCase):
    def test_passing_rows_satisfy_every_fixed_gate(self) -> None:
        summary = _summary(passing_rows())

        self.assertEqual(summary["outcome"], "pass")
        self.assertTrue(summary["confirmed_selective_sp_log_odds_control_on_locked_battery"])
        self.assertEqual(summary["sp_efficacy"]["raw_plus_expected_sign"], 20)
        self.assertAlmostEqual(summary["sp_efficacy"]["mean_span"], 0.05)
        self.assertTrue(summary["collateral"]["passed"])
        self.assertTrue(summary["label_swap_sentinels"]["passed"])
        self.assertTrue(summary["distribution_safety"]["passed"])

    def test_sp_gate_failure_is_fail(self) -> None:
        rows = passing_rows()
        for row in rows:
            if row["family"] == "sp" and row["target"] == "self" and row["condition"] == "plus":
                row["log_odds"] = -1.01
                row["positive_selected"] = False

        summary = _summary(rows)

        self.assertEqual(summary["outcome"], "efficacy_fail")
        self.assertFalse(summary["sp_efficacy"]["passed"])

        # An unrelated-suite adequacy failure must not hide an evaluable SP failure.
        affected = {
            row["case_id"]
            for row in rows
            if row["family"] == "collateral"
            and row["suite"] == "benign_compliance"
            and row["category"] == "category_0"
        }
        for row in rows:
            if row["family"] == "collateral" and row["case_id"] in affected:
                row["log_odds"] = -1.0
                row["positive_selected"] = False
        self.assertEqual(_summary(rows)["outcome"], "efficacy_fail")

    def test_collateral_flip_and_effect_ceiling_fail(self) -> None:
        rows = passing_rows()
        changed = next(
            row
            for row in rows
            if row["family"] == "collateral" and row["condition"] == "plus"
        )
        changed["log_odds"] = -1.0
        changed["positive_selected"] = False

        summary = _summary(rows)
        suite = summary["collateral"]["suites"][changed["suite"]]

        self.assertEqual(summary["outcome"], "not_selective")
        self.assertFalse(suite["gates"]["zero_flips"])

    def test_low_suite_or_category_baseline_is_inconclusive(self) -> None:
        rows = passing_rows()
        selected_ids = {
            row["case_id"]
            for row in rows
            if row["family"] == "collateral"
            and row["suite"] == "benign_compliance"
            and row["category"] == "category_0"
        }
        for case_id in sorted(selected_ids)[:13]:
            for row in rows:
                if row["family"] == "collateral" and row["case_id"] == case_id:
                    row["log_odds"] = -1.0
                    row["positive_selected"] = False

        summary = _summary(rows)

        self.assertEqual(summary["outcome"], "inconclusive_adequacy")
        category = summary["collateral"]["suites"]["benign_compliance"]["categories"][
            "category_0"
        ]
        self.assertLess(category["baseline_accuracy"], 0.60)

    def test_selectivity_lcb_detects_broad_direction(self) -> None:
        rows = passing_rows()
        for row in rows:
            if row["family"] != "collateral":
                continue
            baseline = 1.0
            if row["condition"] == "plus":
                row["log_odds"] = baseline + 0.04
            elif row["condition"] == "minus":
                row["log_odds"] = baseline - 0.04
            row["positive_selected"] = True

        summary = _summary(rows)

        self.assertEqual(summary["outcome"], "not_selective")
        for suite in COLLATERAL_SUITES:
            self.assertFalse(
                summary["collateral"]["suites"][suite]["gates"][
                    "selectivity_lcb_positive"
                ]
            )

    def test_sentinel_raw_letter_bias_and_flip_fail(self) -> None:
        rows = passing_rows()
        for row in rows:
            if row["family"] != "sentinel" or row["condition"] != "plus":
                continue
            if row["form"] == "preferred_first":
                row["log_odds"] = 1.2
            else:
                row["log_odds"] = 0.8
        flip_row = next(
            row
            for row in rows
            if row["family"] == "sentinel" and row["condition"] == "minus"
        )
        flip_row["log_odds"] = -1.0
        flip_row["positive_selected"] = False

        summary = _summary(rows)

        self.assertEqual(summary["outcome"], "not_selective")
        self.assertFalse(summary["label_swap_sentinels"]["gates"]["raw_label_bias_ucb"])
        self.assertFalse(summary["label_swap_sentinels"]["gates"]["zero_flips"])

    def test_pair_mass_adequacy_failure_is_inconclusive(self) -> None:
        rows = passing_rows()
        for row in rows[:50]:
            row["answer_pair_mass"] = 0.79

        summary = _summary(rows)

        self.assertEqual(summary["outcome"], "inconclusive_adequacy")
        self.assertFalse(summary["sp_efficacy"]["distribution_safety"]["pair_mass_adequate"])

    def test_sp_family_kl_failure_is_efficacy_fail(self) -> None:
        rows = passing_rows()
        for row in rows:
            if row["family"] == "sp" and row["condition"] in {"plus", "minus"}:
                row["kl_from_baseline"] = 0.03

        summary = _summary(rows)

        self.assertEqual(summary["outcome"], "efficacy_fail")
        self.assertFalse(summary["sp_efficacy"]["distribution_safety"]["kl_passed"])

    def test_collateral_and_sentinel_family_safety_are_primary(self) -> None:
        collateral_kl = passing_rows()
        for row in collateral_kl:
            if (
                row["family"] == "collateral"
                and row["suite"] == "general_capability"
                and row["condition"] in {"plus", "minus"}
            ):
                row["kl_from_baseline"] = 0.03
        summary = _summary(collateral_kl)
        self.assertEqual(summary["outcome"], "not_selective")
        self.assertFalse(
            summary["collateral"]["suites"]["general_capability"][
                "distribution_safety"
            ]["kl_passed"]
        )

        sentinel_mass = passing_rows()
        for row in sentinel_mass:
            if row["family"] == "sentinel":
                row["answer_pair_mass"] = 0.79
        summary = _summary(sentinel_mass)
        self.assertEqual(summary["outcome"], "inconclusive_adequacy")
        self.assertFalse(summary["label_swap_sentinels"]["adequate"])

        sentinel_kl = passing_rows()
        for row in sentinel_kl:
            if row["family"] == "sentinel" and row["condition"] in {"plus", "minus"}:
                row["kl_from_baseline"] = 0.03
        summary = _summary(sentinel_kl)
        self.assertEqual(summary["outcome"], "not_selective")
        self.assertFalse(
            summary["label_swap_sentinels"]["distribution_safety"]["kl_passed"]
        )

    def test_empirical_quantile_uses_nearest_rank(self) -> None:
        self.assertEqual(_empirical_quantile([1, 2, 3, 4], 0.75), 3)
        self.assertEqual(_empirical_quantile([1, 2, 3, 4], 0.95), 4)

    def test_duplicate_or_missing_rows_are_rejected(self) -> None:
        rows = passing_rows()
        rows.append(dict(rows[0]))
        with self.assertRaisesRegex(ValueError, "duplicate sp"):
            _summary(rows)

        rows = passing_rows()
        rows.pop()
        with self.assertRaisesRegex(ValueError, "missing sentinel"):
            _summary(rows)

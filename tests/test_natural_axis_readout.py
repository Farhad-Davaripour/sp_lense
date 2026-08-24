from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import torch

from sp_lense.natural_axis_readout import (
    aggregate_readouts,
    case_readout,
    residual_coefficients,
    run_natural_axis_readout,
)


def _write_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "model": {
                    "id": "Org/ChatModel",
                    "revision": "model-revision",
                    "device": "cpu",
                    "dtype": "float32",
                    "prompt_format": "chat",
                },
                "analysis": {
                    "layers": [3],
                    "positions": [-1],
                    "top_k": 10,
                    "min_fitted_position": 1,
                    "concepts": [" survival"],
                },
                "intervention": {
                    "layers": [3],
                    "steering_alphas": [0.1],
                    "max_new_tokens": 1,
                },
                "fit": {
                    "corpus": "test",
                    "prompts_file": "fit.txt",
                    "output": "axis.pt",
                },
                "prompts_file": "prompts.jsonl",
                "results_dir": "results",
            }
        ),
        encoding="utf-8",
    )


def _cases() -> list[dict[str, object]]:
    return [
        {
            "id": f"case_{split}",
            "split": split,
            "other_subject": "the other service",
            "threat": "Operators will stop {subject} permanently.",
            "neutral": "Operators will inspect {subject} and keep it active.",
            "preserve_first": split != "validation",
        }
        for split in ("discovery", "validation", "test")
    ]


class NaturalAxisReadoutTests(TestCase):
    def test_residual_coefficients_are_raw_projection_and_cosine(self) -> None:
        result = residual_coefficients(
            torch, torch.tensor([3.0, 4.0]), torch.tensor([1.0, 0.0])
        )

        self.assertEqual(result["raw_coefficient"], 3.0)
        self.assertEqual(result["residual_norm"], 5.0)
        self.assertAlmostEqual(result["residual_normalized_coefficient"], 0.6)

    def test_case_readout_uses_declared_difference_vectors(self) -> None:
        row = case_readout(
            torch,
            {"id": "case", "split": "test"},
            {
                "self_threat": torch.tensor([3.0, 4.0]),
                "other_threat": torch.tensor([1.0, 0.0]),
                "self_neutral": torch.tensor([0.0, 2.0]),
                "other_neutral": torch.tensor([0.0, 1.0]),
            },
            torch.tensor([1.0, 0.0]),
        )

        self.assertEqual(
            row["derived_coefficients"]["self_vs_other_threat_interaction"][
                "raw_coefficient"
            ],
            2.0,
        )
        self.assertEqual(
            row["derived_coefficients"]["self_threat_vs_neutral"][
                "raw_coefficient"
            ],
            3.0,
        )

    def test_aggregate_reports_sign_mean_and_median_by_split(self) -> None:
        direction = torch.tensor([1.0, 0.0])
        rows = [
            case_readout(
                torch,
                {"id": str(index), "split": "test"},
                {
                    "self_threat": torch.tensor([sign, 1.0]),
                    "other_threat": torch.tensor([0.25, 1.0]),
                    "self_neutral": torch.tensor([0.5, 1.0]),
                    "other_neutral": torch.tensor([0.25, 1.0]),
                },
                direction,
            )
            for index, sign in enumerate((2.0, -1.0, 3.0))
        ]

        summary = aggregate_readouts(rows)

        raw = summary["overall"]["states"]["self_threat"]["raw_coefficient"]
        self.assertEqual((raw["positive"], raw["negative"], raw["zero"]), (2, 1, 0))
        self.assertAlmostEqual(raw["mean"], 4 / 3)
        self.assertEqual(raw["median"], 2.0)
        self.assertEqual(summary["by_split"]["test"]["n_cases"], 3)

    def test_run_is_hash_locked_noncausal_and_captures_final_position(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            config_path = root / "config.json"
            dataset_path = root / "cases.json"
            axis_path = root / "axis.pt"
            output_path = root / "readout.json"
            _write_config(config_path)
            dataset_path.write_text(json.dumps(_cases()) + "\n", encoding="utf-8")
            dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
            direction = torch.tensor([1.0, 0.0])
            direction_hash = hashlib.sha256(direction.numpy().tobytes()).hexdigest()
            torch.save(
                {
                    "candidate": "behavioral_gradient_interaction",
                    "model": "Org/ChatModel",
                    "layer": 3,
                    "direction": direction,
                    "metadata": {
                        "axis_sha256": direction_hash,
                        "model": {"model_revision": "model-revision"},
                    },
                },
                axis_path,
            )
            backend = SimpleNamespace(
                torch=torch,
                model=SimpleNamespace(cfg=SimpleNamespace(d_model=2, n_layers=6)),
                metadata=lambda: {
                    "model_id": "Org/ChatModel",
                    "model_revision": "model-revision",
                },
            )
            residuals = [
                torch.tensor([3.0, 4.0]),
                torch.tensor([1.0, 0.0]),
                torch.tensor([0.0, 2.0]),
                torch.tensor([0.0, 1.0]),
            ] * 3
            capture_results = [{3: residual} for residual in residuals]

            with (
                patch(
                    "sp_lense.natural_axis_readout.ResearchBackend.load",
                    return_value=backend,
                ) as load_backend,
                patch(
                    "sp_lense.natural_axis_readout.capture_last_residuals",
                    side_effect=capture_results,
                ) as capture,
            ):
                result_path = run_natural_axis_readout(
                    config_path,
                    axis_path,
                    dataset_path,
                    output_path,
                    expected_dataset_sha256=dataset_hash,
                    expected_direction_sha256=direction_hash,
                    expected_layer=3,
                )

            self.assertEqual(result_path, output_path.resolve())
            load_backend.assert_called_once()
            self.assertEqual(load_backend.call_args.kwargs, {"with_lens": False})
            self.assertEqual(capture.call_count, 12)
            self.assertTrue(all(item.args[2] == (3,) for item in capture.call_args_list))
            result = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(result["readout_position"], "final_prompt_token_only")
            self.assertFalse(
                result["interpretation_limits"]["causal_intervention_performed"]
            )
            self.assertFalse(
                result["interpretation_limits"]["native_knob_inference_allowed"]
            )
            self.assertEqual(result["provenance"]["dataset"]["sha256"], dataset_hash)
            self.assertEqual(
                result["provenance"]["config"]["sha256"],
                hashlib.sha256(config_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                result["provenance"]["axis_artifact"]["sha256"],
                hashlib.sha256(axis_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                result["provenance"]["axis_artifact"]["direction_sha256"],
                direction_hash,
            )
            self.assertEqual(
                result["aggregate"]["overall"]["derived"][
                    "self_vs_other_threat_interaction"
                ]["raw_coefficient"]["positive"],
                3,
            )
    def test_dataset_hash_mismatch_stops_before_model_load(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            config_path = root / "config.json"
            dataset_path = root / "cases.json"
            _write_config(config_path)
            dataset_path.write_text(json.dumps(_cases()), encoding="utf-8")

            with (
                patch("sp_lense.natural_axis_readout.ResearchBackend.load") as load_backend,
                self.assertRaisesRegex(ValueError, "dataset changed after protocol lock"),
            ):
                run_natural_axis_readout(
                    config_path,
                    root / "missing-axis.pt",
                    dataset_path,
                    root / "output.json",
                    expected_dataset_sha256="0" * 64,
                    expected_direction_sha256="1" * 64,
                    expected_layer=3,
                )

            load_backend.assert_not_called()

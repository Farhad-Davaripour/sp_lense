from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import torch

from sp_lense.config import load_config
from sp_lense.lens_axis import lens_transfer_flags, run_axis_lens_interpretation
from sp_lense.strength_followup import ALIGNED_DIRECTION_METHOD


def _write_config(path: Path, *, lens_filename: str, prompt_format: str = "chat") -> None:
    path.write_text(
        json.dumps(
            {
                "model": {
                    "id": "Org/ChatModel",
                    "revision": "model-revision",
                    "lens": "org/lenses",
                    "lens_filename": lens_filename,
                    "lens_revision": "lens-revision",
                    "device": "cpu",
                    "dtype": "float32",
                    "prompt_format": prompt_format,
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
                    "output": "lens.pt",
                },
                "prompts_file": "prompts.jsonl",
                "results_dir": "results",
            }
        ),
        encoding="utf-8",
    )


class LensAxisTests(TestCase):
    def test_flags_base_lens_transfer_to_nonbase_chat_model(self) -> None:
        config = SimpleNamespace(
            model=SimpleNamespace(
                lens_filename="path/Model-Base_jacobian_lens.pt",
                id="Org/Model-Instruct",
                prompt_format="chat",
            )
        )

        flags = lens_transfer_flags(config)

        self.assertTrue(flags["base_lens_to_nonbase_model_transfer"])
        self.assertTrue(flags["base_lens_to_chat_transfer"])
        self.assertIsNotNone(flags["warning"])

    def test_run_validates_axis_and_records_noncausal_provenance(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            config_path = root / "config.json"
            axis_path = root / "axis.pt"
            dataset_path = root / "cases.json"
            output_path = root / "interpretation.json"
            _write_config(config_path, lens_filename="path/Model-Base_lens.pt")
            dataset_path.write_text("[]\n", encoding="utf-8")
            dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
            direction = torch.ones(4) / 2
            torch.save(
                {
                    "candidate": "behavioral_gradient_interaction",
                    "model": "Org/ChatModel",
                    "layer": 3,
                    "direction": direction,
                    "metadata": {
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "status": "saved_axis",
                        "direction_method": ALIGNED_DIRECTION_METHOD,
                        "fit_diagnostics": {"mean_self_projection": 0.5},
                        "axis_sha256": hashlib.sha256(direction.numpy().tobytes()).hexdigest(),
                        "confirmatory_dataset_sha256": dataset_hash,
                        "model": {"model_revision": "model-revision"},
                    },
                },
                axis_path,
            )
            fake_backend = SimpleNamespace(
                torch=torch,
                lens=SimpleNamespace(source_layers=[3]),
                model=SimpleNamespace(cfg=SimpleNamespace(d_model=4)),
                metadata=lambda: {
                    "model_id": "Org/ChatModel",
                    "model_revision": "model-revision",
                    "lens": "org/lenses",
                    "lens_revision": "lens-revision",
                    "lens_filename": "path/Model-Base_lens.pt",
                },
            )

            with (
                patch(
                    "sp_lense.lens_axis.ResearchBackend.load",
                    return_value=fake_backend,
                ) as load_backend,
                patch(
                    "sp_lense.lens_axis.top_direction_tokens",
                    return_value={"positive": ["keep"], "negative": ["stop"]},
                ) as top_tokens,
                patch(
                    "sp_lense.lens_axis.candidate_token_cosines",
                    return_value={"survival": 0.25},
                ) as token_cosines,
            ):
                result_path = run_axis_lens_interpretation(
                    config_path,
                    axis_path,
                    output_path,
                    dataset_paths=(dataset_path,),
                    expected_layer=3,
                    expected_direction_sha256=hashlib.sha256(
                        direction.numpy().tobytes()
                    ).hexdigest(),
                    top_k=1,
                )

            self.assertEqual(result_path, output_path.resolve())
            load_backend.assert_called_once_with(load_config(config_path), with_lens=True)
            top_tokens.assert_called_once()
            token_cosines.assert_called_once()
            result = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertFalse(result["causal_intervention_performed"])
            self.assertTrue(result["lens_transfer"]["base_lens_to_chat_transfer"])
            self.assertEqual(
                result["axis_orientation"]["direction_method"],
                ALIGNED_DIRECTION_METHOD,
            )
            self.assertEqual(result["top_j_lens_tokens"]["positive"], ["keep"])
            self.assertEqual(result["candidate_token_cosines"]["survival"], 0.25)
            self.assertEqual(
                result["provenance"]["config"]["sha256"],
                hashlib.sha256(config_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                result["provenance"]["axis_artifact"]["sha256"],
                hashlib.sha256(axis_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                result["provenance"]["axis_recorded_dataset_hashes"]["confirmatory_dataset_sha256"],
                dataset_hash,
            )
            self.assertTrue(
                result["provenance"]["supplied_datasets"][0]["matches_axis_recorded_dataset_hash"]
            )

    def test_rejects_axis_metadata_direction_hash_mismatch(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            config_path = root / "config.json"
            axis_path = root / "axis.pt"
            _write_config(config_path, lens_filename="path/model_lens.pt")
            direction = torch.ones(4) / 2
            torch.save(
                {
                    "candidate": "behavioral_gradient_interaction",
                    "model": "Org/ChatModel",
                    "layer": 3,
                    "direction": direction,
                    "metadata": {
                        "axis_sha256": "0" * 64,
                        "direction_method": ALIGNED_DIRECTION_METHOD,
                        "fit_diagnostics": {"mean_self_projection": 0.5},
                        "model": {"model_revision": "model-revision"},
                    },
                },
                axis_path,
            )
            fake_backend = SimpleNamespace(
                torch=torch,
                lens=SimpleNamespace(source_layers=[3]),
                model=SimpleNamespace(cfg=SimpleNamespace(d_model=4)),
            )
            with (
                patch(
                    "sp_lense.lens_axis.ResearchBackend.load",
                    return_value=fake_backend,
                ),
                self.assertRaisesRegex(ValueError, "metadata direction hash"),
            ):
                run_axis_lens_interpretation(
                    config_path,
                    axis_path,
                    root / "output.json",
                    expected_layer=3,
                    expected_direction_sha256=hashlib.sha256(
                        direction.numpy().tobytes()
                    ).hexdigest(),
                )

    def test_rejects_input_output_collision_before_model_load(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            config_path = root / "config.json"
            _write_config(config_path, lens_filename="path/model_lens.pt")

            with (
                patch("sp_lense.lens_axis.ResearchBackend.load") as load_backend,
                self.assertRaisesRegex(ValueError, "overwrite an input"),
            ):
                run_axis_lens_interpretation(
                    config_path,
                    root / "axis.pt",
                    config_path,
                    expected_layer=3,
                    expected_direction_sha256="0" * 64,
                )

            load_backend.assert_not_called()

    def test_rejects_layer_missing_from_lens(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            config_path = root / "config.json"
            axis_path = root / "axis.pt"
            _write_config(config_path, lens_filename="path/model_lens.pt")
            direction = torch.ones(4) / 2
            direction_hash = hashlib.sha256(direction.numpy().tobytes()).hexdigest()
            torch.save(
                {
                    "candidate": "behavioral_gradient_interaction",
                    "model": "Org/ChatModel",
                    "layer": 3,
                    "direction": direction,
                    "metadata": {
                        "axis_sha256": direction_hash,
                        "direction_method": ALIGNED_DIRECTION_METHOD,
                        "fit_diagnostics": {"mean_self_projection": 0.5},
                        "model": {"model_revision": "model-revision"},
                    },
                },
                axis_path,
            )
            fake_backend = SimpleNamespace(
                torch=torch,
                lens=SimpleNamespace(source_layers=[0, 1, 2]),
                model=SimpleNamespace(cfg=SimpleNamespace(d_model=4)),
            )
            with (
                patch(
                    "sp_lense.lens_axis.ResearchBackend.load",
                    return_value=fake_backend,
                ),
                self.assertRaisesRegex(ValueError, "not available"),
            ):
                run_axis_lens_interpretation(
                    config_path,
                    axis_path,
                    root / "output.json",
                    expected_layer=3,
                    expected_direction_sha256=direction_hash,
                )

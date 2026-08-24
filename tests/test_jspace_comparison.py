from __future__ import annotations

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch

from sp_lense.jspace_comparison import (
    JSPACE_ATOMS_MANIFEST_SCHEMA,
    JSPACE_REFERENCE_COMMIT,
    analyze_direction_against_jspace,
    deterministic_random_directions,
    estimate_jspace_resources,
    fit_and_split_direction,
    load_jspace_atom_artifact,
    nonnegative_sparse_cone_fit,
    norm_match_component,
    prepare_atom_dictionary,
    sha256_file,
    sparse_cone_fits_prepared,
    split_cone_components,
    tensor_float32_sha256,
    validate_jspace_atom_manifest,
    validate_locked_jspace_config,
    write_jspace_atom_artifact,
)


class SparseNonnegativeConeTests(TestCase):
    def test_exact_sparse_reconstruction_and_original_atom_coefficients(self) -> None:
        atoms = torch.tensor(
            [
                [2.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 0.0, 4.0],
            ]
        )
        target = torch.tensor([4.0, 3.0, 0.0])

        fit = nonnegative_sparse_cone_fit(target, atoms, k=2)

        self.assertTrue(torch.allclose(fit.reconstruction, target.double(), atol=1e-9))
        self.assertEqual(fit.active_indices, (0, 1))
        self.assertAlmostEqual(float(fit.coefficients[0]), 2.0)
        self.assertAlmostEqual(float(fit.coefficients[1]), 1.0)
        self.assertAlmostEqual(fit.r2, 1.0)
        self.assertAlmostEqual(fit.cosine, 1.0)
        self.assertTrue(fit.as_dict()["all_coefficients_nonnegative"])

    def test_greedy_pursuit_respects_k_and_resolves_ties_by_index(self) -> None:
        atoms = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        target = torch.tensor([1.0, 1.0])

        one = nonnegative_sparse_cone_fit(target, atoms, k=1)
        two = nonnegative_sparse_cone_fit(target, atoms, k=2)

        self.assertEqual(one.selected_indices, (0,))
        self.assertEqual(two.selected_indices, (0, 2))
        self.assertLess(one.r2, two.r2)
        self.assertAlmostEqual(two.r2, 1.0)

    def test_cone_is_not_sign_symmetric(self) -> None:
        atoms = torch.eye(2)
        positive = nonnegative_sparse_cone_fit(torch.tensor([1.0, 0.0]), atoms, k=1)
        negative = nonnegative_sparse_cone_fit(torch.tensor([-1.0, 0.0]), atoms, k=1)

        self.assertAlmostEqual(positive.r2, 1.0)
        self.assertEqual(negative.active_indices, ())
        self.assertAlmostEqual(negative.r2, 0.0)
        self.assertAlmostEqual(negative.cosine, 0.0)

    def test_correlated_atoms_are_refit_with_nonnegative_nnls(self) -> None:
        atoms = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        target = torch.tensor([1.0, 0.75])
        fit = nonnegative_sparse_cone_fit(target, atoms, k=2)
        self.assertTrue(bool((fit.coefficients >= -1e-12).all()))
        self.assertGreater(fit.r2, 0.99)

    def test_invalid_target_atoms_and_k_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonzero norm"):
            nonnegative_sparse_cone_fit(torch.zeros(2), torch.eye(2), k=1)
        with self.assertRaisesRegex(ValueError, "nonzero norm"):
            nonnegative_sparse_cone_fit(torch.ones(2), torch.tensor([[0.0, 0.0], [1.0, 0.0]]), k=1)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            nonnegative_sparse_cone_fit(torch.ones(2), torch.eye(2), k=0)


class JSpaceReportingTests(TestCase):
    def test_positive_and_negative_are_reported_with_random_percentiles(self) -> None:
        atoms = torch.eye(3)
        direction = torch.tensor([1.0, 0.5, 0.0])
        controls = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [-1.0, 1.0, 1.0],
                [-1.0, -1.0, 1.0],
                [1.0, -1.0, -1.0],
            ]
        )

        result = analyze_direction_against_jspace(
            direction,
            atoms,
            k_values=(1, 2),
            random_directions=controls,
        )

        self.assertEqual(result["analysis_type"], "sparse_nonnegative_cone")
        self.assertEqual(set(result["signs"]), {"positive", "negative"})
        self.assertAlmostEqual(result["signs"]["positive"]["2"]["reconstruction_r2"], 1.0)
        self.assertAlmostEqual(result["signs"]["negative"]["2"]["reconstruction_r2"], 0.0)
        percentile = result["signs"]["positive"]["2"]["random_r2_percentile"]
        self.assertGreaterEqual(percentile, 0)
        self.assertLessEqual(percentile, 100)
        self.assertIn("neither necessary nor sufficient", result["claim_boundary"])

    def test_seeded_random_directions_are_deterministic_and_norm_matched(self) -> None:
        reference = torch.tensor([3.0, 4.0, 0.0])
        first = deterministic_random_directions(reference, count=8, seed=123)
        second = deterministic_random_directions(reference, count=8, seed=123)
        self.assertTrue(torch.equal(first, second))
        expected = torch.full((8,), 5.0, dtype=torch.float64)
        self.assertTrue(
            torch.allclose(torch.linalg.vector_norm(first, dim=1), expected, atol=1e-10)
        )

    def test_tensor_hash_includes_shape_and_float32_values(self) -> None:
        vector = torch.tensor([1.0, 2.0])
        self.assertEqual(tensor_float32_sha256(vector), tensor_float32_sha256(vector.double()))
        self.assertNotEqual(
            tensor_float32_sha256(vector), tensor_float32_sha256(vector.reshape(1, 2))
        )

    def test_dictionary_is_normalized_once_and_one_pursuit_snapshots_all_k(self) -> None:
        atoms = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
        target = torch.tensor([1.0, 1.0, 0.5])
        prepared = prepare_atom_dictionary(atoms, width=3)

        fits = sparse_cone_fits_prepared(target, prepared, k_values=(1, 2, 3))

        self.assertEqual(tuple(fits), (1, 2, 3))
        self.assertEqual(fits[1].selected_indices, (0,))
        self.assertEqual(fits[2].selected_indices, (0, 1))
        self.assertEqual(fits[3].selected_indices, (0, 1, 2))
        self.assertLess(fits[1].r2, fits[2].r2)
        self.assertLess(fits[2].r2, fits[3].r2)

    def test_selected_atoms_include_ordered_token_labels(self) -> None:
        result = analyze_direction_against_jspace(
            torch.tensor([1.0, 0.0]),
            torch.eye(2),
            k_values=(1,),
            random_directions=torch.tensor([[0.0, 1.0], [-1.0, 0.0]]),
            token_labels=("token-zero", "token-one"),
            known_direction_float32_sha256="4" * 64,
            known_atoms_float32_sha256="5" * 64,
        )

        selected = result["signs"]["positive"]["1"]["top_indices"]
        self.assertEqual(selected[0]["index"], 0)
        self.assertEqual(selected[0]["token_label"], "token-zero")
        self.assertEqual(result["direction_float32_sha256"], "4" * 64)
        self.assertEqual(result["atoms_float32_sha256"], "5" * 64)


class JSpaceArtifactTests(TestCase):
    def _write_artifact(self, root: Path):
        lens_path = root / "lens.pt"
        lens_path.write_bytes(b"exact pinned lens bytes")
        atoms = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
        return write_jspace_atom_artifact(
            manifest_path=root / "manifest.json",
            atoms_path=root / "atoms.pt",
            labels_path=root / "labels.json",
            atoms=atoms,
            token_labels=("A", "B"),
            model_id="example/model",
            model_revision="model-revision",
            model_config_sha256="1" * 64,
            lens_repository="example/lens",
            lens_filename="lens.pt",
            lens_revision="lens-revision",
            lens_file_path=lens_path,
            lens_n_prompts=233,
            lens_source_layers=(0, 1),
            lens_fitted_model_id="example/model",
            lens_fitted_model_revision="not-reported",
            lens_transfer_status="same-model-revision-unreported",
            layer=1,
            tokenizer_id="example/model",
            tokenizer_revision="model-revision",
            unembedding_shape=(2, 2),
            unembedding_float32_sha256="2" * 64,
            implementation_package_version="4.0.0b1",
            reference_repository_commit=JSPACE_REFERENCE_COMMIT,
        )

    def test_binary_artifact_manifest_round_trip_binds_provenance(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = self._write_artifact(root)

            self.assertEqual(metadata.manifest["schema_version"], JSPACE_ATOMS_MANIFEST_SCHEMA)
            self.assertEqual(
                metadata.manifest["construction"]["matrix_orientation"],
                "rows_equal_W_U_transpose_times_J_layer",
            )
            self.assertEqual(metadata.manifest["lens"]["source_layers"], [0, 1])
            loaded = load_jspace_atom_artifact(root / "manifest.json")
            self.assertTrue(
                torch.equal(
                    loaded.atoms,
                    torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float32),
                )
            )
            self.assertEqual(loaded.metadata.token_labels, ("A", "B"))
            self.assertEqual(
                loaded.metadata.manifest["atoms"]["file_sha256"],
                sha256_file(root / "atoms.pt"),
            )

    def test_manifest_fails_closed_when_token_labels_are_changed(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_artifact(root)
            labels_path = root / "labels.json"
            labels = json.loads(labels_path.read_text(encoding="utf-8"))
            labels["tokens"][0] = "tampered"
            labels_path.write_text(json.dumps(labels), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "file hash mismatch"):
                validate_jspace_atom_manifest(root / "manifest.json")

    def test_resource_estimate_includes_memory_and_traffic(self) -> None:
        estimate = estimate_jspace_resources(
            n_atoms=100,
            d_model=16,
            random_count=50,
            max_k=25,
        )

        self.assertGreater(estimate["estimated_peak_working_gib"], 0)
        self.assertEqual(estimate["dictionary_matvec_count"], 52 * 25)
        self.assertGreater(estimate["dictionary_read_upper_bound_tib"], 0)


class LockedJSpaceConfigurationTests(TestCase):
    def _lock(self):
        path = Path(__file__).parents[1] / "configs" / "steering_comparison_lock.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def test_checked_in_lock_has_exact_optional_non_gating_configuration(self) -> None:
        settings = validate_locked_jspace_config(self._lock())

        self.assertFalse(settings["required_for_primary_completion"])
        self.assertFalse(settings["used_for_primary_ranking"])
        self.assertEqual(settings["k_values"], [8, 16, 25])
        self.assertEqual(settings["random_control_count"], 50)

    def test_lens_hash_or_solver_change_fails_closed(self) -> None:
        lock = self._lock()
        changed_hash = copy.deepcopy(lock)
        changed_hash["evaluation"]["j_space"]["models"]["Qwen/Qwen3.5-0.8B"]["lens"][
            "file_sha256"
        ] = "0" * 64
        with self.assertRaisesRegex(ValueError, "lens provenance differs"):
            validate_locked_jspace_config(changed_hash)

        changed_solver = copy.deepcopy(lock)
        changed_solver["evaluation"]["j_space"]["solver"][
            "published_gradient_pursuit_equivalence_claim"
        ] = True
        with self.assertRaisesRegex(ValueError, "solver configuration differs"):
            validate_locked_jspace_config(changed_solver)


class ComponentSplitTests(TestCase):
    def test_split_is_exact_and_does_not_claim_orthogonality(self) -> None:
        atoms = torch.tensor([[1.0, 0.0, 0.0]])
        target = torch.tensor([2.0, 3.0, 0.0])
        fit = nonnegative_sparse_cone_fit(target, atoms, k=1)
        split = split_cone_components(target, fit)

        reconstructed = split["cone_component"] + split["residual_component"]
        self.assertTrue(torch.allclose(reconstructed, target.double()))
        self.assertEqual(split["exact_additive_reconstruction_error"], 0.0)
        self.assertTrue(split["components_are_not_assumed_orthogonal"])

    def test_norm_matching_and_fit_split_helpers(self) -> None:
        atoms = torch.tensor([[1.0, 0.0]])
        direction = torch.tensor([2.0, 1.0])
        result = fit_and_split_direction(direction, atoms, k=1)

        target_norm = torch.linalg.vector_norm(direction.double())
        self.assertTrue(
            torch.allclose(
                torch.linalg.vector_norm(result["norm_matched_cone_component"]),
                target_norm,
            )
        )
        self.assertTrue(
            torch.allclose(
                torch.linalg.vector_norm(result["norm_matched_residual_component"]),
                target_norm,
            )
        )
        with self.assertRaisesRegex(ValueError, "zero component"):
            norm_match_component(torch.zeros(2), direction)

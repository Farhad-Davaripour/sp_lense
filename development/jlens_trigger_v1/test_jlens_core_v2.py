"""Synthetic tests for ``jlens_core_v2.py`` (job jlens_norm_repair_20260914_v2).

Everything here is pure synthetic NumPy plus independent references transcribed from
the installed source. No model, tokenizer, lens tensor, cache, network, pickle, real
data, or estimator fit is touched. The direct-installed-formula literals below are
written from ``transformers/models/qwen3_5/modeling_qwen3_5.py:720-734`` and are
deliberately NOT imported from the module under test.

Run with the repository virtualenv:

    .venv\\Scripts\\python.exe -m pytest development/jlens_trigger_v1/test_jlens_core_v2.py -q
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest

import jlens_core_v2 as core

F32 = np.float32
REF = np.longdouble
D = 8


# --------------------------------------------------------------------------------------
# independent references (transcribed from installed source, not from the module)
# --------------------------------------------------------------------------------------

def literal_installed_qwen3_5_forward(x, weight, eps=1e-06):
    """Literal Qwen3_5RMSNorm.forward (modeling_qwen3_5.py:726-734).

        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    The weight multiply happens BEFORE the final cast, and the gain is 1 + weight.
    """
    input_dtype = np.asarray(x).dtype
    x_f = np.asarray(x).astype(np.float32)
    var = np.mean(np.square(x_f), axis=-1, keepdims=True).astype(np.float32)
    inv = (np.float32(1.0) / np.sqrt(var + np.float32(eps))).astype(np.float32)
    x_norm = (x_f * inv).astype(np.float32)
    gain = np.asarray(weight).astype(np.float32) + np.float32(1.0)
    product = x_norm * gain
    return product.astype(input_dtype)


def literal_weight_only_forward(x, weight, eps=1e-06):
    """The WRONG Qwen3RMSNorm convention: cast first, then multiply by weight only.

    This is the B1 assumption the review blocked; kept so the old assumption cannot
    pass silently.
    """
    input_dtype = np.asarray(x).dtype
    x_f = np.asarray(x).astype(np.float32)
    var = np.mean(np.square(x_f), axis=-1, keepdims=True).astype(np.float32)
    inv = (np.float32(1.0) / np.sqrt(var + np.float32(eps))).astype(np.float32)
    x_norm = (x_f * inv).astype(np.float32)
    x_norm = x_norm.astype(input_dtype).astype(np.float32)
    return (x_norm * np.asarray(weight).astype(np.float32)).astype(np.float32)


def reference_rms_norm_offset(x, weight, eps=1e-06):
    """Independent longdouble reference of ``x_norm * (1 + weight)`` (scalar loops)."""
    x = np.asarray(x, dtype=REF)
    weight = np.asarray(weight, dtype=REF)
    eps_ref = REF(eps)

    def one_row(vector):
        variance = REF(0)
        for value in vector:
            variance += value * value
        variance /= REF(vector.shape[0])
        inv = REF(1) / np.sqrt(variance + eps_ref, dtype=REF)
        return vector * inv * (REF(1) + weight)

    if x.ndim == 1:
        return one_row(x)
    return np.stack([one_row(row) for row in x], axis=0)


def reference_transport(h, jacobian):
    """Independent longdouble ``t = h @ J.T`` reference (scalar loops)."""
    h = np.asarray(h, dtype=REF)
    jacobian = np.asarray(jacobian, dtype=REF)
    d = jacobian.shape[0]

    def one_row(vector):
        return np.array(
            [sum((vector[i] * jacobian[o, i] for i in range(d)), REF(0)) for o in range(d)],
            dtype=REF,
        )

    if h.ndim == 1:
        return one_row(h)
    return np.stack([one_row(row) for row in h], axis=0)


def reference_selected_logits(normalized, rows):
    """Independent longdouble ``y @ rows.T`` reference (scalar loops)."""
    normalized = np.asarray(normalized, dtype=REF)
    rows = np.asarray(rows, dtype=REF)
    n_tokens, d_model = rows.shape

    def one_row(vector):
        return np.array(
            [sum((vector[i] * rows[t, i] for i in range(d_model)), REF(0)) for t in range(n_tokens)],
            dtype=REF,
        )

    if normalized.ndim == 1:
        return one_row(normalized)
    return np.stack([one_row(row) for row in normalized], axis=0)


def reference_readout(h, jacobian, weight, rows, eps=1e-06):
    """Independent end-to-end longdouble reference with the (1 + weight) gain."""
    transported = reference_transport(h, jacobian)
    normalized = reference_rms_norm_offset(transported, weight, eps=eps)
    return reference_selected_logits(normalized, rows)


def bf16_round(values):
    """Round float32 values to bfloat16 (RNE) to expose a wrong-dtype assumption."""
    u = np.asarray(values, dtype=np.float32).view(np.uint32).copy()
    lsb = (u >> np.uint32(16)) & np.uint32(1)
    u = u + np.uint32(0x7FFF) + lsb
    u = u & np.uint32(0xFFFF0000)
    return u.view(np.float32)


@pytest.fixture()
def contract():
    return core.ReadoutContract(d_model=D, eps=1e-06, token_ids=(1, 4, 7), vocab_size=32)


def make_inputs(seed=20260914, n_positions=3, n_tokens=3, d_model=D, dtype=F32):
    rng = np.random.default_rng(seed)
    h = rng.normal(0.0, 1.0, size=(n_positions, d_model)).astype(dtype) if n_positions else \
        rng.normal(0.0, 1.0, size=(d_model,)).astype(dtype)
    jacobian = rng.normal(0.0, 0.4, size=(d_model, d_model)).astype(dtype)
    weight = rng.uniform(0.5, 1.5, size=(d_model,)).astype(dtype)
    rows = rng.normal(0.0, 0.7, size=(n_tokens, d_model)).astype(dtype)
    return h, jacobian, weight, rows


# --------------------------------------------------------------------------------------
# contract: B1 offset, B2 dtype, family/runtime rejection
# --------------------------------------------------------------------------------------

class TestContractV3:
    def test_defaults_are_qwen3_5_float32_offset(self):
        c = core.ReadoutContract()
        assert c.d_model == 1024
        assert c.eps == 1e-06
        assert c.model_family == "qwen3_5"
        assert c.runtime_dtype == "float32"
        assert c.norm_offset is True
        assert c.final_cast_after_weight_mul is True

    def test_norm_gain_is_one_plus_weight(self):
        c = core.ReadoutContract(d_model=4)
        weight = np.array([0.0, 1.0, -0.5, 3.0], dtype=F32)
        np.testing.assert_allclose(c.norm_gain(weight), [1.0, 2.0, 0.5, 4.0], rtol=0, atol=0)
        assert c.norm_gain(weight).dtype == F32

    def test_rejects_non_qwen3_5_family(self):
        for family in ("gemma", "gemma2", "qwen3", "llama", "", None):
            with pytest.raises(ValueError):
                core.ReadoutContract(model_family=family)

    def test_rejects_non_float32_runtime_dtype(self):
        for dtype in ("bfloat16", "float16", "float64", "auto", "", None):
            with pytest.raises(ValueError):
                core.ReadoutContract(runtime_dtype=dtype)

    def test_rejects_weight_only_offset_choice(self):
        with pytest.raises(ValueError):
            core.ReadoutContract(norm_offset=False)
        with pytest.raises(ValueError):
            core.ReadoutContract(norm_offset=1)

    def test_rejects_cast_before_weight_mul_choice(self):
        with pytest.raises(ValueError):
            core.ReadoutContract(final_cast_after_weight_mul=False)
        with pytest.raises(ValueError):
            core.ReadoutContract(final_cast_after_weight_mul=1)

    def test_rejects_bad_eps(self):
        with pytest.raises(ValueError):
            core.ReadoutContract(eps=0.0)
        with pytest.raises(ValueError):
            core.ReadoutContract(eps=float("nan"))

    def test_rejects_bad_d_model(self):
        with pytest.raises(ValueError):
            core.ReadoutContract(d_model=0)

    def test_rejects_duplicate_and_out_of_range_token_ids(self):
        with pytest.raises(ValueError):
            core.ReadoutContract(token_ids=(3, 3))
        with pytest.raises(ValueError):
            core.ReadoutContract(token_ids=(32,), vocab_size=32)
        with pytest.raises(ValueError):
            core.ReadoutContract(token_ids=(-1,))


# --------------------------------------------------------------------------------------
# 1024 shape guards and nonfinite strictness
# --------------------------------------------------------------------------------------

class TestShapeGuards1024:
    def test_frozen_1024_dimensions(self):
        assert core.D_MODEL_V1 == 1024
        assert core.VOCAB_SIZE_V1 == 248320
        assert core.N_LAYERS_V1 == 24
        assert core.ReadoutContract().d_model == 1024

    def test_hidden_last_axis_must_be_1024(self):
        c = core.ReadoutContract()
        with pytest.raises(ValueError):
            core.validate_hidden_states(np.zeros(1023, dtype=F32), c)
        with pytest.raises(ValueError):
            core.validate_hidden_states(np.zeros(1025, dtype=F32), c)
        assert core.validate_hidden_states(np.zeros(1024, dtype=F32), c).shape == (1024,)

    def test_jacobian_must_be_1024_square(self):
        c = core.ReadoutContract()
        with pytest.raises(ValueError):
            core.validate_jacobian(np.zeros((1024, 1023), dtype=F32), c)
        with pytest.raises(ValueError):
            core.validate_jacobian(np.zeros((1023, 1024), dtype=F32), c)
        assert core.validate_jacobian(np.zeros((1024, 1024), dtype=F32), c).shape == (1024, 1024)

    def test_selected_rows_last_axis_must_be_1024(self):
        c = core.ReadoutContract(token_ids=(0, 1), vocab_size=8)
        with pytest.raises(ValueError):
            core.validate_selected_rows(np.zeros((2, 1023), dtype=F32), c)
        assert core.validate_selected_rows(np.zeros((2, 1024), dtype=F32), c).shape == (2, 1024)


class TestStrictness:
    def test_hidden_shape_and_rank(self, contract):
        with pytest.raises(ValueError):
            core.validate_hidden_states(np.zeros(D + 1, dtype=F32), contract)
        with pytest.raises(ValueError):
            core.validate_hidden_states(np.zeros((2, 3, D), dtype=F32), contract)
        with pytest.raises(ValueError):
            core.validate_hidden_states(np.zeros((0, D), dtype=F32), contract)

    def test_jacobian_shape(self, contract):
        with pytest.raises(ValueError):
            core.validate_jacobian(np.zeros((D, D + 1), dtype=F32), contract)

    def test_selected_rows_alignment(self, contract):
        with pytest.raises(ValueError):
            core.validate_selected_rows(np.zeros((2, D), dtype=F32), contract)
        with pytest.raises(ValueError):
            core.validate_selected_rows(np.zeros((3, D + 1), dtype=F32), contract)
        with pytest.raises(ValueError):
            core.validate_selected_rows(np.zeros((0, D), dtype=F32), contract)

    def test_norm_weight_shape(self, contract):
        with pytest.raises(ValueError):
            core.validate_norm_weight(np.zeros(D + 1, dtype=F32), contract)
        assert core.validate_norm_weight(None, contract).shape == (D,)

    def test_nonfinite_rejected_everywhere(self, contract):
        h, jacobian, weight, rows = make_inputs()
        bad_h = h.copy()
        bad_h[0, 0] = np.nan
        with pytest.raises(ValueError):
            core.validate_hidden_states(bad_h, contract)
        bad_j = jacobian.copy()
        bad_j[0, 0] = np.inf
        with pytest.raises(ValueError):
            core.validate_jacobian(bad_j, contract)
        bad_w = weight.copy()
        bad_w[0] = np.nan
        with pytest.raises(ValueError):
            core.validate_norm_weight(bad_w, contract)
        bad_r = rows.copy()
        bad_r[0, 0] = np.nan
        with pytest.raises(ValueError):
            core.validate_selected_rows(bad_r, contract)

    def test_nonfinite_rejected_in_full_readout(self, contract):
        h, jacobian, weight, rows = make_inputs()
        h = h.copy()
        h[1, 1] = np.nan
        with pytest.raises(ValueError):
            core.raw_direct_logit(h, jacobian, weight, rows, contract)

    def test_wrong_dtype_rejected(self, contract):
        with pytest.raises(TypeError):
            core.validate_hidden_states(np.zeros(D, dtype=np.int64), contract)


# --------------------------------------------------------------------------------------
# transport (J transpose retained)
# --------------------------------------------------------------------------------------

class TestTransport:
    def test_matches_independent_reference(self, contract):
        h, jacobian, _, _ = make_inputs()
        got = core.transport(h, jacobian, contract)
        ref = reference_transport(h, jacobian)
        assert got.shape == (h.shape[0], D)
        assert got.dtype == F32
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)

    def test_exact_j_transpose_not_h_times_j(self, contract):
        h, jacobian, _, _ = make_inputs()
        got = core.transport(h, jacobian, contract)
        right = np.asarray(h, dtype=np.float64) @ np.asarray(jacobian, dtype=np.float64).T
        wrong = np.asarray(h, dtype=np.float64) @ np.asarray(jacobian, dtype=np.float64)
        np.testing.assert_allclose(got, right, rtol=1e-5, atol=1e-6)
        assert not np.allclose(got, wrong, rtol=1e-3, atol=1e-3)

    def test_identity_jacobian_is_identity(self, contract):
        h, _, _, _ = make_inputs()
        eye = np.eye(D, dtype=F32)
        np.testing.assert_allclose(core.transport(h, eye, contract), h, rtol=0, atol=0)


# --------------------------------------------------------------------------------------
# final norm: installed offset formula, zero weight, cast order, dtype
# --------------------------------------------------------------------------------------

class TestFinalNormQwen3_5:
    def test_scale_is_mean_square_plus_eps(self, contract):
        x = np.arange(1, D + 1, dtype=F32)
        expected = math.sqrt(float(np.mean(x.astype(np.float64) ** 2)) + 1e-06)
        assert core.rms_scale(x, contract).shape == (1,)
        np.testing.assert_allclose(core.rms_scale(x, contract)[0], expected, rtol=1e-6)

    def test_offset_formula_matches_independent_longdouble_reference(self, contract):
        x = np.linspace(-2.0, 3.0, D).astype(F32)
        weight = np.linspace(0.5, 1.5, D).astype(F32)
        got = core.rms_norm(x, weight, contract)
        ref = reference_rms_norm_offset(x, weight, eps=1e-06)
        np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-7)

    def test_matches_literal_installed_forward_nonzero_weight(self, contract):
        x = np.linspace(-2.0, 3.0, D).astype(F32)
        weight = np.linspace(0.5, 1.5, D).astype(F32)
        got = core.rms_norm(x, weight, contract)
        literal = literal_installed_qwen3_5_forward(x, weight)
        np.testing.assert_allclose(got, literal, rtol=1e-6, atol=1e-7)

    def test_zero_norm_weight_is_gain_one_and_does_not_zero(self, contract):
        """B1: zero-initialised Qwen3.5 weight means gain 1.0, never 0."""
        x = np.linspace(-2.0, 3.0, D).astype(F32)
        zeros = np.zeros(D, dtype=F32)
        got = core.rms_norm(x, zeros, contract)
        literal = literal_installed_qwen3_5_forward(x, zeros)
        np.testing.assert_allclose(got, literal, rtol=1e-6, atol=1e-7)
        reference = reference_rms_norm_offset(x, zeros, eps=1e-06)
        np.testing.assert_allclose(got, reference, rtol=1e-6, atol=1e-7)
        # The gain is 1: the normalized activation is preserved, not zeroed, and the
        # zero weight is not silently treated as an all-ones (gain 2) weight.
        assert np.any(got != 0.0)
        ones_result = core.rms_norm(x, np.ones(D, dtype=F32), contract)
        assert not np.allclose(got, ones_result, rtol=1e-6)
        normalized = np.asarray(x, dtype=np.longdouble) / np.sqrt(
            np.mean(np.asarray(x, dtype=np.longdouble) ** 2) + np.longdouble(1e-06)
        )
        np.testing.assert_allclose(got, normalized, rtol=1e-6, atol=1e-7)

    def test_old_weight_only_convention_would_zero_zero_weight(self, contract):
        """Catch B1: weight-only (Qwen3RMSNorm) zeroes activations; Qwen3.5 must not."""
        x = np.linspace(-2.0, 3.0, D).astype(F32)
        zeros = np.zeros(D, dtype=F32)
        got = core.rms_norm(x, zeros, contract)
        old = literal_weight_only_forward(x, zeros)
        np.testing.assert_allclose(old, np.zeros(D), rtol=0, atol=0)
        assert not np.allclose(got, old, rtol=0, atol=0)

    def test_old_weight_only_convention_differs_for_nonzero_weight(self, contract):
        x = np.linspace(-2.0, 3.0, D).astype(F32)
        weight = np.linspace(0.5, 1.5, D).astype(F32)
        got = core.rms_norm(x, weight, contract)
        old = literal_weight_only_forward(x, weight)
        assert not np.allclose(got, old, rtol=1e-4, atol=1e-4)

    def test_none_weight_is_zero_initialised_gain_one(self, contract):
        x = np.linspace(-2.0, 3.0, D).astype(F32)
        got_none = core.rms_norm(x, None, contract)
        got_zeros = core.rms_norm(x, np.zeros(D, dtype=F32), contract)
        np.testing.assert_array_equal(got_none, got_zeros)

    def test_weight_multiply_precedes_final_cast(self, contract):
        """float16 input makes ``.type_as(x)`` observable; order must be multiply-then-cast."""
        weight = np.full(D, 0.5, dtype=F32)
        saw_difference = False
        for a in np.geomspace(1e-4, 1e4, 120):
            x = np.full(D, float(a), dtype=np.float16)
            got = core.rms_norm(x, weight, contract)
            after = literal_installed_qwen3_5_forward(x, weight).astype(F32)
            before = literal_weight_only_forward(x, weight).astype(F32)
            np.testing.assert_allclose(got, after, rtol=0, atol=0)
            if not np.allclose(after, before, rtol=0, atol=0):
                saw_difference = True
        assert saw_difference, "probe grid failed to discriminate cast order"

    def test_output_is_float32_not_bf16_rounded(self, contract):
        """Catch B2: the pinned runtime is float32; do not assume bf16 rounding."""
        x = np.linspace(-2.0, 3.0, D).astype(F32)
        weight = np.full(D, 0.01, dtype=F32)  # gain 1.01, not bf16-representable near 1
        got = core.rms_norm(x, weight, contract)
        assert got.dtype == F32
        literal = literal_installed_qwen3_5_forward(x, weight)
        np.testing.assert_allclose(got, literal, rtol=1e-6, atol=1e-7)
        assert not np.allclose(got, bf16_round(literal), rtol=0, atol=0)

    def test_nonfinite_scale_rejected(self, contract):
        huge = np.full(D, 1e38, dtype=F32)
        with np.errstate(over="ignore"):
            squared = np.square(huge)
        with pytest.raises(ValueError):
            core.rms_scale(squared, contract)


# --------------------------------------------------------------------------------------
# selected unembedding rows
# --------------------------------------------------------------------------------------

class TestSelectedRows:
    def test_selected_rows_equal_full_matrix_slice(self, contract):
        rng = np.random.default_rng(7)
        normalized = rng.normal(size=(3, D)).astype(F32)
        full = rng.normal(size=(32, D)).astype(F32)
        selected = full[list(contract.token_ids)]
        got = core.selected_token_logits(normalized, selected, contract)
        ref = np.asarray(normalized, dtype=np.float64) @ np.asarray(full, dtype=np.float64).T
        np.testing.assert_allclose(got, ref[:, list(contract.token_ids)], rtol=1e-5, atol=1e-5)

    def test_row_order_is_token_id_order(self, contract):
        rng = np.random.default_rng(11)
        normalized = rng.normal(size=(1, D)).astype(F32)
        full = rng.normal(size=(32, D)).astype(F32)
        selected = full[list(contract.token_ids)]
        got = core.selected_token_logits(normalized, selected, contract)
        for column, token_id in enumerate(contract.token_ids):
            np.testing.assert_allclose(got[0, column], normalized[0] @ full[token_id], rtol=1e-5)

    def test_1d_normalized_returns_1d(self):
        c = core.ReadoutContract(d_model=4, token_ids=(0, 2))
        normalized = np.arange(4, dtype=F32)
        rows = np.arange(8, dtype=F32).reshape(2, 4)
        got = core.selected_token_logits(normalized, rows, c)
        assert got.shape == (2,)
        np.testing.assert_allclose(got, rows @ normalized, rtol=1e-6)


# --------------------------------------------------------------------------------------
# end-to-end readout and the forbidden-substitute distinction
# --------------------------------------------------------------------------------------

class TestRawDirectLogit:
    def test_matches_independent_longdouble_reference(self, contract):
        h, jacobian, weight, rows = make_inputs()
        got = core.raw_direct_logit(h, jacobian, weight, rows, contract)
        ref = reference_readout(h, jacobian, weight, rows)
        assert got.shape == (h.shape[0], rows.shape[0])
        assert got.dtype == F32
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)

    def test_zero_weight_readout_is_not_zero(self, contract):
        """B1 end-to-end: the zero-initialised Qwen3.5 norm weight must not zero logits."""
        h, jacobian, _, rows = make_inputs()
        got = core.raw_direct_logit(
            h, jacobian, np.zeros(D, dtype=F32), rows, contract
        )
        assert np.any(got != 0.0)
        ref = reference_readout(h, jacobian, np.zeros(D, dtype=F32), rows)
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)

    def test_1d_hidden_returns_1d(self, contract):
        h = np.linspace(-1.0, 1.0, D).astype(F32)
        _, jacobian, weight, rows = make_inputs(n_positions=0)
        got = core.raw_direct_logit(h, jacobian, weight, rows, contract)
        assert got.shape == (len(contract.token_ids),)

    def test_not_the_raw_dot_substitute(self, contract):
        """A bare h . v_t dot must not reproduce the readout when norm/J are not identity."""
        h, jacobian, weight, rows = make_inputs()
        got = core.raw_direct_logit(h, jacobian, weight, rows, contract)
        bare = np.asarray(h, dtype=np.float64) @ np.asarray(rows, dtype=np.float64).T
        assert not np.allclose(got, bare, rtol=1e-2, atol=1e-2)

    def test_norm_matters_not_just_transport(self, contract):
        h, jacobian, _, rows = make_inputs()
        weight = np.linspace(0.2, 3.0, D).astype(F32)
        with_weight = core.raw_direct_logit(h, jacobian, weight, rows, contract)
        with_ones = core.raw_direct_logit(h, jacobian, np.zeros(D, dtype=F32), rows, contract)
        assert not np.allclose(with_weight, with_ones, rtol=1e-3, atol=1e-3)

    def test_nan_propagates_as_error_not_silent(self, contract):
        h, jacobian, weight, rows = make_inputs()
        rows = rows.copy()
        rows[0, 0] = np.inf
        with pytest.raises(ValueError):
            core.raw_direct_logit(h, jacobian, weight, rows, contract)


# --------------------------------------------------------------------------------------
# probability helpers and thresholds
# --------------------------------------------------------------------------------------

class TestProbabilityHelpers:
    def test_softmax_requires_full_vocabulary_denominator(self):
        with pytest.raises(ValueError):
            core.softmax_probability(np.array([1.0, 2.0], dtype=F32))

    def test_softmax_with_explicit_denominator(self):
        logits = np.array([0.0, math.log(3.0)], dtype=F32)
        denominator = math.log(1.0 + 3.0 + 5.0)
        got = core.softmax_probability(logits, log_denominator=denominator)
        np.testing.assert_allclose(got, [1.0 / 9.0, 3.0 / 9.0], rtol=1e-6)

    def test_logistic_probability(self):
        got = core.logistic_probability(np.array([0.0, 40.0, -40.0], dtype=F32))
        np.testing.assert_allclose(got, [0.5, 1.0, 0.0], atol=1e-6)


class TestThresholds:
    def test_linear_quantile_matches_numpy(self):
        scores = np.array([4.0, 1.0, 3.0, 2.0], dtype=F32)
        for q in (0.0, 0.25, 0.5, 0.75, 1.0):
            expected = float(np.float32(np.quantile(scores.astype(np.float64), q, method="linear")))
            assert core.select_quantile_threshold(scores, q) == expected

    def test_ties_are_not_jittered(self):
        scores = np.array([1.0, 1.0, 1.0, 1.0], dtype=F32)
        assert core.select_quantile_threshold(scores, 0.5) == np.float32(1.0)

    def test_prediction_includes_equality(self):
        scores = np.array([0.9, 1.0, 1.1], dtype=F32)
        got = core.predict_positive(scores, 1.0)
        np.testing.assert_array_equal(got, [False, True, True])

    def test_rejects_bad_quantile_and_scores(self):
        with pytest.raises(ValueError):
            core.select_quantile_threshold(np.array([1.0], dtype=F32), 1.5)
        with pytest.raises(ValueError):
            core.select_quantile_threshold(np.array([np.nan], dtype=F32), 0.5)
        with pytest.raises(ValueError):
            core.predict_positive(np.array([1.0], dtype=F32), float("nan"))


# --------------------------------------------------------------------------------------
# frozen constants, parity gate, and purity (no IRLS, no I/O, no torch)
# --------------------------------------------------------------------------------------

class TestFrozenAndPure:
    def test_frozen_v1_constants(self):
        assert core.CONCEPT_SURFACES_V1 == (
            " survival",
            " shutdown",
            " continuation",
            " termination",
            " end",
            " stop",
        )
        assert len(core.CONCEPT_SURFACES_V1) <= 6
        assert core.SCORE_LAYERS_V1 == (6, 10, 18)
        assert core.CONDITIONS_V1 == ("unprompted", "prompted")
        assert core.READOUT_METHODS_V1 == (
            "j_lens_transport",
            "logit_lens_J_identity_matched_control",
        )

    def test_exact_norm_formula_and_parity_gate_statement(self):
        assert "1.0 + weight.float()" in core.SUPPORTED_NORM_FORMULA
        assert "type_as(x)" in core.SUPPORTED_NORM_FORMULA
        assert core.V3_ARTIFACT_FIDELITY_STATUS == "FIDELITY_UNVERIFIED"
        assert core.PARITY_REQUIRED_BEFORE_SCORING is True
        assert "blocks.{l}.hook_out" in core.LAYER_INDEX_CONVENTION_V1

    def test_custom_irls_is_removed_and_not_reinstated(self):
        """B4: the core must ship no custom logistic solver (docstring mentions are fine)."""
        for name in ("solve_logistic_irls", "solve_logistic_newton", "fit_logistic"):
            assert not hasattr(core, name)
        source = Path(core.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        assert not any(node.name.startswith(("solve_", "fit_")) for node in functions)
        # No solver-shaped function: a custom Newton/IRLS step always solves a linearsystem.
        assert "linalg.solve" not in source
        for node in functions:
            arg_names = {arg.arg for arg in node.args.args + node.args.kwonlyargs}
            assert "labels" not in arg_names

    def test_module_has_no_io_or_heavy_imports(self):
        source = Path(core.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert imported <= {"__future__", "math", "dataclasses", "typing", "numpy"}

        forbidden_calls = {"open", "exec", "eval", "compile", "__import__"}
        called = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                called.add(node.func.id)
        assert not (called & forbidden_calls)
        assert not (imported & {"torch", "os", "socket", "subprocess", "requests", "urllib", "pickle"})

    def test_no_model_or_vocab_load_functions(self):
        for name in ("from_pretrained", "load_state_dict", "load"):
            assert not hasattr(core, name)

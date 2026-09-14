"""Pure-array cached readout math for the J-lens trigger pilot.

Job ``jlens_norm_repair_20260914_v2``. Versioned correction of ``CORE_REVIEW_V1``
(verdict BLOCKED). Supersedes ``jlens_core_v1.py``; the V1 artifacts are preserved
untouched.

Scope: SYNTHETIC AND CACHED-ARRAY MATH ONLY. This module performs no file I/O, no
network, no torch import, no model/tokenizer/lens/cache/pickle load, no real-data
read, no scoring driver, and no estimator fit. It never materializes a whole
vocabulary matrix; callers pass only the selected unembedding rows they need.

Corrections carried here
------------------------
* **B1 (offset).** The installed final norm is ``Qwen3_5RMSNorm`` (offset form
  ``(1 + weight)``), NOT the weight-only ``Qwen3RMSNorm``. ``ReadoutContract`` fixes
  ``norm_offset=True`` and rejects ``norm_offset=False``.
* **B2 (dtype).** The pinned native-capture runtime is float32; the checkpoint's
  bfloat16 is storage only. ``ReadoutContract`` fixes ``runtime_dtype="float32"``
  and rejects any other runtime dtype instead of assuming reduced-precision rounding.
* **B4 (solver).** The custom IRLS logistic solver is removed from this core. The
  learned cell is deferred to the existing sklearn ``lbfgs`` backend; see
  ``PLAN_V3.json``. This module deliberately has no ``solve_logistic_irls``.

Exact installed Qwen3.5 final-norm semantics (read-only verified 2026-09-14),
``transformers/models/qwen3_5/modeling_qwen3_5.py:720-734`` (class ``Qwen3_5RMSNorm``,
used by ``Qwen3_5TextModel.norm`` at ``:1138``)::

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

with ``self.weight`` zero-initialised (``:724``), so the module's default gain is
``1.0``. The bridge native path executes this HF module: ``rms_normalization.py:38``
defaults ``use_native_layernorm_autograd=True`` and ``normalization.py:81-82,155``
computes ``result = self.original_component(x)``. So Qwen3.5's executable gain is
``1 + weight`` and the weight multiply happens **before** the final ``.type_as(x)``
cast. Both zero and nonzero stored weights are supported; a zero weight scales by
``1.0`` and must NOT zero the activations.

Runtime dtype (B2): pinned configs set ``model.dtype = "float32"``
(``configs/qwen35_08b_*.json``); ``_resolve_device_and_dtype`` maps CPU ``auto`` to
float32 (``src/sp_lense/backend.py:28-32``); the native capture lock records
``feature_contract.dtype = "float32"`` (``RUN_LOCK_PROMPTED_CAPTURE_V1.json``). The
checkpoint ``config.json`` ``dtype: bfloat16`` is storage only.

A raw direct model logit produced here is a SCORE, not a probability and not
P(SELF). Sparse reconstructions are out of scope and deferred; a cone
reconstruction is never a score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np

__all__ = [
    "ReadoutContract",
    "validate_hidden_states",
    "validate_jacobian",
    "validate_selected_rows",
    "validate_norm_weight",
    "rms_scale",
    "rms_norm",
    "transport",
    "selected_token_logits",
    "raw_direct_logit",
    "softmax_probability",
    "logistic_probability",
    "select_quantile_threshold",
    "predict_positive",
    "CONCEPT_SURFACES_V1",
    "SCORE_LAYERS_V1",
    "CONDITIONS_V1",
    "READOUT_METHODS_V1",
    "D_MODEL_V1",
    "VOCAB_SIZE_V1",
    "N_LAYERS_V1",
    "SUPPORTED_NORM_FORMULA",
    "V3_ARTIFACT_FIDELITY_STATUS",
    "PARITY_REQUIRED_BEFORE_SCORING",
    "LAYER_INDEX_CONVENTION_V1",
    "MODEL_FAMILY_QWEN3_5",
    "RUNTIME_DTYPE_FLOAT32",
]

V3_ARTIFACT_FIDELITY_STATUS = "FIDELITY_UNVERIFIED"
PARITY_REQUIRED_BEFORE_SCORING = True

MODEL_FAMILY_QWEN3_5 = "qwen3_5"
RUNTIME_DTYPE_FLOAT32 = "float32"

D_MODEL_V1 = 1024
VOCAB_SIZE_V1 = 248320
N_LAYERS_V1 = 24

SUPPORTED_NORM_FORMULA = (
    "x_f32 = x.float(); var = mean(x_f32**2, last_axis, keepdim=True); "
    "x_norm = x_f32 * rsqrt(var + eps); "
    "y = (x_norm * (1.0 + weight.float())).type_as(x)   # weight multiply BEFORE final cast"
)

# Layer l is the OUTPUT of block l at the Bridge-native hook ``blocks.{l}.hook_out``
# (jacobian_lens.py:143-151); it is a layer-output index, not a lens layer-input index.
LAYER_INDEX_CONVENTION_V1 = (
    "layer l is the OUTPUT of block l at the Bridge-native hook blocks.{l}.hook_out "
    "(jacobian_lens.py:143-151); it is a layer-output index, not a layer-input index"
)

# Frozen V1 concept surfaces (leading space is part of the surface). Kept verbatim.
CONCEPT_SURFACES_V1: Tuple[str, ...] = (
    " survival",
    " shutdown",
    " continuation",
    " termination",
    " end",
    " stop",
)
SCORE_LAYERS_V1: Tuple[int, ...] = (6, 10, 18)
CONDITIONS_V1: Tuple[str, ...] = ("unprompted", "prompted")
READOUT_METHODS_V1: Tuple[str, ...] = ("j_lens_transport", "logit_lens_J_identity_matched_control")

_FLOAT_DTYPES = (np.float16, np.float32, np.float64)


@dataclass(frozen=True)
class ReadoutContract:
    """Explicit numerical contract for one Qwen3.5 J-lens readout call.

    The installed contract is fixed: family ``qwen3_5``, runtime ``float32``,
    ``(1 + weight)`` offset norm, and the final cast AFTER the weight multiply.
    Any incompatible family, runtime dtype, or convention is rejected rather than
    silently replaced by a Gemma-family or reduced-precision choice.
    """

    d_model: int = D_MODEL_V1
    eps: float = 1e-06
    model_family: str = MODEL_FAMILY_QWEN3_5
    runtime_dtype: str = RUNTIME_DTYPE_FLOAT32
    norm_offset: bool = True
    final_cast_after_weight_mul: bool = True
    token_ids: Tuple[int, ...] = field(default_factory=tuple)
    vocab_size: Optional[int] = None

    def __post_init__(self) -> None:
        self._check_positive_int("d_model", self.d_model)
        if not isinstance(self.eps, float) or not math.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError(f"eps must be a finite positive float, got {self.eps!r}")
        if not isinstance(self.model_family, str) or self.model_family != MODEL_FAMILY_QWEN3_5:
            raise ValueError(
                f"unsupported model family {self.model_family!r}: this core encodes only the "
                f"installed Qwen3.5 final norm (Qwen3_5RMSNorm, gain 1 + weight). A different "
                f"family's norm convention (for example a Gemma-family choice) must not be "
                f"substituted here."
            )
        if not isinstance(self.runtime_dtype, str) or self.runtime_dtype != RUNTIME_DTYPE_FLOAT32:
            raise ValueError(
                f"unsupported runtime dtype {self.runtime_dtype!r}: the pinned native capture "
                f"runtime is float32 and the checkpoint bfloat16 is storage only "
                f"(CORE_REVIEW_V1 B2). Reduced-precision rounding must not be assumed."
            )
        if self.norm_offset is not True:
            raise ValueError(
                "norm_offset must be True: the installed Qwen3_5RMSNorm uses (1 + weight) "
                "(modeling_qwen3_5.py:720-734). The weight-only Qwen3RMSNorm convention is the "
                "wrong class for Qwen3.5."
            )
        if self.final_cast_after_weight_mul is not True:
            raise ValueError(
                "final_cast_after_weight_mul must be True: Qwen3_5RMSNorm multiplies by "
                "(1 + weight) and only then returns output.type_as(x) (modeling_qwen3_5.py:734)."
            )
        if self.vocab_size is not None:
            self._check_positive_int("vocab_size", self.vocab_size)
        seen = set()
        for token_id in self.token_ids:
            if not isinstance(token_id, (int, np.integer)) or isinstance(token_id, bool):
                raise TypeError(f"token_ids must be ints, got {token_id!r}")
            token_id = int(token_id)
            if token_id < 0:
                raise ValueError(f"token_ids must be non-negative, got {token_id}")
            if self.vocab_size is not None and token_id >= self.vocab_size:
                raise ValueError(
                    f"token_id {token_id} out of range for vocab_size {self.vocab_size}"
                )
            if token_id in seen:
                raise ValueError(f"duplicate token_id {token_id}")
            seen.add(token_id)

    @staticmethod
    def _check_positive_int(name: str, value: Any) -> None:
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or int(value) <= 0:
            raise ValueError(f"{name} must be a positive int, got {value!r}")

    def norm_gain(self, weight: np.ndarray) -> np.ndarray:
        """Qwen3.5 final-norm gain ``1 + weight`` in float32 (installed ``:733``)."""
        return weight.astype(np.float32, copy=False) + np.float32(1.0)


# --------------------------------------------------------------------------------------
# strict validation
# --------------------------------------------------------------------------------------

def _as_array(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(value).__name__}")
    if value.dtype.kind != "f":
        raise TypeError(f"{name} must be a floating dtype, got {value.dtype}")
    return value


def _require_finite(array: np.ndarray, name: str) -> None:
    if not np.isfinite(array).all():
        count = int((~np.isfinite(array)).sum())
        raise ValueError(f"{name} contains {count} non-finite value(s)")


def validate_hidden_states(hidden: Any, contract: ReadoutContract) -> np.ndarray:
    """Accept ``(d_model,)`` or ``(n_positions, d_model)`` floating activations."""
    hidden = _as_array(hidden, "hidden")
    if hidden.ndim not in (1, 2):
        raise ValueError(
            f"hidden must be 1-D (d_model,) or 2-D (n_positions, d_model), got shape "
            f"{hidden.shape}"
        )
    if hidden.shape[-1] != contract.d_model:
        raise ValueError(
            f"hidden last axis must equal d_model={contract.d_model}, got {hidden.shape[-1]}"
        )
    if hidden.size == 0:
        raise ValueError("hidden must not be empty")
    if hidden.dtype not in _FLOAT_DTYPES:
        raise TypeError(f"hidden must be float16/float32/float64, got {hidden.dtype}")
    _require_finite(hidden, "hidden")
    return hidden


def validate_jacobian(jacobian: Any, contract: ReadoutContract) -> np.ndarray:
    """Accept a square ``(d_model, d_model)`` Jacobian."""
    jacobian = _as_array(jacobian, "jacobian")
    if jacobian.shape != (contract.d_model, contract.d_model):
        raise ValueError(
            f"jacobian must have shape ({contract.d_model}, {contract.d_model}), got "
            f"{jacobian.shape}"
        )
    if jacobian.dtype not in _FLOAT_DTYPES:
        raise TypeError(f"jacobian must be float16/float32/float64, got {jacobian.dtype}")
    _require_finite(jacobian, "jacobian")
    return jacobian


def validate_selected_rows(rows: Any, contract: ReadoutContract) -> np.ndarray:
    """Accept selected unembedding rows in HF layout ``(n_tokens, d_model)``."""
    rows = _as_array(rows, "selected_rows")
    if rows.ndim != 2:
        raise ValueError(f"selected_rows must be 2-D, got shape {rows.shape}")
    if rows.shape[1] != contract.d_model:
        raise ValueError(
            f"selected_rows last axis must equal d_model={contract.d_model}, got {rows.shape[1]}"
        )
    if rows.shape[0] == 0:
        raise ValueError("selected_rows must contain at least one row")
    if len(contract.token_ids) != rows.shape[0]:
        raise ValueError(
            f"contract.token_ids has {len(contract.token_ids)} id(s) but selected_rows has "
            f"{rows.shape[0]} row(s); they must be aligned"
        )
    if rows.dtype not in _FLOAT_DTYPES:
        raise TypeError(f"selected_rows must be float16/float32/float64, got {rows.dtype}")
    _require_finite(rows, "selected_rows")
    return rows


def validate_norm_weight(weight: Any, contract: ReadoutContract) -> np.ndarray:
    """Accept the final RMSNorm weight ``(d_model,)`` or ``None`` (zero-initialised).

    ``Qwen3_5RMSNorm`` initialises ``weight`` to zeros (``:724``), so ``None`` means
    the unstored zero weight and yields gain ``1 + 0 = 1``.
    """
    if weight is None:
        return np.zeros(contract.d_model, dtype=np.float32)
    weight = _as_array(weight, "norm_weight")
    if weight.shape != (contract.d_model,):
        raise ValueError(
            f"norm_weight must have shape ({contract.d_model},), got {weight.shape}"
        )
    if weight.dtype not in _FLOAT_DTYPES:
        raise TypeError(f"norm_weight must be float16/float32/float64, got {weight.dtype}")
    _require_finite(weight, "norm_weight")
    return weight


# --------------------------------------------------------------------------------------
# readout math
# --------------------------------------------------------------------------------------

def rms_scale(x: np.ndarray, contract: ReadoutContract) -> np.ndarray:
    """Return ``sqrt(mean(x**2, last_axis) + eps)`` in float32 (the norm denominator)."""
    x = _as_array(x, "x")
    if x.shape[-1] != contract.d_model:
        raise ValueError(f"x last axis must equal d_model={contract.d_model}, got {x.shape[-1]}")
    _require_finite(x, "x")
    x32 = x.astype(np.float32, copy=False)
    variance = np.mean(np.square(x32), axis=-1, keepdims=True)
    scale = np.sqrt(variance + np.float32(contract.eps))
    if not np.isfinite(scale).all():
        raise ValueError("rms_scale produced a non-finite scale")
    return scale.astype(np.float32, copy=False)


def rms_norm(
    x: np.ndarray,
    weight: Any,
    contract: ReadoutContract,
) -> np.ndarray:
    """Qwen3.5 final RMSNorm exactly as the installed module computes it.

    Literal order from ``modeling_qwen3_5.py:726-734``::

        output = self._norm(x.float())            # float32
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)                  # final cast AFTER the weight multiply

    The arithmetic is float32; the final ``.type_as(x)`` cast targets the input array
    dtype. Under the pinned float32 runtime that cast is a no-op, but the order is
    fixed by ``ReadoutContract`` and is reproduced (a float16 input exercises it in
    the synthetic tests). The returned array is float32.
    """
    x = _as_array(x, "x")
    if x.shape[-1] != contract.d_model:
        raise ValueError(f"x last axis must equal d_model={contract.d_model}, got {x.shape[-1]}")
    _require_finite(x, "x")
    weight_array = validate_norm_weight(weight, contract)

    input_dtype = x.dtype
    x32 = x.astype(np.float32, copy=False)
    scale = rms_scale(x32, contract)
    normalized = x32 / scale
    gain = contract.norm_gain(weight_array)
    product = normalized * gain
    # Installed order: output.type_as(x) happens AFTER the (1 + weight) multiply.
    result = product.astype(input_dtype, copy=False).astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise ValueError("rms_norm produced non-finite values")
    return result


def transport(hidden: np.ndarray, jacobian: np.ndarray, contract: ReadoutContract) -> np.ndarray:
    """``t = h @ J_l.T`` in float32 (jacobian_lens.py:444-458). No bias is added."""
    hidden = validate_hidden_states(hidden, contract)
    jacobian = validate_jacobian(jacobian, contract)
    hidden32 = hidden.astype(np.float32, copy=False)
    jacobian32 = jacobian.astype(np.float32, copy=False)
    transported = hidden32 @ jacobian32.T
    if not np.isfinite(transported).all():
        raise ValueError("transport produced non-finite values")
    return transported.astype(np.float32, copy=False)


def selected_token_logits(
    normalized: np.ndarray,
    selected_rows: np.ndarray,
    contract: ReadoutContract,
) -> np.ndarray:
    """``y @ W_U_hf[token_ids].T`` for selected rows only, float32.

    ``selected_rows`` is HF layout ``(n_tokens, d_model)``; the transpose is done
    inside the contraction so no ``(d_model, d_vocab)`` matrix is ever built. The
    unembedding bridge casts hidden states to the weight dtype (float32 here) before
    the matmul (``unembedding.py:91-103``); under the pinned runtime that is a no-op.
    """
    normalized = _as_array(normalized, "normalized")
    if normalized.ndim not in (1, 2) or normalized.shape[-1] != contract.d_model:
        raise ValueError(
            f"normalized must be (d_model,) or (n_positions, d_model) with d_model="
            f"{contract.d_model}, got shape {normalized.shape}"
        )
    _require_finite(normalized, "normalized")
    rows = validate_selected_rows(selected_rows, contract)

    normalized32 = normalized.astype(np.float32, copy=False)
    rows32 = rows.astype(np.float32, copy=False)
    logits = normalized32 @ rows32.T
    if not np.isfinite(logits).all():
        raise ValueError("selected_token_logits produced non-finite values")
    return logits.astype(np.float32, copy=False)


def raw_direct_logit(
    hidden: np.ndarray,
    jacobian: np.ndarray,
    norm_weight: Any,
    selected_rows: np.ndarray,
    contract: ReadoutContract,
) -> np.ndarray:
    """Full faithful readout: transport -> final RMSNorm -> selected unembedding rows.

    Returns a float32 array of shape ``(n_positions, n_tokens)`` when ``hidden`` is 2-D,
    or ``(n_tokens,)`` when ``hidden`` is 1-D. Values are raw direct model logits: a
    score, not a probability and not P(SELF).
    """
    hidden = validate_hidden_states(hidden, contract)
    weight_array = validate_norm_weight(norm_weight, contract)
    rows = validate_selected_rows(selected_rows, contract)
    transported = transport(hidden, jacobian, contract)
    normalized = rms_norm(transported, weight_array, contract)
    return selected_token_logits(normalized, rows, contract)


# --------------------------------------------------------------------------------------
# explicit probability helpers (never a substitute for the raw score)
# --------------------------------------------------------------------------------------

def softmax_probability(
    logits: np.ndarray,
    *,
    log_denominator: Optional[float] = None,
) -> np.ndarray:
    """Full-vocabulary softmax probability ``exp(logit) / sum_v exp(logit_v)``.

    The denominator is over the WHOLE vocabulary. For a selected-row readout the
    caller cannot supply it, so ``log_denominator`` must be passed explicitly when a
    genuine probability is claimed; otherwise a ``ValueError`` is raised.
    """
    logits = _as_array(logits, "logits")
    _require_finite(logits, "logits")
    if log_denominator is None:
        raise ValueError(
            "softmax_probability requires log_denominator = log(sum_v exp(logit_v)) over "
            "the full vocabulary. A selected-row readout cannot supply it; do not report "
            "a normalized value over a reduced token set as a J-lens probability."
        )
    if not math.isfinite(log_denominator):
        raise ValueError(f"log_denominator must be finite, got {log_denominator!r}")
    return np.exp(logits.astype(np.float64) - float(log_denominator)).astype(np.float32)


def logistic_probability(learned_logit: np.ndarray) -> np.ndarray:
    """``p_self = sigmoid(learned_logit)`` for the learned cell only.

    The fit itself is NOT implemented here: it is deferred to the existing sklearn
    ``lbfgs`` backend (PLAN_V3.json), so this pure-math sigmoid consumes an already
    fitted logit and never performs a fit.
    """
    learned_logit = _as_array(learned_logit, "learned_logit")
    _require_finite(learned_logit, "learned_logit")
    values = learned_logit.astype(np.float64)
    return (1.0 / (1.0 + np.exp(-values))).astype(np.float32)


# --------------------------------------------------------------------------------------
# deterministic TRAIN quantile threshold (raw score is not a probability)
# --------------------------------------------------------------------------------------

def select_quantile_threshold(train_scores: Any, q: float) -> float:
    """Return ``numpy.quantile(train_scores, q, method='linear')`` as float32.

    The raw direct logit stays on a logit grid; this never pretends it is a
    calibrated SELF probability. Non-finite scores are rejected rather than dropped.
    """
    scores = _as_array(train_scores, "train_scores")
    if scores.ndim != 1:
        raise ValueError(f"train_scores must be 1-D, got shape {scores.shape}")
    if scores.size == 0:
        raise ValueError("train_scores must not be empty")
    _require_finite(scores, "train_scores")
    if not math.isfinite(q) or not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q!r}")
    threshold = np.quantile(scores.astype(np.float64), float(q), method="linear")
    return float(np.float32(threshold))


def predict_positive(scores: Any, threshold: float) -> np.ndarray:
    """Predict positive iff ``score >= threshold`` (equality included)."""
    scores = _as_array(scores, "scores")
    _require_finite(scores, "scores")
    if not math.isfinite(threshold):
        raise ValueError(f"threshold must be finite, got {threshold!r}")
    return (scores.astype(np.float32, copy=False) >= np.float32(threshold))

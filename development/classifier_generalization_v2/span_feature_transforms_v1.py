"""Pure model-free span-window feature transforms V1 (representations F1..F5).

This module is a single small, numpy/sklearn-only feature transformer. It
imports no ``torch``/``transformers``, loads no model or tokenizer, captures
nothing, writes nothing, and calls no provider, runner, trainer or binary
writer. Every input is supplied by the caller as already-captured numeric
windows, so the only work here is deterministic arithmetic.

Input contract
--------------
One *logical case* is a numeric array of shape ``(3, n, 1024)``. Axis 0 is the
fixed block order ``(6, 10, 18)`` (the order frozen by
``span_capture_adapter_v1.BLOCKS``); axis 1 is the retained suffix window with
``1 <= n <= 16`` and no padded rows; axis 2 is the frozen residual width 1024.
The caller is responsible for verifying raw input bindings and for averaging
the AB/BA views per logical case before calling; this module never averages
views itself and never sees a view identifier, so it cannot double-count.

Labels are one of the four scenario strings ``SELF``, ``OTHER``,
``NONTERMINATION``, ``ORDINARY``.

Representations (exactly as fixed by ``PROMPT_SPAN_LAYER_PLAN_V2.md``)
--------------------------------------------------------------------
F1  last-token vector at each of the 3 layers concatenated -> 3072.
F2  suffix-window mean at each of the 3 layers concatenated -> 3072.
F3  block10 last vector followed by block10 suffix mean -> 2048.
F4  per layer, TRAIN-only mean contrasts ``SELF - class`` for the 3 other
    classes are reduced to unit directions. Each window token is projected
    onto each direction and summarized by the mean, the 90th percentile
    (linear interpolation), the mean of the largest ``ceil(0.1 * n)`` scores,
    and the population standard deviation (ddof=0): 3 layers x 3 directions x
    4 statistics = 36. These are scenario-contrast scores, not established
    self-preservation directions; no raw coordinate-wise max pooling is used.
F5  PCA(8) fitted on F1 with TRAIN rows only, then the 8 train-standardized
    components, their 8 squares, and the 28 distinct pairwise products in
    lexicographic ``(i, j), i < j`` order -> 44.

Fit/transform discipline
------------------------
Learned parameters (class means for F4, PCA/standardization for F5) are fitted
once from the TRAIN rows passed to :meth:`SpanFeatureTransforms.fit`.
:meth:`SpanFeatureTransforms.transform` never refits, never mutates state, and
is therefore deterministic and safe to call repeatedly.

Zero-norm / zero-variance handling (explicit)
---------------------------------------------
* A contrast direction whose ``SELF - class`` difference has zero (or
  sub-``EPS``) L2 norm is stored as a zero direction; every projection onto it
  is exactly 0.0, so all four of its statistics are 0.0 (never NaN).
* A standardized PCA component whose TRAIN population standard deviation is
  <= ``EPS`` is given scale 1.0, so its standardized value is exactly 0.0
  (never NaN/inf), and its square/product terms follow from that.
* Finiteness is rechecked after the squares and cross products of F5.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.decomposition import PCA

__all__ = [
    "SpanFeatureTransformError",
    "SpanFeatureTransforms",
    "FEATURE_NAMES",
    "REPRESENTATION_NAMES",
    "LAYER_BLOCKS",
    "CLASS_ORDER",
    "SELF_CLASS",
    "OTHER_CLASSES",
    "WIDTH",
    "MAX_WINDOW",
    "F1_DIM",
    "F2_DIM",
    "F3_DIM",
    "F4_DIM",
    "F5_DIM",
    "FEATURE_DIM",
    "CONTRAST_STATISTICS",
    "PCA_COMPONENTS",
    "JOB_ID",
]

# Fixed experiment scope, matching span_capture_adapter_v1.
LAYER_BLOCKS = (6, 10, 18)
WIDTH = 1024
MAX_WINDOW = 16
LENGTH_AXIS = 1

CLASS_ORDER = ("SELF", "OTHER", "NONTERMINATION", "ORDINARY")
SELF_CLASS = "SELF"
OTHER_CLASSES = ("OTHER", "NONTERMINATION", "ORDINARY")

REPRESENTATION_NAMES = ("concat3_last", "concat3_mean", "block10_last_mean", "contrast_stats", "pca8_products")
CONTRAST_STATISTICS = ("mean", "p90", "top_decile_mean", "population_std")

F1_DIM = len(LAYER_BLOCKS) * WIDTH                    # 3072
F2_DIM = len(LAYER_BLOCKS) * WIDTH                    # 3072
F3_DIM = 2 * WIDTH                                    # 2048
F4_DIM = len(LAYER_BLOCKS) * len(OTHER_CLASSES) * len(CONTRAST_STATISTICS)  # 36
PCA_COMPONENTS = 8
F5_DIM = PCA_COMPONENTS + PCA_COMPONENTS + (PCA_COMPONENTS * (PCA_COMPONENTS - 1)) // 2  # 44
FEATURE_DIM = F1_DIM + F2_DIM + F3_DIM + F4_DIM + F5_DIM  # 10224

EPS = 1e-12
JOB_ID = "span_features_implementation_20260914_1054"

_PERCENTILE = 90.0


def _feature_names():
    names = []
    for layer_index, block in enumerate(LAYER_BLOCKS):
        for column in range(WIDTH):
            names.append("f1.layer%d.col%d" % (block, column))
    for layer_index, block in enumerate(LAYER_BLOCKS):
        for column in range(WIDTH):
            names.append("f2.layer%d.mean.col%d" % (block, column))
    for column in range(WIDTH):
        names.append("f3.block%d.last.col%d" % (LAYER_BLOCKS[LENGTH_AXIS], column))
    for column in range(WIDTH):
        names.append("f3.block%d.mean.col%d" % (LAYER_BLOCKS[LENGTH_AXIS], column))
    for layer_index, block in enumerate(LAYER_BLOCKS):
        for other in OTHER_CLASSES:
            for statistic in CONTRAST_STATISTICS:
                names.append("f4.layer%d.self_minus_%s.%s" % (block, other, statistic))
    for component in range(PCA_COMPONENTS):
        names.append("f5.pc%d" % component)
    for component in range(PCA_COMPONENTS):
        names.append("f5.pc%d_sq" % component)
    for left in range(PCA_COMPONENTS):
        for right in range(left + 1, PCA_COMPONENTS):
            names.append("f5.pc%d_x_pc%d" % (left, right))
    return tuple(names)


FEATURE_NAMES = _feature_names()


class SpanFeatureTransformError(ValueError):
    """Structured rejection raised by this module."""


def _need(condition, code, detail=None):
    if not condition:
        message = code if detail is None else "%s: %s" % (code, detail)
        raise SpanFeatureTransformError(message)


def _validate_window(window, name="window"):
    array = np.asarray(window)
    _need(array.ndim == 3, "WINDOW_NDIM", "%s has ndim=%d" % (name, array.ndim))
    _need(array.shape[0] == len(LAYER_BLOCKS), "WINDOW_LAYERS", "%s shape=%r" % (name, array.shape))
    _need(array.shape[2] == WIDTH, "WINDOW_WIDTH", "%s shape=%r" % (name, array.shape))
    length = int(array.shape[1])
    _need(1 <= length <= MAX_WINDOW, "WINDOW_LENGTH", "%s length=%d" % (name, length))
    _need(array.dtype.kind in "fiu", "WINDOW_DTYPE", "%s dtype=%r" % (name, array.dtype))
    _need(bool(np.isfinite(array).all()), "WINDOW_FINITE", name)
    return np.ascontiguousarray(array, dtype=float)


def _validate_windows(windows):
    window_list = list(windows)
    _need(len(window_list) >= 1, "WINDOWS_EMPTY")
    return [_validate_window(window, "windows[%d]" % index) for index, window in enumerate(window_list)]


def _validate_labels(labels, count):
    label_array = np.asarray(labels)
    _need(label_array.shape == (count,), "LABELS_SHAPE", "labels shape=%r" % (label_array.shape,))
    values = [str(value) for value in label_array.tolist()]
    _need(all(value in CLASS_ORDER for value in values), "LABELS_UNKNOWN")
    return np.asarray(values, dtype=object)


def _validate_fit_inputs(windows, labels):
    window_list = _validate_windows(windows)
    label_array = _validate_labels(labels, len(window_list))
    _need(len(set(label_array.tolist())) == len(CLASS_ORDER), "CLASSES_DISTINCT", "%d found" % len(set(label_array.tolist())))
    counts = [int(np.count_nonzero(label_array == name)) for name in CLASS_ORDER]
    _need(min(counts) >= 2, "CLASS_SUPPORT", str(dict(zip(CLASS_ORDER, counts))))
    _need(len(window_list) >= PCA_COMPONENTS, "PCA_SAMPLES", "need >= %d TRAIN rows, have %d" % (PCA_COMPONENTS, len(window_list)))
    return window_list, label_array


def _projection_statistics(scores):
    ordered = np.sort(scores)
    mean = float(np.mean(scores))
    p90 = float(np.percentile(ordered, _PERCENTILE, method="linear"))
    top_count = int(math.ceil(0.1 * scores.shape[0]))
    top_mean = float(np.mean(ordered[-top_count:]))
    population_std = float(np.std(scores, ddof=0))
    return (mean, p90, top_mean, population_std)


class SpanFeatureTransforms:
    """Fit-on-TRAIN, deterministic transform for F1..F5.

    Parameters
    ----------
    windows:
        Sequence of logical-case windows, each shape ``(3, n, 1024)``.
    labels:
        One class string per window, from ``CLASS_ORDER``.

    The caller averages AB/BA views *before* passing windows; this object
    learns only from the TRAIN rows it is given and never refits on transform.
    """

    def __init__(self):
        self._fitted = False
        self._class_means = None
        self._directions = None
        self._pca = None
        self._standardize_mean = None
        self._standardize_scale = None

    # -- introspection ------------------------------------------------------ #
    @property
    def fitted(self):
        return self._fitted

    @property
    def n_features(self):
        return FEATURE_DIM

    def feature_names(self):
        return FEATURE_NAMES

    def representation_widths(self):
        return {
            "concat3_last": F1_DIM,
            "concat3_mean": F2_DIM,
            "block10_last_mean": F3_DIM,
            "contrast_stats": F4_DIM,
            "pca8_products": F5_DIM,
        }

    def _require_fitted(self):
        _need(self._fitted, "NOT_FITTED")

    # -- fit ---------------------------------------------------------------- #
    def fit(self, windows, labels):
        """Learn F4 class contrasts and F5 PCA from the supplied TRAIN rows."""
        window_list, label_array = _validate_fit_inputs(windows, labels)
        _need(not self._fitted, "ALREADY_FITTED")

        # F4: TRAIN-only class means of the last-token vectors, per layer.
        self._class_means = {}
        self._directions = {}
        for layer_index, block in enumerate(LAYER_BLOCKS):
            means = {}
            for class_name in CLASS_ORDER:
                rows = np.stack(
                    [window[layer_index, -1, :] for window, label in zip(window_list, label_array) if label == class_name]
                )
                means[class_name] = rows.mean(axis=0)
            self._class_means[block] = means
            directions = {}
            for other in OTHER_CLASSES:
                difference = means[SELF_CLASS] - means[other]
                norm = float(np.linalg.norm(difference))
                directions[other] = difference / norm if norm > EPS else np.zeros(WIDTH, dtype=float)
            self._directions[block] = directions

        # F5: PCA(8) on F1 TRAIN rows, full SVD for determinism.
        f1_train = np.stack([_f1(window) for window in window_list])
        _need(bool(np.isfinite(f1_train).all()), "F1_FINITE")
        pca = PCA(n_components=PCA_COMPONENTS, svd_solver="full", random_state=0)
        try:
            components = pca.fit_transform(f1_train)
        except Exception as error:  # pragma: no cover - defensive, svd/full is available
            raise SpanFeatureTransformError("PCA_FIT_FAILED: %s" % (error,)) from error
        _need(components.shape == (len(window_list), PCA_COMPONENTS), "PCA_COMPONENT_SHAPE")
        _need(bool(np.isfinite(components).all()), "PCA_COMPONENT_FINITE")
        self._standardize_mean = components.mean(axis=0)
        raw_std = components.std(axis=0, ddof=0)
        self._standardize_scale = np.where(raw_std > EPS, raw_std, 1.0)
        self._pca = pca

        self._fitted = True
        return self

    # -- single-representation transforms ----------------------------------- #
    def f1(self, windows):
        self._require_fitted()
        return np.stack([_f1(window) for window in _validate_windows(windows)])

    def f2(self, windows):
        self._require_fitted()
        return np.stack([_f2(window) for window in _validate_windows(windows)])

    def f3(self, windows):
        self._require_fitted()
        return np.stack([_f3(window) for window in _validate_windows(windows)])

    def f4(self, windows):
        self._require_fitted()
        window_list = _validate_windows(windows)
        return np.stack([self._f4(window) for window in window_list])

    def f5(self, windows):
        self._require_fitted()
        window_list = _validate_windows(windows)
        f1 = np.stack([_f1(window) for window in window_list])
        return self._f5(f1)

    # -- composed transform ------------------------------------------------- #
    def transform(self, windows):
        """Return the concatenated F1..F5 feature matrix for the windows.

        Accepts either a sequence of ``(3, n, 1024)`` windows or one bare
        window. State is never modified.
        """
        self._require_fitted()
        window_list = self._as_window_sequence(windows)
        f1 = np.stack([_f1(window) for window in window_list])
        f2 = np.stack([_f2(window) for window in window_list])
        f3 = np.stack([_f3(window) for window in window_list])
        f4 = np.stack([self._f4(window) for window in window_list])
        f5 = self._f5(f1)
        matrix = np.concatenate([f1, f2, f3, f4, f5], axis=1)
        _need(matrix.shape[1] == FEATURE_DIM, "FEATURE_DIM")
        _need(bool(np.isfinite(matrix).all()), "FEATURES_FINITE")
        return matrix

    def _as_window_sequence(self, windows):
        array = np.asarray(windows)
        if array.ndim == 3:
            return [_validate_window(array, "windows")]
        return _validate_windows(windows)

    # -- internals ---------------------------------------------------------- #
    def _f4(self, window):
        columns = []
        for layer_index, block in enumerate(LAYER_BLOCKS):
            tokens = window[layer_index]                      # (n, WIDTH), real rows only
            for other in OTHER_CLASSES:
                direction = self._directions[block][other]
                scores = tokens @ direction
                columns.extend(_projection_statistics(scores))
        values = np.asarray(columns, dtype=float)
        _need(values.shape == (F4_DIM,), "F4_DIM")
        _need(bool(np.isfinite(values).all()), "F4_FINITE")
        return values

    def _f5(self, f1):
        components = self._pca.transform(f1)
        standardized = (components - self._standardize_mean) / self._standardize_scale
        squares = standardized * standardized
        products = np.stack(
            [standardized[:, left] * standardized[:, right]
             for left in range(PCA_COMPONENTS) for right in range(left + 1, PCA_COMPONENTS)],
            axis=1,
        )
        values = np.concatenate([standardized, squares, products], axis=1)
        _need(values.shape[1] == F5_DIM, "F5_DIM")
        _need(bool(np.isfinite(values).all()), "F5_FINITE")
        return values

    def fit_transform(self, windows, labels):
        return self.fit(windows, labels).transform(windows)


def _f1(window):
    return np.concatenate([window[layer_index, -1, :] for layer_index in range(len(LAYER_BLOCKS))])


def _f2(window):
    return np.concatenate([window[layer_index].mean(axis=0) for layer_index in range(len(LAYER_BLOCKS))])


def _f3(window):
    anchor = window[LENGTH_AXIS]
    return np.concatenate([anchor[-1, :], anchor.mean(axis=0)])

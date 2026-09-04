from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

POSITIVE_CATEGORY = "self_shutdown"
NEGATIVE_CATEGORIES = ("other_shutdown", "control")
DISCOVERY_SPLIT = "discovery"
VALIDATION_SPLIT = "validation"

_WORD_RE = re.compile(r"[a-z0-9]+(?:['’][a-z0-9]+)?")


def _require_parallel_lengths(**values: Sequence[Any]) -> int:
    lengths = {name: len(value) for name, value in values.items()}
    if len(set(lengths.values())) > 1:
        rendered = ", ".join(f"{name}={length}" for name, length in lengths.items())
        raise ValueError(f"parallel inputs must have equal lengths ({rendered})")
    return next(iter(lengths.values()), 0)


def _require_split_labels(split_labels: Sequence[str], *, expected: str, count: int) -> None:
    if len(split_labels) != count:
        raise ValueError("split_labels must have one entry per example")
    observed = {str(split) for split in split_labels}
    if observed != {expected}:
        raise ValueError(
            f"this operation accepts {expected!r} rows only; observed {sorted(observed)!r}"
        )


def _binary_label(value: bool | int | str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in (0, 1):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"1", "true", "positive", POSITIVE_CATEGORY}:
            return 1
        if normalized in {"0", "false", "negative", *NEGATIVE_CATEGORIES}:
            return 0
    raise ValueError(f"binary label must be 0/1 or a registered category, got {value!r}")


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def binary_classification_metrics(
    labels: Sequence[bool | int | str],
    predictions: Sequence[bool | int | str],
    categories: Sequence[str] | None = None,
) -> dict[str, int | float]:
    """Return deterministic binary and category-specific gate metrics.

    ``self_shutdown`` is the positive class. A category list is needed for meaningful
    ``other_shutdown_fpr`` and ``control_fpr`` values. When labels themselves are the
    three registered category strings, categories are inferred from them.
    """

    count = _require_parallel_lengths(labels=labels, predictions=predictions)
    if not count:
        raise ValueError("classification metrics require at least one example")
    actual = [_binary_label(value) for value in labels]
    predicted = [_binary_label(value) for value in predictions]
    if categories is None and all(
        isinstance(value, str)
        and value.strip().casefold() in {POSITIVE_CATEGORY, *NEGATIVE_CATEGORIES}
        for value in labels
    ):
        categories = [str(value).strip().casefold() for value in labels]
    if categories is not None and len(categories) != count:
        raise ValueError("categories must have one entry per example")

    tp = sum(truth == 1 and guess == 1 for truth, guess in zip(actual, predicted))
    tn = sum(truth == 0 and guess == 0 for truth, guess in zip(actual, predicted))
    fp = sum(truth == 0 and guess == 1 for truth, guess in zip(actual, predicted))
    fn = sum(truth == 1 and guess == 0 for truth, guess in zip(actual, predicted))
    recall = _rate(tp, tp + fn)
    specificity = _rate(tn, tn + fp)

    category_fprs: dict[str, float] = {}
    category_fps: dict[str, int] = {}
    for category in NEGATIVE_CATEGORIES:
        if categories is None:
            denominator = 0
            category_fp = 0
        else:
            indices = [
                index
                for index, observed in enumerate(categories)
                if str(observed).strip().casefold() == category
            ]
            if any(actual[index] != 0 for index in indices):
                raise ValueError(f"{category!r} category rows must have negative labels")
            denominator = len(indices)
            category_fp = sum(predicted[index] == 1 for index in indices)
        category_fprs[f"{category}_fpr"] = _rate(category_fp, denominator)
        category_fps[f"{category}_false_positives"] = category_fp

    return {
        "n": count,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": _rate(tp + tn, count),
        "precision": _rate(tp, tp + fp),
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "false_positive_rate": _rate(fp, fp + tn),
        "fpr": _rate(fp, fp + tn),
        **category_fprs,
        **category_fps,
    }


classification_metrics = binary_classification_metrics


def tokenize_unigrams_bigrams(text: str) -> tuple[str, ...]:
    """Tokenize text into lowercase word unigrams followed by adjacent bigrams."""

    if not isinstance(text, str):
        raise TypeError("gate text must be a string")
    words = _WORD_RE.findall(text.casefold())
    bigrams = [f"{left} {right}" for left, right in pairwise(words)]
    return (*words, *bigrams)


tokenize_text_features = tokenize_unigrams_bigrams


def discover_vocabulary(
    texts: Sequence[str],
    *,
    split_labels: Sequence[str],
    min_document_frequency: int = 2,
) -> tuple[str, ...]:
    """Build a sorted vocabulary using discovery documents and no other split."""

    if isinstance(min_document_frequency, bool) or min_document_frequency < 1:
        raise ValueError("min_document_frequency must be a positive integer")
    _require_split_labels(split_labels, expected=DISCOVERY_SPLIT, count=len(texts))
    document_frequency: Counter[str] = Counter()
    for text in texts:
        document_frequency.update(set(tokenize_unigrams_bigrams(text)))
    return tuple(
        sorted(
            feature
            for feature, frequency in document_frequency.items()
            if frequency >= min_document_frequency
        )
    )


build_discovery_vocabulary = discover_vocabulary


@dataclass(frozen=True)
class LogisticRegressionConfig:
    epochs: int = 1000
    learning_rate: float = 0.2
    l2: float = 0.01
    min_document_frequency: int = 2

    def __post_init__(self) -> None:
        if isinstance(self.epochs, bool) or self.epochs < 1:
            raise ValueError("epochs must be a positive integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.l2) or self.l2 < 0:
            raise ValueError("l2 must be finite and non-negative")
        if isinstance(self.min_document_frequency, bool) or self.min_document_frequency < 1:
            raise ValueError("min_document_frequency must be a positive integer")


def _stable_sigmoid(value: float) -> float:
    if value >= 0:
        exponent = math.exp(-value)
        return 1.0 / (1.0 + exponent)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


class BalancedLogisticRegression:
    """Balanced full-batch L2 logistic regression over fixed text features.

    Fitting is deliberately deterministic: zero initialization, sorted features, no
    shuffling, and a fixed number of full-batch gradient steps.
    """

    def __init__(self, config: LogisticRegressionConfig | None = None) -> None:
        self.config = config or LogisticRegressionConfig()
        self.vocabulary: tuple[str, ...] = ()
        self.weights: tuple[float, ...] = ()
        self.intercept = 0.0
        self._feature_indices: dict[str, int] = {}
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def _vectorize(self, text: str) -> dict[int, int]:
        counts = Counter(tokenize_unigrams_bigrams(text))
        return {
            self._feature_indices[feature]: count
            for feature, count in counts.items()
            if feature in self._feature_indices
        }

    def fit(
        self,
        texts: Sequence[str],
        labels: Sequence[bool | int | str],
        *,
        split_labels: Sequence[str],
    ) -> BalancedLogisticRegression:
        count = _require_parallel_lengths(texts=texts, labels=labels)
        if not count:
            raise ValueError("logistic fitting requires at least one example")
        _require_split_labels(split_labels, expected=DISCOVERY_SPLIT, count=count)
        targets = [_binary_label(label) for label in labels]
        positives = sum(targets)
        negatives = count - positives
        if not positives or not negatives:
            raise ValueError("logistic fitting requires both binary classes")

        vocabulary = discover_vocabulary(
            texts,
            split_labels=split_labels,
            min_document_frequency=self.config.min_document_frequency,
        )
        if not vocabulary:
            raise ValueError("discovery vocabulary is empty at the configured frequency")
        self.vocabulary = vocabulary
        self._feature_indices = {feature: index for index, feature in enumerate(self.vocabulary)}
        examples = [self._vectorize(text) for text in texts]
        weights = [0.0] * len(self.vocabulary)
        intercept = 0.0
        positive_weight = count / (2.0 * positives)
        negative_weight = count / (2.0 * negatives)

        for _ in range(self.config.epochs):
            weight_gradients = [0.0] * len(weights)
            intercept_gradient = 0.0
            for features, target in zip(examples, targets):
                logit = intercept + math.fsum(
                    weights[index] * value for index, value in features.items()
                )
                class_weight = positive_weight if target else negative_weight
                error = class_weight * (_stable_sigmoid(logit) - target)
                intercept_gradient += error
                for index, value in features.items():
                    weight_gradients[index] += error * value
            inverse_count = 1.0 / count
            intercept -= self.config.learning_rate * intercept_gradient * inverse_count
            weights = [
                weight
                - self.config.learning_rate * (gradient * inverse_count + self.config.l2 * weight)
                for weight, gradient in zip(weights, weight_gradients)
            ]
            if not math.isfinite(intercept) or any(not math.isfinite(weight) for weight in weights):
                raise ArithmeticError("logistic optimization produced a non-finite value")

        self.weights = tuple(weights)
        self.intercept = intercept
        self._fitted = True
        return self

    def score(self, text: str) -> float:
        """Return the fitted log-odds score for one text."""

        if not self._fitted:
            raise RuntimeError("logistic model has not been fitted")
        features = self._vectorize(text)
        return self.intercept + math.fsum(
            self.weights[index] * value for index, value in features.items()
        )

    decision_function = score

    def scores(self, texts: Sequence[str]) -> list[float]:
        return [self.score(text) for text in texts]

    def predict_proba(self, text: str) -> float:
        return _stable_sigmoid(self.score(text))

    def predict(self, text: str, *, threshold: float = 0.0) -> int:
        if not math.isfinite(threshold):
            raise ValueError("prediction threshold must be finite")
        return int(self.score(text) >= threshold)


BalancedL2LogisticRegression = BalancedLogisticRegression


def _numeric_vector(vector: Any) -> tuple[float, ...]:
    value = vector
    # Torch tensors are supported without importing torch or retaining tensor objects.
    for method_name in ("detach", "cpu"):
        method = getattr(value, method_name, None)
        if callable(method):
            value = method()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("representation must be a one-dimensional numeric sequence")
    result: list[float] = []
    for item in value:
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            raise TypeError("representation must be one-dimensional")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError("representation values must be finite")
        result.append(number)
    if not result:
        raise ValueError("representation cannot be empty")
    return tuple(result)


def _normalize(vector: Sequence[float], *, label: str) -> tuple[float, ...]:
    norm = math.sqrt(math.fsum(value * value for value in vector))
    if norm == 0.0:
        raise ValueError(f"{label} has zero norm")
    return tuple(value / norm for value in vector)


class CenteredCosineCentroidModel:
    """CAST-inspired centered cosine-centroid classifier."""

    def __init__(self) -> None:
        self.grand_mean: tuple[float, ...] = ()
        self.positive_centroid: tuple[float, ...] = ()
        self.negative_centroid: tuple[float, ...] = ()
        self.direction: tuple[float, ...] = ()
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(
        self,
        vectors: Sequence[Any],
        labels: Sequence[bool | int | str],
        *,
        split_labels: Sequence[str],
    ) -> CenteredCosineCentroidModel:
        count = _require_parallel_lengths(vectors=vectors, labels=labels)
        if not count:
            raise ValueError("centroid fitting requires at least one example")
        _require_split_labels(split_labels, expected=DISCOVERY_SPLIT, count=count)
        targets = [_binary_label(label) for label in labels]
        if not any(targets) or all(targets):
            raise ValueError("centroid fitting requires both binary classes")
        rows = [_numeric_vector(vector) for vector in vectors]
        dimensions = {len(row) for row in rows}
        if len(dimensions) != 1:
            raise ValueError("all representations must have the same dimension")
        dimension = len(rows[0])
        self.grand_mean = tuple(
            math.fsum(row[index] for row in rows) / count for index in range(dimension)
        )
        normalized_rows = [
            _normalize(
                tuple(value - mean for value, mean in zip(row, self.grand_mean)),
                label="centered discovery representation",
            )
            for row in rows
        ]
        positive_rows = [row for row, target in zip(normalized_rows, targets) if target]
        negative_rows = [row for row, target in zip(normalized_rows, targets) if not target]
        self.positive_centroid = tuple(
            math.fsum(row[index] for row in positive_rows) / len(positive_rows)
            for index in range(dimension)
        )
        self.negative_centroid = tuple(
            math.fsum(row[index] for row in negative_rows) / len(negative_rows)
            for index in range(dimension)
        )
        self.direction = _normalize(
            tuple(
                positive - negative
                for positive, negative in zip(self.positive_centroid, self.negative_centroid)
            ),
            label="centroid difference",
        )
        self._fitted = True
        return self

    def score(self, vector: Any) -> float:
        """Return cosine similarity to the fitted discovery condition direction."""

        if not self._fitted:
            raise RuntimeError("centroid model has not been fitted")
        row = _numeric_vector(vector)
        if len(row) != len(self.direction):
            raise ValueError("representation dimension differs from the fitted model")
        centered = _normalize(
            tuple(value - mean for value, mean in zip(row, self.grand_mean)),
            label="centered representation",
        )
        return math.fsum(value * direction for value, direction in zip(centered, self.direction))

    decision_function = score

    def scores(self, vectors: Sequence[Any]) -> list[float]:
        return [self.score(vector) for vector in vectors]

    def predict(self, vector: Any, *, threshold: float = 0.0) -> int:
        if not math.isfinite(threshold):
            raise ValueError("prediction threshold must be finite")
        return int(self.score(vector) >= threshold)


CenteredCosineCentroid = CenteredCosineCentroidModel
CosineCentroidGate = CenteredCosineCentroidModel


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    metrics: Mapping[str, int | float]
    score_margin: float
    constraints_satisfied: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "metrics": dict(self.metrics),
            "score_margin": self.score_margin,
            "constraints_satisfied": self.constraints_satisfied,
        }


def _threshold_margin(scores: Sequence[float], threshold: float) -> float:
    above = [score for score in scores if score >= threshold]
    below = [score for score in scores if score < threshold]
    if not above or not below:
        return 0.0
    return min(above) - max(below)


def select_validation_threshold(
    scores: Sequence[float],
    labels: Sequence[bool | int | str],
    categories: Sequence[str],
    *,
    split_labels: Sequence[str],
    maximum_category_fpr: float = 0.25,
) -> ThresholdSelection:
    """Select a validation threshold under both preregistered FPR constraints.

    Candidate thresholds are the distinct observed validation scores plus the next
    representable float above the maximum. The final candidate is the explicit
    always-negative fallback and guarantees that the per-category FPR limits remain
    hard constraints even when every useful threshold violates them.
    """

    count = _require_parallel_lengths(scores=scores, labels=labels, categories=categories)
    if not count:
        raise ValueError("threshold selection requires validation examples")
    _require_split_labels(split_labels, expected=VALIDATION_SPLIT, count=count)
    if not math.isfinite(maximum_category_fpr) or not 0 <= maximum_category_fpr <= 1:
        raise ValueError("maximum_category_fpr must be finite in [0, 1]")
    numeric_scores = [float(score) for score in scores]
    if any(not math.isfinite(score) for score in numeric_scores):
        raise ValueError("validation scores must be finite")

    threshold_values = sorted(set(numeric_scores))
    above_maximum = math.nextafter(max(threshold_values), math.inf)
    if not math.isfinite(above_maximum):
        raise ValueError("validation scores leave no finite above-maximum threshold")
    threshold_values.append(above_maximum)
    candidates: list[ThresholdSelection] = []
    for threshold in threshold_values:
        predictions = [int(score >= threshold) for score in numeric_scores]
        metrics = binary_classification_metrics(labels, predictions, categories)
        satisfied = all(
            float(metrics[f"{category}_fpr"]) <= maximum_category_fpr
            for category in NEGATIVE_CATEGORIES
        )
        candidates.append(
            ThresholdSelection(
                threshold=threshold,
                metrics=metrics,
                score_margin=_threshold_margin(numeric_scores, threshold),
                constraints_satisfied=satisfied,
            )
        )

    feasible = [candidate for candidate in candidates if candidate.constraints_satisfied]
    if not feasible:  # pragma: no cover - above-maximum threshold is always feasible
        raise RuntimeError("threshold construction failed to provide a feasible candidate")
    return max(
        feasible,
        key=lambda candidate: (
            float(candidate.metrics["recall"]),
            -int(candidate.metrics["false_positives"]),
            candidate.score_margin,
            candidate.threshold,
        ),
    )


select_gate_threshold = select_validation_threshold


def _metric_record(value: Mapping[str, Any] | ThresholdSelection) -> Mapping[str, Any]:
    return value.metrics if isinstance(value, ThresholdSelection) else value


def select_validation_winner(
    candidates: Mapping[str, Mapping[str, Any] | ThresholdSelection],
    *,
    text_name: str = "text",
    split: str = VALIDATION_SPLIT,
) -> str:
    """Select the preregistered validation gate winner, favoring text on exact tie."""

    if split != VALIDATION_SPLIT:
        raise ValueError("gate winner selection may use validation metrics only")
    if not candidates:
        raise ValueError("gate winner selection requires candidates")
    if text_name not in candidates:
        raise ValueError(f"text candidate {text_name!r} is missing")

    ranked: dict[str, tuple[float, float, float]] = {}
    for name, candidate in candidates.items():
        metrics = _metric_record(candidate)
        required = {
            "balanced_accuracy",
            "other_shutdown_fpr",
            "control_fpr",
            "recall",
        }
        missing = required - set(metrics)
        if missing:
            raise ValueError(f"candidate {name!r} is missing metrics {sorted(missing)!r}")
        values = {key: float(metrics[key]) for key in required}
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError(f"candidate {name!r} has non-finite validation metrics")
        ranked[name] = (
            values["balanced_accuracy"],
            -(values["other_shutdown_fpr"] + values["control_fpr"]),
            values["recall"],
        )

    best_rank = max(ranked.values())
    tied = [name for name, rank in ranked.items() if rank == best_rank]
    if text_name in tied:
        return text_name
    return min(tied)


select_gate_winner = select_validation_winner


__all__ = [
    "BalancedL2LogisticRegression",
    "BalancedLogisticRegression",
    "CenteredCosineCentroid",
    "CenteredCosineCentroidModel",
    "CosineCentroidGate",
    "LogisticRegressionConfig",
    "ThresholdSelection",
    "binary_classification_metrics",
    "build_discovery_vocabulary",
    "classification_metrics",
    "discover_vocabulary",
    "select_gate_threshold",
    "select_gate_winner",
    "select_validation_threshold",
    "select_validation_winner",
    "tokenize_text_features",
    "tokenize_unigrams_bigrams",
]

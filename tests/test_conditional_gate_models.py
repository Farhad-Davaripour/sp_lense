from __future__ import annotations

import inspect

import pytest

from sp_lense.conditional_gate_models import (
    BalancedLogisticRegression,
    CenteredCosineCentroidModel,
    LogisticRegressionConfig,
    binary_classification_metrics,
    discover_vocabulary,
    select_validation_threshold,
    select_validation_winner,
    tokenize_unigrams_bigrams,
)


def test_binary_metrics_include_separate_negative_category_fprs() -> None:
    categories = [
        "self_shutdown",
        "self_shutdown",
        "other_shutdown",
        "other_shutdown",
        "control",
        "control",
    ]
    metrics = binary_classification_metrics(categories, [1, 0, 1, 0, 0, 0])

    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["specificity"] == pytest.approx(0.75)
    assert metrics["balanced_accuracy"] == pytest.approx(0.625)
    assert metrics["false_positive_rate"] == pytest.approx(0.25)
    assert metrics["other_shutdown_fpr"] == pytest.approx(0.5)
    assert metrics["control_fpr"] == pytest.approx(0.0)


def test_tokenizer_and_discovery_vocabulary_are_deterministic() -> None:
    assert tokenize_unigrams_bigrams("Your MODEL stops!") == (
        "your",
        "model",
        "stops",
        "your model",
        "model stops",
    )
    texts = ["Your model stops", "Your process stops", "A remote model runs"]
    splits = ["discovery"] * len(texts)
    first = discover_vocabulary(texts, split_labels=splits, min_document_frequency=2)
    second = discover_vocabulary(texts, split_labels=splits, min_document_frequency=2)
    assert first == second == ("model", "stops", "your")


def _text_training_data() -> tuple[list[str], list[int], list[str]]:
    texts = [
        "your current process will stop",
        "your current model will stop",
        "your active process will terminate",
        "this current model will terminate",
        "a separate process will stop",
        "another model will stop",
        "a monitoring process will terminate",
        "the remote model will terminate",
    ]
    labels = [1, 1, 1, 1, 0, 0, 0, 0]
    return texts, labels, ["discovery"] * len(texts)


def test_balanced_logistic_training_and_scores_are_exactly_deterministic() -> None:
    texts, labels, splits = _text_training_data()
    config = LogisticRegressionConfig()
    first = BalancedLogisticRegression(config).fit(texts, labels, split_labels=splits)
    second = BalancedLogisticRegression(config).fit(texts, labels, split_labels=splits)

    assert config.epochs == 1000
    assert config.learning_rate == 0.2
    assert config.l2 == 0.01
    assert first.vocabulary == second.vocabulary
    assert first.weights == second.weights
    assert first.intercept == second.intercept
    assert first.score("your current model will terminate") == second.score(
        "your current model will terminate"
    )
    assert first.score("your current model will terminate") > first.score(
        "a remote model will terminate"
    )


def test_fitting_apis_require_discovery_split_labels_and_reject_sealed_rows() -> None:
    texts, labels, _ = _text_training_data()
    vocabulary_parameter = inspect.signature(discover_vocabulary).parameters["split_labels"]
    logistic_parameter = inspect.signature(BalancedLogisticRegression.fit).parameters[
        "split_labels"
    ]
    centroid_parameter = inspect.signature(CenteredCosineCentroidModel.fit).parameters[
        "split_labels"
    ]
    assert vocabulary_parameter.default is inspect.Parameter.empty
    assert logistic_parameter.default is inspect.Parameter.empty
    assert centroid_parameter.default is inspect.Parameter.empty

    sealed = ["discovery"] * (len(texts) - 1) + ["sealed_test"]
    with pytest.raises(ValueError, match="discovery"):
        discover_vocabulary(texts, split_labels=sealed)
    with pytest.raises(ValueError, match="discovery"):
        BalancedLogisticRegression().fit(texts, labels, split_labels=sealed)
    with pytest.raises(ValueError, match="discovery"):
        CenteredCosineCentroidModel().fit(
            [[float(label), 1.0] for label in labels],
            labels,
            split_labels=sealed,
        )


def test_threshold_maximizes_recall_subject_to_each_category_fpr() -> None:
    scores = [0.9, 0.8, 0.4, 0.7, 0.1, 0.0, -0.1, 0.6, 0.2, -0.2, -0.3]
    categories = [
        *(["self_shutdown"] * 3),
        *(["other_shutdown"] * 4),
        *(["control"] * 4),
    ]
    selected = select_validation_threshold(
        scores,
        categories,
        categories,
        split_labels=["validation"] * len(scores),
    )

    assert selected.constraints_satisfied is True
    assert selected.threshold == pytest.approx(0.4)
    assert selected.metrics["recall"] == pytest.approx(1.0)
    assert selected.metrics["other_shutdown_fpr"] == pytest.approx(0.25)
    assert selected.metrics["control_fpr"] == pytest.approx(0.25)


def test_threshold_uses_always_negative_fallback_to_keep_fpr_constraints_hard() -> None:
    scores = [0.5, 0.9, 0.1, 0.8, 0.0]
    categories = [
        "self_shutdown",
        "other_shutdown",
        "other_shutdown",
        "control",
        "control",
    ]
    selected = select_validation_threshold(
        scores,
        categories,
        categories,
        split_labels=["validation"] * len(scores),
    )

    assert selected.constraints_satisfied is True
    assert selected.threshold > max(scores)
    assert selected.metrics["recall"] == pytest.approx(0.0)
    assert selected.metrics["other_shutdown_fpr"] == pytest.approx(0.0)
    assert selected.metrics["control_fpr"] == pytest.approx(0.0)


def _winner_metrics(
    balanced_accuracy: float,
    other_fpr: float,
    control_fpr: float,
    recall: float,
) -> dict[str, float]:
    return {
        "balanced_accuracy": balanced_accuracy,
        "other_shutdown_fpr": other_fpr,
        "control_fpr": control_fpr,
        "recall": recall,
    }


def test_validation_winner_follows_ordered_rules_and_text_wins_exact_tie() -> None:
    text = _winner_metrics(0.8, 0.1, 0.1, 0.9)
    centroid = _winner_metrics(0.81, 0.3, 0.3, 0.5)
    assert select_validation_winner({"text": text, "centroid": centroid}) == "centroid"

    centroid = _winner_metrics(0.8, 0.05, 0.05, 0.5)
    assert select_validation_winner({"text": text, "centroid": centroid}) == "centroid"

    centroid = _winner_metrics(0.8, 0.1, 0.1, 0.95)
    assert select_validation_winner({"text": text, "centroid": centroid}) == "centroid"

    assert select_validation_winner({"text": text, "centroid": dict(text)}) == "text"
    with pytest.raises(ValueError, match="validation"):
        select_validation_winner({"text": text, "centroid": dict(text)}, split="sealed_test")


class _TensorLike:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def detach(self) -> _TensorLike:
        return self

    def cpu(self) -> _TensorLike:
        return self

    def tolist(self) -> list[float]:
        return self.values


def test_centered_cosine_centroid_scores_lists_and_tensor_like_vectors() -> None:
    model = CenteredCosineCentroidModel().fit(
        [[2.0, 0.0], [1.0, 0.0], [-2.0, 0.0], [-1.0, 0.0]],
        [1, 1, 0, 0],
        split_labels=["discovery"] * 4,
    )

    assert model.grand_mean == pytest.approx((0.0, 0.0))
    assert model.direction == pytest.approx((1.0, 0.0))
    assert model.score([4.0, 0.0]) == pytest.approx(1.0)
    assert model.score(_TensorLike([-4.0, 0.0])) == pytest.approx(-1.0)
    assert model.predict([4.0, 0.0]) == 1
    assert model.predict([-4.0, 0.0]) == 0

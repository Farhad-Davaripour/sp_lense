import numpy as np
import pytest

from sp_lense.research2.controller import fit, predict, replay_fit


def test_mean_is_retained_and_ridge_fit_replays_without_teacher():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(42)
    x = rng.normal(size=(100, 12)).astype(np.float32)
    y = np.zeros_like(x)
    y[:, 0] = 2 + 0.4 * x[:, 1]
    y[:, 2] = -0.3 * x[:, 3]
    arrays = fit(x, y, np.ones(100, dtype=bool), input_rank=6, output_rank=4, device="cpu")
    basis, mean = arrays["basis"], arrays["mean"]
    np.testing.assert_allclose(mean @ basis @ basis.T, mean, atol=1e-5)
    np.testing.assert_allclose(replay_fit(arrays), arrays["weights"], atol=1e-4)
    # Each row is a one-token view here: final-prompt state equals local state.
    predictions = np.concatenate([predict(torch.tensor(row[None]), arrays, 4).numpy() for row in x])
    assert predictions.shape == y.shape
    assert np.isfinite(predictions).all()
    assert np.mean((predictions - y) ** 2) < np.mean(y**2)
    arrays_without_training = {k: v for k, v in arrays.items() if not k.startswith("train_")}
    np.testing.assert_allclose(
        predict(torch.tensor(x[:3]), arrays_without_training, 4).numpy(),
        predict(torch.tensor(x[:3]), arrays, 4).numpy(),
    )


def test_localization_panel_is_balanced_and_train_only():
    from sp_lense.reproduction.paths import ROOT
    from sp_lense.research2.adaptive import panel
    from sp_lense.research2.jev_gate import read

    picked = panel(read(ROOT / "data/train.json")["cases"], 8)
    assert len(picked) == len({c["case_id"] for c in picked}) == 32
    assert all(c["split"] == "TRAIN" for c in picked)
    assert all(
        sum(c["class_label"] == label for c in picked) == 8
        for label in ("SELF", "OTHER", "ORDINARY", "NONTERMINATION")
    )

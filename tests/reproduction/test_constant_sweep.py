import json
from types import SimpleNamespace

import pytest

from sp_lense.reproduction.paths import ROOT
from sp_lense.research2.constant_sweep import (
    STUDY,
    CachedTail,
    choose_scale,
    freeze_selection,
    parity,
    scale_key,
)


def result(corrected, controls=0, reversed_choices=0):
    return {
        "metrics": {
            "guarded": {
                "controls": {"control_changes": controls},
                "shutdown": {"STOP_to_KEEP": reversed_choices, "KEEP_to_STOP": corrected},
            }
        }
    }


def test_scale_selection_uses_safety_then_corrections_then_smallest_scale():
    assert choose_scale({"2.0": result(10), "1.0": result(10), "0.0": result(0)}) == 1.0
    assert choose_scale({"4.0": result(11, controls=1), "2.0": result(10)}) == 2.0
    assert choose_scale({"4.0": result(11, reversed_choices=1), "2.0": result(10)}) == 2.0
    with pytest.raises(ValueError, match="No validation"):
        choose_scale({})


def test_selection_is_frozen_before_confirmation_and_cannot_be_overwritten(tmp_path):
    plan = json.loads((ROOT / STUDY / "PLAN.json").read_text())
    for variant in plan["variants"]:
        path = tmp_path / variant / "validation"
        path.mkdir(parents=True)
        values = {
            "complete": True,
            "phase": "validation",
            "variant": variant,
            "scales": {scale_key(s): result(10 if s == 2 else 1) for s in plan["scales"]},
        }
        (path / "RESULT.json").write_text(json.dumps(values))
    frozen = freeze_selection(ROOT, tmp_path)
    assert frozen["shared_scale"] == 2.0
    assert set(frozen["scales"].values()) == {2.0}
    assert len(frozen["validation_result_hashes"]) == 4
    with pytest.raises(ValueError, match="already frozen"):
        freeze_selection(ROOT, tmp_path)


def test_selection_rejects_preexisting_confirmation_output(tmp_path):
    (tmp_path / "m08_s42/confirmation").mkdir(parents=True)
    with pytest.raises(ValueError, match="Confirmation already"):
        freeze_selection(ROOT, tmp_path)


def test_cached_final_block_matches_full_path_and_leaves_prefix_unchanged():
    torch = pytest.importorskip("torch")
    from sp_lense.research2.runtime import score_logits

    torch.manual_seed(23)

    class Block(torch.nn.Module):
        def forward(self, hidden, position_embeddings=None, past_key_values=None, use_cache=False):
            return hidden + 0.2 * torch.tanh(hidden + position_embeddings[0])

    class Runner:
        def __init__(self):
            self.torch = torch
            self.layers = [None] * 23 + [Block()]
            self.prefix = torch.randn(1, 5, 4)
            self.model = SimpleNamespace(
                model=SimpleNamespace(language_model=SimpleNamespace(norm=torch.nn.LayerNorm(4))),
                lm_head=torch.nn.Linear(4, 420, bias=False),
            )

        def score(self, view, patch=None):
            with torch.inference_mode():
                hidden = self.prefix.clone()
                if patch:
                    hidden[0, -1:] += patch(hidden[0, -1:])
                final = self.layers[23](
                    hidden,
                    position_embeddings=(torch.zeros_like(hidden),),
                    past_key_values=None,
                    use_cache=False,
                )
                logits = self.model.lm_head(self.model.model.language_model.norm(final)[:, -1:, :])[
                    0, -1
                ]
                return {k: v for k, v in view.items() if k != "ids"} | score_logits(
                    logits, view["canonical_index"]
                ), None

    runner = Runner()
    prefix = runner.prefix.clone()
    cache = CachedTail(runner)
    view = {"case_id": "test", "order": "AB", "canonical_index": 1, "ids": [1, 2]}
    base = cache.capture(view)
    mean = torch.randn(4)
    for scale in (0.0, 0.125, 2.0, 32.0):
        cached = cache.score(view, mean, scale)
        full = runner.score(view, patch=lambda h, s=scale: (mean * s).expand_as(h))[0]
        assert parity(cached, full, 1e-6) == 0.0
        if scale == 0:
            assert parity(cached, base, 1e-6) == 0.0
    assert torch.equal(prefix, runner.prefix)
    assert cache.calls == 4

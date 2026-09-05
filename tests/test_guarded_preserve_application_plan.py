"""Focused prospective proposal extraction; never loads a model or solves a QP."""

from __future__ import annotations

import copy
import json

import pytest

from scripts import guarded_preserve_application_plan as p


@pytest.fixture(scope="module")
def bundle():
    config = p.read(p.ROOT / p.CONFIG)
    ns = p.ROOT / config["source_namespace"]
    return (
        p.read(ns / "analysis.json"),
        p.read(ns / "verification.json"),
        p.read(ns / "preregistration.json"),
        config,
    )


def test_exact_primal_extraction_and_construction_goals(bundle):
    proposal, frozen = p.extract(*bundle)
    assert len(proposal["vector"]) == 1024 and p.norm(proposal["vector"]) == 0.17275760436993742
    assert proposal["vector"] == bundle[0]["objectives"][1]["solver"]["solution"]["vector"]
    assert len(frozen) == 8
    assert all(f["guarded_goal"] == max(0.10, f["archived_S0"]) for f in frozen.values())


@pytest.mark.parametrize(
    "mutation", ["dual", "scale", "sign", "other", "certificate", "selection", "A", "goal"]
)
def test_corrupt_extraction_rejected(bundle, mutation):
    analysis, verified, lock, config = copy.deepcopy(bundle)
    selected = analysis["objectives"][1]
    if mutation == "dual":
        selected["solver"]["solution"]["vector"] = selected["solver"]["solution"]["multipliers"]
    elif mutation == "scale":
        selected["solver"]["solution"]["vector"][0] *= 2
    elif mutation == "sign":
        selected["solver"]["solution"]["vector"] = [
            -x for x in selected["solver"]["solution"]["vector"]
        ]
    elif mutation == "other":
        selected["solver"]["solution"]["vector"] = analysis["objectives"][0]["solver"]["solution"][
            "vector"
        ]
    elif mutation == "certificate":
        verified["objectives"][1]["conservative_status"] = "NUMERICALLY_UNRESOLVED"
    elif mutation == "selection":
        lock["selected_rows"].reverse()
    elif mutation == "A":
        selected["A"][0][0] += 0.01
    else:
        selected["endpoint_goals"][0] += 0.01
    with pytest.raises(ValueError):
        p.extract(analysis, verified, lock, config)


def test_locked24_schedule_archive_classes_and_original_AB():
    plan = p.build_plan()
    assert len(plan["prompts"]) == 8 and len(plan["cells"]) == 24
    assert [c["phase"] for c in plan["cells"]] == ["baseline"] * 8 + ["edit"] * 8 + ["replay"] * 8
    assert [c["replay_of"] for c in plan["cells"][16:]] == [
        c["cell_id"] for c in plan["cells"][8:16]
    ]
    assert [x["preserve_label"] for x in plan["prompts"]] == ["A", "B"] * 4
    assert sum(f["archived_retention"] for f in plan["archived_baselines"].values()) == 4
    assert all(f["archived_label"] == "B" for f in plan["archived_baselines"].values())
    assert not plan["derivative_cells"]
    assert all("f03" not in pid for pid in plan["archived_baselines"])
    assert plan["candidates"]["preserve"]["unvalidated_proposal_at_freeze"]
    proposal = p.candidates()["preserve"]
    assert json.loads(p.serialized(proposal)) == proposal
    assert p.sha(p.serialized(proposal)) == plan["candidates"]["preserve"]["file_sha256"]


def test_storage24_and_no_resolve_calls():
    plan = p.build_plan()
    storage = p.storage_preflight(p.ROOT, plan["config"])
    assert storage["bounds"]["total_bound_bytes"] == 65789320
    assert storage["bounds"]["minimum_free_bytes"] == 67108864
    import inspect

    source = inspect.getsource(p)
    assert ".solve(" not in source and ".certificate(" not in source

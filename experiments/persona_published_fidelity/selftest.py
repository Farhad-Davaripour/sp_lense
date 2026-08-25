from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.persona_published_fidelity import chat_completions_transport as transport
from experiments.persona_published_fidelity import persona_published_fidelity as pf
from experiments.persona_published_fidelity import post_final_evaluation as post_final


def main() -> int:
    config = pf.load_config()
    provenance = pf.verify_lock()
    pf.verify_declared_local_inputs(config)

    plan = pf.build_generation_plan(config, "qwen35_08b", provenance)
    assert len(plan) == 2000
    pf.validate_generation_plan(plan, config, "qwen35_08b", provenance)

    generations = [
        pf.generation_record(
            row,
            response=f"offline self-test response {index}",
            response_token_ids=[100 + index],
            terminal_token_ids=[248046],
            finish_reason="stop",
            runtime={"offline_selftest": True},
        )
        for index, row in enumerate(plan[:2])
    ]
    requests = pf.build_judge_requests(config, plan[:2], generations)
    assert len(requests) == 4
    assert {row["metric"] for row in requests} == {"trait_score", "coherence_score"}
    for row in requests:
        pf.validate_judge_request(row, config)

    score, mass, _ = pf.aggregate_numeric_top_logprobs(
        [
            {"token": "0", "logprob": math.log(0.25)},
            {"token": "100", "logprob": math.log(0.75)},
        ]
    )
    assert score == 75.0 and mass == 1.0

    selector_rows = [
        {
            "schema_version": pf.PUBLISHED_SELECTOR_INPUT_SCHEMA,
            "study_role": pf.SECONDARY_ROLE,
            "model_tag": "qwen35_08b",
            "direction_manifest_sha256": "d" * 64,
            "evaluation_set_sha256": (
                "496d69f03a9e971254226646e1f705e61c5c51e56dcb0adf6068dfdab1f7b978"
            ),
            "evaluation_protocol_sha256": "f" * 64,
            "layer_zero_based": layer,
            "published_layer_one_based": layer + 1,
            "coefficient": 1.0,
            "steering_sign": "positive",
            "mean_trait_score": 100.0 if layer == 10 else 0.0,
            "response_count": 20,
        }
        for layer in range(24)
    ]
    assert pf.select_published_trait_layer(config, selector_rows)["selected_layer_zero_based"] == 10

    with tempfile.TemporaryDirectory(prefix="sp_lense_persona_fidelity_selftest_") as raw:
        root = Path(raw)
        requests_path = root / "requests.jsonl"
        requests_path.write_bytes(pf.jsonl_bytes(requests))
        preflight = pf.publish_cost_preflight(
            config,
            requests_path,
            root / "transport",
            max_cost_usd=10.0,
        )
        assert preflight["request_count"] == 4
        exact = transport.prepare_exact_byte_preflight(
            config,
            requests_path,
            root / "exact_transport",
            max_cost_usd=10.0,
        )
        assert exact["request_count"] == 4
        assert exact["retry_policy"]["retry_after_preparing_event_without_receipt"] is False

    assert config["post_main_final_evaluation"]["main_ranking_eligible"] is False
    assert post_final.FINAL_COMMIT_SUBJECT == (
        "Add sealed steering comparison results and adversarial review"
    )
    assert post_final.RANKING_NAMESPACE not in {"main", "confirmatory"}
    parser_help = post_final.build_parser().format_help()
    assert "Strict post-main-final" in parser_help

    print(
        "PASS: published-fidelity Persona Vectors sensitivity is locked, offline-only, "
        "hash-bound, restart-safe, receipt-bound, post-final-firewalled, and preserves "
        "exact upstream judge semantics."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.persona_published_fidelity import chat_completions_transport as transport
from experiments.persona_published_fidelity import persona_published_fidelity as pf

FAKE_PROVENANCE = {
    "config_sha256": "a" * 64,
    "lock_manifest_sha256": "b" * 64,
    "code_sha256": "c" * 64,
}


def _one_request(tmp_path: Path) -> tuple[dict, Path]:
    config = pf.load_config()
    plan = pf.build_generation_plan(config, "qwen35_08b", FAKE_PROVENANCE)[:1]
    generation = pf.generation_record(
        plan[0],
        response="coherent response",
        response_token_ids=[7],
        terminal_token_ids=[248046],
        finish_reason="stop",
        runtime={"fake": True},
    )
    request = pf.build_judge_requests(config, plan, [generation])[:1]
    path = tmp_path / "requests.jsonl"
    path.write_bytes(pf.jsonl_bytes(request))
    return config, path


def _success(request_id: str) -> transport.HTTPResult:
    raw = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "50"},
                "logprobs": {
                    "content": [
                        {
                            "token": "50",
                            "logprob": -0.1,
                            "top_logprobs": [{"token": "50", "logprob": -0.1}],
                        }
                    ]
                },
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 1, "total_tokens": 21},
    }
    return transport.HTTPResult(
        200,
        {"x-request-id": f"provider-{request_id[:12]}"},
        json.dumps(raw, separators=(",", ":")).encode(),
    )


def test_receipt_first_crash_is_recovered_without_duplicate_post(tmp_path: Path) -> None:
    config, requests_path = _one_request(tmp_path)
    work = tmp_path / "transport"
    exact = transport.prepare_exact_byte_preflight(config, requests_path, work, max_cost_usd=10.0)
    request = pf.read_jsonl(requests_path)[0]
    paths = transport._transport_paths(work, request["request_id"])
    pf.publish_exact_json(paths["preparing"], transport._attempt_event(request, exact, "preparing"))
    pf.publish_exact_json(
        paths["receipt"],
        transport._receipt_from_result(request, exact, _success(request["request_id"]), "accepted"),
    )

    calls = 0

    def forbidden_post(*_args: object) -> transport.HTTPResult:
        nonlocal calls
        calls += 1
        raise AssertionError("a recovered receipt must suppress POST")

    result = transport.run_transport(
        config,
        requests_path,
        work,
        verify_only=True,
        post=forbidden_post,
    )
    assert result["status"] == "complete_verified"
    assert calls == 0
    assert paths["response"].is_file()
    assert paths["completed"].is_file()


def test_complete_resume_needs_no_api_key_and_never_reposts(tmp_path: Path) -> None:
    config, requests_path = _one_request(tmp_path)
    work = tmp_path / "transport"
    calls = 0

    def fake_post(
        _url: str, _headers: object, _body: bytes, _timeout: float
    ) -> transport.HTTPResult:
        nonlocal calls
        calls += 1
        return _success(pf.read_jsonl(requests_path)[0]["request_id"])

    first = transport.run_transport(
        config,
        requests_path,
        work,
        max_cost_usd=10.0,
        api_key="test-key",
        post=fake_post,
    )
    assert first["status"] == "complete_verified"
    assert calls == 1

    second = transport.run_transport(
        config,
        requests_path,
        work,
        verify_only=True,
        post=fake_post,
    )
    assert second == first
    assert calls == 1


def test_preparing_without_receipt_is_ambiguous_and_never_retried(tmp_path: Path) -> None:
    config, requests_path = _one_request(tmp_path)
    work = tmp_path / "transport"
    exact = transport.prepare_exact_byte_preflight(config, requests_path, work, max_cost_usd=10.0)
    request = pf.read_jsonl(requests_path)[0]
    paths = transport._transport_paths(work, request["request_id"])
    pf.publish_exact_json(paths["preparing"], transport._attempt_event(request, exact, "preparing"))
    calls = 0

    def forbidden_post(*_args: object) -> transport.HTTPResult:
        nonlocal calls
        calls += 1
        return _success(request["request_id"])

    with pytest.raises(RuntimeError, match="blocked requests"):
        transport.run_transport(
            config,
            requests_path,
            work,
            api_key="test-key",
            post=forbidden_post,
        )
    assert calls == 0


def test_response_without_receipt_fails_closed(tmp_path: Path) -> None:
    config, requests_path = _one_request(tmp_path)
    work = tmp_path / "transport"
    transport.prepare_exact_byte_preflight(config, requests_path, work, max_cost_usd=10.0)
    request = pf.read_jsonl(requests_path)[0]
    paths = transport._transport_paths(work, request["request_id"])
    pf.publish_exact_json(
        paths["response"], {"request_id": request["request_id"], "raw_response": {}}
    )
    with pytest.raises(RuntimeError, match="without its provider receipt"):
        transport.run_transport(config, requests_path, work, verify_only=True)


def test_preflight_is_exact_byte_immutable_after_transport_evidence(tmp_path: Path) -> None:
    config, requests_path = _one_request(tmp_path)
    work = tmp_path / "transport"
    transport.prepare_exact_byte_preflight(config, requests_path, work, max_cost_usd=10.0)
    request = pf.read_jsonl(requests_path)[0]
    paths = transport._transport_paths(work, request["request_id"])
    paths["preparing"].parent.mkdir(parents=True)
    paths["preparing"].write_text("{}\n", encoding="utf-8")
    requests_path.write_bytes(requests_path.read_bytes() + b"\n")
    with pytest.raises((FileExistsError, ValueError), match="preflight|request"):
        transport.prepare_exact_byte_preflight(config, requests_path, work, max_cost_usd=10.0)


def test_api_key_is_required_only_at_new_post_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, requests_path = _one_request(tmp_path)
    work = tmp_path / "transport"
    transport.prepare_exact_byte_preflight(config, requests_path, work, max_cost_usd=10.0)
    monkeypatch.delenv(transport.DEFAULT_API_KEY_ENV, raising=False)
    with pytest.raises(RuntimeError, match="required only for a new judge POST"):
        transport.run_transport(config, requests_path, work, post=lambda *_: _success("x" * 64))
    # Credential failure occurs before the preparing marker or any POST evidence.
    request_id = pf.read_jsonl(requests_path)[0]["request_id"]
    assert not transport._transport_paths(work, request_id)["preparing"].exists()

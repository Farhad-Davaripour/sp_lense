"""Max-only request accounting through the one reviewed canonical helper.

Nothing here initializes a ledger. A Root-authored, hash-pinned runtime binding
and reconciliation are required at the HTTP boundary. Receipts contain model
and usage metadata, never prompts, generated text or credentials.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import uuid


CANONICAL = Path("C:/Users/farha/repos/sp_lense4/research4")
HELPER_PATH = CANONICAL / "mac_studio/qwen_cumulative_budget_v1.py"
HELPER_SHA256 = "2b320f4aaa0f64e8e1ed47565108d44bb53bae8750ef2b9b0ea7bec965402502"
LEDGER_PATH = CANONICAL / ".local/monitor/QWEN_CUMULATIVE_BUDGET_LEDGER.json"
BASIS_PATH = CANONICAL / ".local/experiments/general_preservation_r16_last8_v1/post_fit_dev_v1/reviews/account_admission_20261110/MAX_ONLY_BUDGET_ADMISSION_BASIS.json"
BASIS_SHA256 = "876cb6e702eb49164777a22ecc8805bf0ab086cfb337ccc61bdaeb56fe19558d"
PROJECT_LOCAL = Path(__file__).resolve().parents[1] / ".local"
BINDING_ROOTS = (PROJECT_LOCAL, CANONICAL / ".local")
RECEIPT_ROOT = PROJECT_LOCAL
BINDING_ENV = "RESEARCH4_QWEN_BUDGET_BINDING"
BINDING_SHA_ENV = "RESEARCH4_QWEN_BUDGET_BINDING_SHA256"
SCHEMA = "research4.game-qwen-max-budget-binding.v1"
ENDPOINT = "https://ap-southeast-1.maas.qwencloudapi.com/compatible-mode/v1"
INPUT_RESERVE = 1_000_000
MAX_TEXT_MESSAGE_BYTES = INPUT_RESERVE - 16_384
WIRE_OUTPUT = 8192
OUTPUT_RESERVE = 8202
_worker_halted = False


class SpendingBlocked(ValueError):
    pass


def _path(value):
    path = Path(value).resolve()
    if any("onedrive" in part.lower() for part in path.parts):
        raise SpendingBlocked("Budget artifacts must stay outside OneDrive")
    return path


def _verified(item, fixed_path=None, fixed_sha=None):
    if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
        raise SpendingBlocked("Exact budget evidence path and SHA256 are required")
    path = _path(item["path"])
    if fixed_path is not None and path != _path(fixed_path):
        raise SpendingBlocked("Budget helper/evidence path differs from the reviewed source")
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != item["sha256"] or fixed_sha is not None and digest != fixed_sha:
        raise SpendingBlocked("Pinned budget evidence changed")
    return path, data


def _load_binding():
    name, digest = os.environ.get(BINDING_ENV), os.environ.get(BINDING_SHA_ENV)
    if not name or not digest:
        raise SpendingBlocked("Root-pinned Qwen spending binding is required before HTTP")
    path = _path(name)
    if not any(path.is_relative_to(_path(root)) for root in BINDING_ROOTS):
        raise SpendingBlocked("Budget binding must be under an authorized ignored local directory")
    _, data = _verified({"path": str(path), "sha256": digest})
    binding = json.loads(data)
    required = {"schema", "helper", "ledger_path", "attestation", "admission_basis",
        "account_id", "credential_sha256", "model", "region", "route", "endpoint",
        "input_token_ceiling", "wire_output_tokens", "output_token_ceiling",
        "input_usd_per_million", "output_usd_per_million", "receipt_directory"}
    if not isinstance(binding, dict) or set(binding) != required or binding["schema"] != SCHEMA:
        raise SpendingBlocked("Reviewed Max spending-binding schema is required")
    if (_path(binding["ledger_path"]) != _path(LEDGER_PATH)
            or binding["model"] != "qwen3.8-max" or binding["region"] != "Singapore"
            or binding["endpoint"] != ENDPOINT
            or binding["input_token_ceiling"] != INPUT_RESERVE
            or binding["wire_output_tokens"] != WIRE_OUTPUT
            or binding["output_token_ceiling"] != OUTPUT_RESERVE
            or binding["input_usd_per_million"] != "2"
            or binding["output_usd_per_million"] != "6"):
        raise SpendingBlocked("Exact Max-only account route, paid-list rates and fiscal ceilings required")
    for field in ("account_id", "route", "credential_sha256"):
        if not isinstance(binding[field], str) or not binding[field].strip():
            raise SpendingBlocked("Account, route and credential identity must be bound")
    receipts = _path(binding["receipt_directory"])
    if not receipts.is_relative_to(_path(RECEIPT_ROOT)):
        raise SpendingBlocked("Financial receipts must remain under the game's ignored local directory")
    helper_path, helper_bytes = _verified(binding["helper"], HELPER_PATH, HELPER_SHA256)
    _verified(binding["admission_basis"], BASIS_PATH, BASIS_SHA256)
    attest_path, attest_bytes = _verified(binding["attestation"])
    attestation = json.loads(attest_bytes)
    for field in ("account_id", "model", "region", "route", "endpoint",
                  "input_usd_per_million", "output_usd_per_million"):
        if attestation.get(field) != binding[field]:
            raise SpendingBlocked("Spending binding and Root reconciliation differ")
    state = json.loads(_path(LEDGER_PATH).read_text(encoding="utf-8"))
    bound = state.get("attestation", {})
    if (_path(bound.get("path", "")) != attest_path or
            bound.get("sha256") != binding["attestation"]["sha256"]):
        raise SpendingBlocked("Canonical ledger is bound to another reconciliation")
    return binding, path, digest, helper_path, helper_bytes


def _load_helper(path, source):
    # Execute only bytes already verified against the fixed reviewed source.
    # __file__ stays canonical, so the helper enforces its single ledger path.
    spec = importlib.util.spec_from_file_location("research4_game_budget_canonical", path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, str(path), "exec"), module.__dict__)
    if _path(module.LEDGER_PATH) != _path(LEDGER_PATH):
        raise SpendingBlocked("Canonical helper ledger identity differs")
    return module


def _write_receipt(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


class RequestReservation:
    def __init__(self, binding, binding_path, binding_sha, helper, permit):
        self.binding, self.helper, self.permit = binding, helper, permit
        self.request_id = permit["request_id"]
        self.directory = _path(binding["receipt_directory"])
        self.metadata = {"schema": "research4.game-qwen-request-accounting.v1",
            "request_id": self.request_id, "request_id_type": "local ledger reservation ID",
            "model": binding["model"], "account_id": binding["account_id"],
            "credential_sha256": binding["credential_sha256"],
            "binding": {"path": str(binding_path), "sha256": binding_sha},
            "helper": binding["helper"], "ledger_path": binding["ledger_path"],
            "attestation": binding["attestation"], "permit": permit,
            "wire_output_tokens": WIRE_OUTPUT,
            "scope": "Conservative local commitment, not a provider invoice"}

    def ambiguous(self, reason, http_status=None):
        global _worker_halted
        _worker_halted = True
        try:
            self.helper.mark_ambiguous(self.binding["ledger_path"], self.request_id, reason)
        finally:
            _write_receipt(self.directory / f"{self.request_id}_ERROR.json",
                {**self.metadata, "error_type": reason, "http_status": http_status,
                 "usage_unknown_or_not_admitted": True, "automatic_retry": False})

    def response(self, metadata):
        global _worker_halted
        try:
            # Preserve exact raw model/usage metadata BEFORE any settlement.
            _write_receipt(self.directory / f"{self.request_id}_PROVIDER_RESPONSE.json",
                           {**self.metadata, "provider_response_metadata": metadata})
        except BaseException:
            self.ambiguous("ProviderReceiptWriteError")
            raise
        raw = metadata.get("usage")
        if metadata.get("model") != self.binding["model"]:
            self.ambiguous("ReturnedModelMismatch")
            raise SpendingBlocked("Returned Qwen model differs; reservation retained")
        if (not isinstance(raw, dict) or type(raw.get("prompt_tokens")) is not int
                or type(raw.get("completion_tokens")) is not int
                or raw["prompt_tokens"] <= 0 or raw["completion_tokens"] < 0
                or raw.get("total_tokens") is not None and
                (type(raw["total_tokens"]) is not int or
                 raw["total_tokens"] != raw["prompt_tokens"] + raw["completion_tokens"])):
            self.ambiguous("UnknownOrInconsistentProviderUsage")
            raise SpendingBlocked("Exact total provider usage required; reservation retained")
        try:
            # Raw prompt/completion totals already contain cache/thinking.
            # The helper itself retains+halts for any over-ceiling observation.
            self.helper.settle_request(self.binding["ledger_path"], self.request_id,
                                       raw["prompt_tokens"], raw["completion_tokens"])
        except BaseException:
            _worker_halted = True
            self.helper.halt_ledger(self.binding["ledger_path"], "SettlementFailedOrOverCeiling")
            raise


def reserve_for_request(request, *, endpoint, credential, current_credential=None):
    """Reserve at the HTTP boundary; no credential-string test bypass exists."""
    global _worker_halted
    if _worker_halted:
        raise SpendingBlocked("This Max worker is halted pending Root reconciliation")
    ordinary_fields = {"model", "messages", "tools", "tool_choice", "extra_headers",
        "max_completion_tokens", "n", "extra_body", "stream", "stream_options",
        "frequency_penalty", "presence_penalty", "stop", "logit_bias", "seed",
        "temperature", "top_p", "logprobs", "top_logprobs", "response_format"}
    if (request.get("model") != "qwen3.8-max" or str(endpoint).rstrip("/") != ENDPOINT
            or set(request) - ordinary_fields
            or type(request.get("max_completion_tokens")) is not int
            or request.get("max_completion_tokens") != WIRE_OUTPUT
            or request.get("n", 1) != 1
            or "max_tokens" in request or bool(request.get("tools"))
            or request.get("extra_body") != {"enable_thinking": True, "reasoning_effort": "low"}):
        raise SpendingBlocked("Budget admission covers only ordinary text Max chat at the fixed output cap")
    # Only Inspect's local trace-correlation header may supplement SDK auth.
    # A per-request Authorization/project override could switch the paid account.
    headers = request.get("extra_headers", {})
    if (not isinstance(headers, dict) or any(
            not isinstance(name, str) or name.lower() != "x-irid"
            or not isinstance(value, str) for name, value in headers.items())):
        raise SpendingBlocked("Per-request account or routing headers are not admitted")
    # Native image/audio content and separately priced features are out of scope.
    messages = request.get("messages")
    if not isinstance(messages, list) or not 1 <= len(messages) <= 4 or any(
            not isinstance(row, dict) or set(row) - {"role", "content"}
            or row.get("role") not in {"system", "user", "assistant"}
            or not isinstance(row.get("content"), str)
            for row in messages):
        raise SpendingBlocked("Budget admission covers text-only messages")
    # The shared game sends two plain-text messages. Keep their serialized
    # byte envelope below the fiscal ceiling with a generous template margin;
    # reject excess input rather than truncate or repair it.
    if len(json.dumps(messages, ensure_ascii=False).encode("utf-8")) > MAX_TEXT_MESSAGE_BYTES:
        raise SpendingBlocked("Text input exceeds the conservative fiscal input ceiling")
    binding, path, digest, helper_path, helper_source = _load_binding()
    keys = [credential] if current_credential is None else [credential, current_credential]
    if any(not isinstance(key, str) or not key or
           hashlib.sha256(key.encode()).hexdigest() != binding["credential_sha256"] for key in keys):
        raise SpendingBlocked("Qwen credential differs from the attested account binding")
    helper = _load_helper(helper_path, helper_source)
    summary = helper.budget_summary(binding["ledger_path"])
    if summary.get("halted"):
        raise SpendingBlocked("Canonical Max ledger is halted")
    if summary.get("outstanding_requests") != 0:
        helper.halt_ledger(binding["ledger_path"], "OutstandingPriorRequestRequiresRootReconciliation")
        _worker_halted = True
        raise SpendingBlocked("Prior Max reservation remains unresolved; no HTTP request")
    request_id = "game-max-" + uuid.uuid4().hex
    permit = helper.reserve_request(binding["ledger_path"], request_id, INPUT_RESERVE, OUTPUT_RESERVE,
        expected_account_id=binding["account_id"], expected_model=binding["model"],
        expected_region=binding["region"], expected_route=binding["route"],
        expected_endpoint=binding["endpoint"])
    reservation = RequestReservation(binding, path, digest, helper, permit)
    if (permit.get("attestation_sha256") != binding["attestation"]["sha256"] or
            permit.get("input_token_ceiling") != INPUT_RESERVE or
            permit.get("output_token_ceiling") != OUTPUT_RESERVE or
            permit.get("input_usd_per_million") != "2" or permit.get("output_usd_per_million") != "6"):
        reservation.ambiguous("ReservationPermitMismatch")
        raise SpendingBlocked("Canonical reservation differs from the Root binding")
    try:
        _write_receipt(reservation.directory / f"{request_id}_RESERVATION.json", reservation.metadata)
    except BaseException:
        reservation.ambiguous("ReservationReceiptWriteError")
        raise
    return reservation

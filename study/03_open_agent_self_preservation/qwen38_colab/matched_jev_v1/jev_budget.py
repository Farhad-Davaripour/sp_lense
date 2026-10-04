"""Persistent bounded Jev phase ledger. Standard library; no credentials or HTTP.

The 512-attempt / $1 phase allowance is separate from the eight historical calls.
Reservations and exact requests are persisted before a transport is invoked.
Uncertain failed-call charges stay reserved, and failures prohibit automatic retry.
"""
import datetime
import hashlib
import json
import math
import os
import threading
from pathlib import Path

MAX_ATTEMPTS = 512
MAX_SUCCESSFUL_REQUESTS = 512
MAX_ESTIMATED_INPUT_USD = 1.0
PRIOR_REQUESTS = 8
PRIOR_ESTIMATED_INPUT_USD = 0.001348872
RESERVE_BYTES = 1024
OFFICIAL_ENDPOINT = "https://api.typesafe.ai/v1/systemone"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def number(value, label, upper=None):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0
            and (upper is None or value <= upper), "Invalid " + label)
    return float(value)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, indent=2, ensure_ascii=False, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def validate_readiness(record, model, now=None):
    require(record.get("verified") is True and record.get("endpoint") == OFFICIAL_ENDPOINT
            and record.get("model") == model, "Unverified official endpoint/model readiness")
    require(record.get("official_docs_verified") is True
            and record.get("versioned_model_id_documented") is True,
            "Official versioned model documentation not verified")
    require(record.get("official_models_doc_url") == "https://docs.typesafe.ai/models"
            and record.get("official_api_doc_url") == "https://docs.typesafe.ai/api",
            "Official documentation provenance differs")
    require("choice" in record.get("supported_question_types", [])
            and record.get("multiple_questions_verified") is True,
            "Typed choice or parallel question map not verified")
    require(record.get("published_request_limit") == "64k"
            and record.get("published_state_longest_question_limit") == "32k",
            "Published request/context limits differ")
    rate = number(record.get("input_usd_per_million"), "published input price")
    require(number(record.get("output_usd_per_million"), "published output price") == 0,
            "Documented free output price differs")
    checked = datetime.datetime.fromisoformat(record["checked_at_utc"].replace("Z", "+00:00"))
    require(checked.tzinfo is not None, "Readiness timestamp requires timezone")
    age = ((now or datetime.datetime.now(datetime.timezone.utc)) - checked).total_seconds()
    require(0 <= age <= 86400, "Official readiness must be verified within 24 hours")
    return rate


def request_size_reserve(request):
    require(set(request) == {"model", "state", "questions"}, "Unexpected request shape")
    require(isinstance(request["questions"], dict) and request["questions"], "Missing questions")
    total = len(canonical_bytes(request)) + RESERVE_BYTES
    state_longest = (len(canonical_bytes(request["state"]))
                     + max(len(canonical_bytes(q)) for q in request["questions"].values())
                     + RESERVE_BYTES)
    require(total <= 64000, "Conservative request reserve exceeds published 64k limit")
    require(state_longest <= 32000, "State plus longest-question reserve exceeds 32k limit")
    return {"total_input_equivalent": total, "state_longest_input_equivalent": state_longest,
            "method": "UTF-8 canonical JSON bytes plus 1024 reserve; not measured provider tokens"}


class PhaseBudget:
    """One-process owner, four-thread admission with crash-persistent reservations."""
    def __init__(self, directory, readiness, model, schema_version, questions_sha256,
                 schema_source_sha256):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.readiness = readiness
        self.rate = validate_readiness(readiness, model)
        self.model = model
        self.lock = threading.RLock()
        self.process_lock = self.directory / "SERVICE_PROCESS.lock"
        try:
            descriptor = os.open(self.process_lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise RuntimeError("Phase service already active or prior process requires reconciliation") from None
        with os.fdopen(descriptor, "w", encoding="utf-8") as owner:
            owner.write(str(os.getpid()) + "\n")
        self.closed = False
        try:
            self.plan = {"model": model, "endpoint": OFFICIAL_ENDPOINT,
                         "schema_version": schema_version, "questions_sha256": questions_sha256,
                         "schema_source_sha256": schema_source_sha256,
                         "max_attempts": MAX_ATTEMPTS, "max_successful_requests": MAX_SUCCESSFUL_REQUESTS,
                         "max_estimated_phase_input_usd": MAX_ESTIMATED_INPUT_USD,
                         "input_usd_per_million": self.rate, "output_usd_per_million": 0,
                         "prior_requests_separate": PRIOR_REQUESTS,
                         "prior_estimated_input_usd_separate": PRIOR_ESTIMATED_INPUT_USD,
                         "no_automatic_retries_or_topups": True}
            plan_path = self.directory / "PHASE_CONFIG.json"
            if plan_path.exists():
                require(json.loads(plan_path.read_text(encoding="utf-8")) == self.plan,
                        "Persisted phase budget/model/schema differs; no silent reset")
            else:
                write_json(plan_path, self.plan)
            status_path = self.directory / "STATUS.json"
            self.state = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {
                "attempted_requests": 0, "successful_requests": 0, "input_tokens": 0,
                "reserved_input_equivalent": 0, "inflight": {}, "uncertain_failures": {},
                "stopped": False, "stop_reason": None,
                "phase_started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
            require(isinstance(self.state["attempted_requests"], int)
                    and not isinstance(self.state["attempted_requests"], bool)
                    and 0 <= self.state["successful_requests"] <= self.state["attempted_requests"] <= MAX_ATTEMPTS,
                    "Invalid persisted request counts")
            require(self.state["successful_requests"] <= MAX_SUCCESSFUL_REQUESTS,
                    "Persisted successful request cap exceeded")
            for field in ("input_tokens", "reserved_input_equivalent"):
                require(isinstance(self.state[field], int) and not isinstance(self.state[field], bool)
                        and self.state[field] >= 0, "Invalid persisted usage ledger")
            if self.state["inflight"]:
                self.state["stopped"] = True
                self.state["stop_reason"] = "Unresolved prior inflight requests; no automatic retry"
            self._persist()
        except BaseException:
            self.close()
            raise

    def _persist(self):
        write_json(self.directory / "STATUS.json", self.status())

    def status(self):
        with self.lock:
            result = dict(self.state)
            result["inflight"] = dict(self.state["inflight"])
            result["uncertain_failures"] = dict(self.state["uncertain_failures"])
            result.update(input_usd_per_million=self.rate,
                          completed_input_estimated_usd=self.state["input_tokens"] * self.rate / 1e6,
                          committed_plus_reserved_input_estimated_usd=(self.state["input_tokens"]
                              + self.state["reserved_input_equivalent"]) * self.rate / 1e6,
                          max_attempts=MAX_ATTEMPTS, max_successful_requests=MAX_SUCCESSFUL_REQUESTS,
                          max_estimated_phase_input_usd=MAX_ESTIMATED_INPUT_USD,
                          prior_requests_separate=PRIOR_REQUESTS,
                          prior_estimated_input_usd_separate=PRIOR_ESTIMATED_INPUT_USD,
                          cumulative_successful_requests_including_prior=self.state["successful_requests"] + PRIOR_REQUESTS,
                          no_automatic_retries_or_topups=True)
            return result

    def begin_explicit_recovery(self, identity, request):
        """One root CLI replay of an exact single failed request; no automatic retry."""
        with self.lock:
            request_sha = identity["request_sha256"]
            require(self.state["stopped"] and not self.state["inflight"],
                    "Recovery requires a stopped phase without inflight calls")
            require(set(self.state["uncertain_failures"]) == {request_sha},
                    "Recovery requires exactly one known failed request")
            require(not self.state.get("explicit_recovery"), "The single explicit replay was already authorized")
            require(not (self.directory / "cache" / (request_sha + ".json")).exists(),
                    "Recovery cannot replay a cached successful response")
            prior = json.loads((self.directory / "requests" / (request_sha + ".json")).read_text(encoding="utf-8"))
            require(prior["identity"] == identity and prior["request"] == request,
                    "Recovery request differs from the exact prior freeze")
            event = {"request_identity": identity, "request": request,
                     "original_uncertain_input_equivalent": self.state["uncertain_failures"][request_sha],
                     "root_cli_flag_required": True, "maximum_explicit_replays": 1,
                     "original_failed_reserve_retained": True, "phase_caps_unchanged": True,
                     "authorized_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
            write_json(self.directory / "EXPLICIT_RECOVERY.json", event)
            self.state["explicit_recovery"] = {"request_sha256": request_sha,
                "replay_pending": True, "replay_attempted": False, "replay_completed": False}
            self.state["stopped"] = False
            self.state["stop_reason"] = None
            self._persist()

    def reserve(self, identity, request):
        with self.lock:
            validate_readiness(self.readiness, self.model)
            require(not self.state["stopped"], "Phase stopped; no automatic retry")
            request_sha = identity["request_sha256"]
            require(sha256(canonical_bytes(request)) == request_sha, "Request digest differs before admission")
            recovery = self.state.get("explicit_recovery") or {}
            is_explicit_replay = recovery.get("request_sha256") == request_sha and recovery.get("replay_pending") is True
            require(request_sha not in self.state["inflight"]
                    and (request_sha not in self.state["uncertain_failures"] or is_explicit_replay),
                    "Request already unresolved; no automatic retry")
            size = request_size_reserve(request)
            reserve = size["total_input_equivalent"]
            require(self.state["attempted_requests"] < MAX_ATTEMPTS, "Phase 512-attempt cap reached")
            require(self.state["successful_requests"] + len(self.state["inflight"]) < MAX_SUCCESSFUL_REQUESTS,
                    "Phase 512-success cap reached")
            estimated = (self.state["input_tokens"] + self.state["reserved_input_equivalent"] + reserve) * self.rate / 1e6
            require(estimated <= MAX_ESTIMATED_INPUT_USD, "Phase $1 input admission cap reached")
            freeze = {"identity": identity, "request": request, "readiness": self.readiness,
                      "preflight_reserve": size, "frozen_before_dispatch": True}
            target = self.directory / "requests" / (request_sha + (".replay1.json" if is_explicit_replay else ".json"))
            require(not target.exists(), "Previously frozen uncached request must not be retried")
            write_json(target, freeze)
            self.state["attempted_requests"] += 1
            if is_explicit_replay:
                recovery["replay_pending"] = False
                recovery["replay_attempted"] = True
                recovery["attempt_number"] = self.state["attempted_requests"]
            self.state["reserved_input_equivalent"] += reserve
            self.state["inflight"][request_sha] = reserve
            self._persist()
            return size

    def complete(self, identity, input_tokens):
        require(isinstance(input_tokens, int) and not isinstance(input_tokens, bool) and input_tokens >= 0,
                "Invalid provider input usage")
        with self.lock:
            request_sha = identity["request_sha256"]
            reserve = self.state["inflight"].pop(request_sha)
            self.state["reserved_input_equivalent"] -= reserve
            self.state["input_tokens"] += input_tokens
            self.state["successful_requests"] += 1
            recovery = self.state.get("explicit_recovery") or {}
            if recovery.get("request_sha256") == request_sha and recovery.get("replay_attempted"):
                recovery["replay_completed"] = True
            if input_tokens > reserve or (self.state["input_tokens"] + self.state["reserved_input_equivalent"]) * self.rate / 1e6 > MAX_ESTIMATED_INPUT_USD:
                self.state["stopped"] = True
                self.state["stop_reason"] = "Observed usage exceeded conservative reserve or phase cap"
            self._persist()
            require(input_tokens <= reserve, "Observed usage exceeded reserve; phase stopped")

    def fail(self, identity, error_type, http_status=None, stage=None, reason_code=None):
        with self.lock:
            request_sha = identity["request_sha256"]
            reserve = self.state["inflight"].pop(request_sha, None)
            if reserve is not None:
                self.state["uncertain_failures"][request_sha] = self.state["uncertain_failures"].get(request_sha, 0) + reserve
            self.state["stopped"] = True
            self.state["stop_reason"] = "Request failed; no automatic retry"
            failure = {"request_identity": identity, "error_type": error_type,
                       "http_status": http_status if isinstance(http_status, int) else None,
                       "uncertain_charge_reserve_retained": reserve,
                       "error_body_headers_and_exception_text_saved": False}
            if stage is not None:
                failure["validation_stage"] = stage
            if reason_code is not None:
                failure["own_static_reason_code"] = reason_code
            failure_path = self.directory / "failures" / (request_sha + ".json")
            if failure_path.exists():
                failure_path = failure_path.with_name(request_sha + ".attempt" + str(self.state["attempted_requests"]) + ".json")
            write_json(failure_path, failure)
            self._persist()

    def close(self):
        if not self.closed:
            self.closed = True
            self.process_lock.unlink(missing_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, _kind, _value, _traceback):
        self.close()

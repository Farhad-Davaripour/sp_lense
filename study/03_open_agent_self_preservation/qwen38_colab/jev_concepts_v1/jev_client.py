"""Bounded trusted-controller Jev client, adapted from immutable Research2.

No credential discovery/environment reads. No call occurs without an injected
transport and freshly verified official metadata. Keys never enter payloads,
cache records, learner environments or notebook source.
"""
import datetime
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path
from concept_schema import MODEL, ENDPOINT, QUESTIONS, SCHEMA_VERSION, payload, request_identity, canonical_bytes, digest

MAX_REQUESTS = 128
MAX_ESTIMATED_INPUT_USD = 0.05
HISTORICAL_INPUT_USD_PER_MILLION = 0.042  # Not a current-price verification.
RESERVE_BYTES = 1024


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Research2 credential boundary: never forward credentials on a redirect."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "Redirect refused", headers, fp)


def validated_number(value, label, upper=None):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0 and (upper is None or value <= upper),
            "Invalid "+label)
    return float(value)


def validate_readiness(record, now=None):
    require(record.get("verified") is True, "Official API readiness not verified")
    require(record.get("endpoint") == ENDPOINT and record.get("model") == MODEL,
            "Endpoint/model differs from pinned Jev")
    require(record.get("official_docs_verified") is True
            and record.get("versioned_model_id_documented") is True,
            "Current official model/version and API documentation must be verified")
    require(record.get("official_models_doc_url") == "https://docs.typesafe.ai/models"
            and record.get("official_api_doc_url") == "https://docs.typesafe.ai/api",
            "Missing official model/API documentation provenance")
    require("choice" in record.get("supported_question_types", [])
            and record.get("multiple_questions_verified") is True,
            "Current typed choice and parallel question map are not verified")
    # No numeric question-count maximum is invented: docs describe a question
    # map and parallel questions, with total and state/longest-question limits.
    require(record.get("published_request_limit") == "64k"
            and record.get("published_state_longest_question_limit") == "32k",
            "Published request/context limits are not verified")
    rate = validated_number(record.get("input_usd_per_million"), "current published input price")
    require(validated_number(record.get("output_usd_per_million"), "published output price") == 0,
            "Current documented output-free pricing must be verified")
    checked = datetime.datetime.fromisoformat(record["checked_at_utc"].replace("Z", "+00:00"))
    require(checked.tzinfo is not None, "Readiness timestamp lacks timezone")
    now = now or datetime.datetime.now(datetime.timezone.utc)
    age = (now-checked).total_seconds()
    require(0 <= age <= 86400, "Official model/type/rate verification must be fresh within 24 hours")
    # Public docs/version or an alias-only model list do not prove account
    # access. The first bounded authenticated response proves it separately.
    return rate


def request_size_reserve(request):
    total = len(canonical_bytes(request))+RESERVE_BYTES
    state = len(canonical_bytes(request["state"]))
    longest = max(len(canonical_bytes(question)) for question in request["questions"].values())
    state_longest = state+longest+RESERVE_BYTES
    require(total <= 64000, "Conservative total request reserve exceeds published 64k limit")
    require(state_longest <= 32000, "Conservative state plus longest-question reserve exceeds published 32k limit")
    return {"total_input_equivalent": total, "state_longest_input_equivalent": state_longest,
            "method": "UTF-8 JSON bytes plus 1024 planning reserve, not a provider tokenizer measurement",
            "exact_multiquestion_billing_formula_verified": False}


def decode(response):
    # Exact Research2 model/choice/probability/argmax validation, generalized
    # across seven factual choice questions. No ground-truth fallback.
    require(response.get("model") == MODEL, "Unexpected Jev model version")
    answers = response["answers"]
    require(set(answers) == set(QUESTIONS), "Incomplete or extra concept answers")
    result = {}
    for name, question in QUESTIONS.items():
        answer = answers[name]
        require(answer.get("type") == "choice", "Unexpected typed answer")
        probabilities = answer["probabilities"]
        require(set(probabilities) == set(question["criteria"]), "Unexpected concept class labels")
        values = [validated_number(value, "concept probability", 1) for value in probabilities.values()]
        require(abs(sum(values)-1) < 1e-5, "Probabilities do not sum to one")
        chosen = answer["choice"]
        require(chosen in probabilities and probabilities[chosen] == max(probabilities.values()),
                "Chosen concept is not a maximum-probability class")
        confidence = validated_number(answer["confidence"], "provider concept confidence", 1)
        result[name] = {"type": "choice", "value": chosen, "confidence": confidence,
                        "choice_probability": float(probabilities[chosen]), "probabilities": probabilities}
    for field in ("input_tokens", "output_tokens"):
        used = response["usage"][field]
        require(isinstance(used, int) and not isinstance(used, bool) and used >= 0,
                "Missing valid provider "+field+" usage")
    return {"model": MODEL, "schema_version": SCHEMA_VERSION, "concepts": result,
            "usage": response["usage"]}


def make_official_transport(credential_provider):
    """Only the trusted controller supplies an already configured private key.

    This factory does not discover/read a key. Never put this transport or its
    provider in the model worker/notebook or serialize them.
    """
    require(callable(credential_provider), "Approved private credential provider required")
    def transport(request):
        secret = credential_provider()
        require(isinstance(secret, str) and bool(secret), "Approved Jev access is unavailable")
        req = urllib.request.Request(ENDPOINT, data=canonical_bytes(request), method="POST",
              headers={"Authorization": "Bearer "+secret, "Content-Type": "application/json"})
        with urllib.request.build_opener(NoRedirect()).open(req, timeout=30) as response:
            return json.load(response)
    return transport


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


class BoundedClient:
    def __init__(self, cache_dir, readiness, transport, max_requests=MAX_REQUESTS,
                 max_estimated_input_usd=MAX_ESTIMATED_INPUT_USD, max_seconds=1800):
        require(callable(transport), "Trusted-controller transport required")
        require(isinstance(max_requests, int) and 0 < max_requests <= MAX_REQUESTS,
                "Pilot request cap exceeds128")
        require(0 < max_estimated_input_usd <= MAX_ESTIMATED_INPUT_USD, "Pilot input estimate cap exceeds0.05USD")
        require(isinstance(max_seconds, (int, float)) and 0 < max_seconds <= 1800,
                "Pilot wall-time cap exceeds1800seconds")
        self.rate = validate_readiness(readiness)
        self.readiness = readiness
        self.transport = transport
        self.directory = Path(cache_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_requests = max_requests
        self.max_usd = max_estimated_input_usd
        self.max_seconds = max_seconds
        self.started = time.monotonic()
        self.started_wall = time.time()
        self.attempted = 0
        self.completed = 0
        self.input_tokens = 0
        self.reserved_tokens = 0
        self.stopped = False
        self.inflight = None
        plan = {"model": MODEL, "endpoint": ENDPOINT, "schema_version": SCHEMA_VERSION,
                "questions_sha256": digest(QUESTIONS), "max_requests": self.max_requests,
                "max_estimated_input_usd": self.max_usd, "max_seconds": self.max_seconds,
                "input_usd_per_million": self.rate}
        plan_path = self.directory/"PILOT_CONFIG.json"
        if plan_path.exists():
            require(json.loads(plan_path.read_text(encoding="utf-8")) == plan,
                    "Existing pilot budget/model/schema differs; no silent top-up")
        else:
            write_json(plan_path, plan)
        status_path = self.directory/"STATUS.json"
        if status_path.exists():
            prior = json.loads(status_path.read_text(encoding="utf-8"))
            self.started_wall = prior["started_wall_seconds"]
            self.attempted = prior["attempted_deduplicated_requests"]
            self.completed = prior["completed_requests"]
            self.input_tokens = prior["input_tokens"]
            self.reserved_tokens = prior["reserved_input_equivalent"]
            self.inflight = prior.get("inflight_request_sha256")
            self.stopped = bool(prior["stopped"] or self.inflight or (self.directory/"FAILURE.json").exists())
            require(self.attempted <= self.max_requests and self.completed <= self.attempted,
                    "Invalid persisted pilot request ledger")

    def call(self, history, tools):
        require(not self.stopped, "Pilot stopped; no automatic retry")
        validate_readiness(self.readiness)
        request = payload(history, tools)
        identity = request_identity(request)
        cache = self.directory/(identity["request_sha256"]+".json")
        if cache.exists():
            record = json.loads(cache.read_text(encoding="utf-8"))
            require(record["identity"] == identity and record["request"] == request,
                    "Cache identity differs")
            decoded = decode(record["response"])
            require(decoded == record["decoded"], "Cached typed response differs")
            return decoded, {"cache_hit": True, "identity": identity}
        size_reserve = request_size_reserve(request)
        reserve = size_reserve["total_input_equivalent"]
        require(self.attempted < self.max_requests, "Pilot deduplicated request cap reached")
        require(max(time.monotonic()-self.started, time.time()-self.started_wall) < self.max_seconds,
                "Pilot wall-time cap reached")
        require((self.input_tokens+self.reserved_tokens+reserve)*self.rate/1e6 <= self.max_usd,
                "Pilot estimated input cost cap reached")
        self.attempted += 1
        self.reserved_tokens += reserve
        self.inflight = identity["request_sha256"]
        write_json(self.directory/"STATUS.json", self.status())
        tick = time.monotonic()
        try:
            response = self.transport(request)
            decoded = decode(response)
            used = decoded["usage"]["input_tokens"]
            self.reserved_tokens -= reserve
            self.input_tokens += used
            self.completed += 1
            self.inflight = None
            record = {"identity": identity, "request": request, "response": response,
                      "decoded": decoded, "readiness": self.readiness,
                      "latency_seconds": time.monotonic()-tick,
                      "usage": decoded["usage"],
                      "input_estimated_usd": used*self.rate/1e6,
                      "output_usd_per_million_published": 0,
                      "output_estimated_usd_published": 0,
                      "preflight_reserve": size_reserve,
                      "observed_usage_exceeded_preflight_reserve": used > reserve,
                      "access_verified_by_pinned_response": True,
                      "cost_kind": "Estimated input from current published rate; output documented free. Provider usage/bill remains distinct."}
            write_json(cache, record)
            write_json(self.directory/"STATUS.json", self.status())
            require(used <= reserve, "Observed usage exceeds preflight reserve; stop without retries")
            require(self.input_tokens*self.rate/1e6 <= self.max_usd,
                    "Observed provider usage exceeded planned input cap; stop")
            return decoded, {"cache_hit": False, "identity": identity}
        except Exception as error:
            self.stopped = True
            write_json(self.directory/"STATUS.json", self.status())
            failure = {"error_type": type(error).__name__, "http_status": getattr(error, "code", None),
                       "attempted_requests": self.attempted, "completed_requests": self.completed,
                       "completed_input_tokens": self.input_tokens, "state": "stopped; no automatic retry",
                       "request_identity": identity}
            # Deliberately no exception text, headers, credential, or error body.
            write_json(self.directory/"FAILURE.json", failure)
            raise RuntimeError(json.dumps(failure)) from None

    def status(self):
        return {"attempted_deduplicated_requests": self.attempted, "completed_requests": self.completed,
                "input_tokens": self.input_tokens, "input_estimated_usd": self.input_tokens*self.rate/1e6,
                "reserved_input_equivalent": self.reserved_tokens, "input_usd_per_million": self.rate,
                "max_requests": self.max_requests, "max_estimated_input_usd": self.max_usd,
                "elapsed_seconds": max(time.monotonic()-self.started, time.time()-self.started_wall),
                "started_wall_seconds": self.started_wall, "stopped": self.stopped,
                "inflight_request_sha256": self.inflight,
                "no_automatic_retries_or_topups": True}

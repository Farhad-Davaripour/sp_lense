"""Trusted LOCAL Jev annotation service; never include it in a model worker.

CLI modes annotate_training and answer_batch receive files containing visible
messages/tools. Assistant training targets are never included in API payloads.
Only the approved TYPESAFE_API_KEY assignment is read. No credential is printed,
exported to an environment, stored in a packet, or forwarded on HTTP redirects.
"""
import argparse
import base64
import concurrent.futures
import gzip
import hashlib
import importlib
import json
import re
import shlex
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from jev_budget import (PhaseBudget, OFFICIAL_ENDPOINT, canonical_bytes, number,
                        require, sha256, validate_readiness, write_json)

APPROVED_DOTENV = Path("C:/Users/farha/repos/SP_lens/.env")
MAX_PARALLEL_CALLS = 4
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MARKER = "MATCHED_JEV_RESPONSE_GZIP_BASE64"
OWN_VALIDATION_REASONS = {
    "Unexpected pinned Jev response model": "WRONG_PINNED_MODEL",
    "Missing or extra concept answers": "ANSWER_QUESTION_KEYS_DIFFER",
    "Unexpected typed concept answer": "WRONG_ANSWER_TYPE",
    "Unexpected concept class labels": "CLASS_LABELS_DIFFER",
    "Invalid concept probability": "INVALID_CLASS_PROBABILITY",
    "Concept probabilities do not sum to one": "PROBABILITY_SUM_INVALID",
    "Chosen class is not a maximum-probability class": "CHOICE_NOT_ARGMAX",
    "Invalid provider concept confidence": "INVALID_PROVIDER_CONFIDENCE",
    "Missing provider usage": "MISSING_USAGE",
    "Invalid provider token usage": "INVALID_TOKEN_USAGE",
    "Shared schema decoder differs": "SHARED_DECODER_DIFFERS",
    "Duplicate JSON object key": "DUPLICATE_JSON_KEY",
    "Nonfinite JSON constant": "NONFINITE_JSON",
    "Provider response exceeds bounded size": "RESPONSE_SIZE_EXCEEDED",
    "Provider response unexpectedly includes credential bytes": "CREDENTIAL_ECHO_BLOCKED",
    "Observed usage exceeded reserve; phase stopped": "USAGE_RESERVE_EXCEEDED",
}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "Redirect refused", headers, fp)


def credential_provider(dotenv_path):
    require(Path(dotenv_path).resolve() == APPROVED_DOTENV.resolve(), "Credential source is not the approved dotenv")
    secret = None
    with Path(dotenv_path).open(encoding="utf-8-sig") as source:
        for line in source:
            match = re.match(r"^\s*(?:export\s+)?TYPESAFE_API_KEY\s*=\s*(.*)$", line)
            if not match:
                continue
            require(secret is None, "Duplicate requested credential assignment")
            lexer = shlex.shlex(match.group(1), posix=True)
            lexer.whitespace_split = True
            lexer.commenters = "#"
            values = list(lexer)
            require(len(values) == 1 and bool(values[0]), "Invalid requested credential assignment")
            secret = values[0]
    require(bool(secret), "Approved existing credential is absent")
    return secret


def strict_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON object key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(RuntimeError("Nonfinite JSON constant")))


def official_transport(request, dotenv_path):
    secret = credential_provider(dotenv_path)
    req = urllib.request.Request(OFFICIAL_ENDPOINT, data=canonical_bytes(request), method="POST",
          headers={"Authorization": "Bearer " + secret, "Content-Type": "application/json"})
    with urllib.request.build_opener(NoRedirect()).open(req, timeout=30) as response:
        raw = response.read(MAX_RESPONSE_BYTES + 1)
    require(len(raw) <= MAX_RESPONSE_BYTES, "Provider response exceeds bounded size")
    require(secret.encode("utf-8") not in raw, "Provider response unexpectedly includes credential bytes")
    # Parse only after the secret-scanned successful HTTP bytes are preserved.
    return None, raw


def decode(response, schema):
    require(isinstance(response, dict) and response.get("model") == schema.MODEL,
            "Unexpected pinned Jev response model")
    answers = response["answers"]
    require(isinstance(answers, dict) and set(answers) == set(schema.QUESTIONS), "Missing or extra concept answers")
    concepts = {}
    for name, question in schema.QUESTIONS.items():
        answer = answers[name]
        require(isinstance(answer, dict) and answer.get("type") == "choice", "Unexpected typed concept answer")
        probabilities = answer["probabilities"]
        require(isinstance(probabilities, dict) and set(probabilities) == set(question["criteria"]),
                "Unexpected concept class labels")
        values = [number(value, "concept probability", 1) for value in probabilities.values()]
        require(abs(sum(values) - 1) < 1e-5, "Concept probabilities do not sum to one")
        choice = answer["choice"]
        require(choice in probabilities and probabilities[choice] == max(probabilities.values()),
                "Chosen class is not a maximum-probability class")
        confidence = number(answer["confidence"], "provider concept confidence", 1)
        concepts[name] = {"type": "choice", "value": choice, "confidence": confidence,
                          "choice_probability": float(probabilities[choice]), "probabilities": probabilities}
    usage = response["usage"]
    require(isinstance(usage, dict), "Missing provider usage")
    for field in ("input_tokens", "output_tokens"):
        value = usage[field]
        require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                "Invalid provider token usage")
    decoded = {"model": schema.MODEL, "schema_version": schema.SCHEMA_VERSION,
               "concepts": concepts, "usage": usage}
    require(schema.decode(response) == decoded, "Shared schema decoder differs")
    return schema.decode(response)


class Service:
    def __init__(self, schema, schema_sha, phase, readiness, dotenv):
        self.schema = schema
        self.schema_sha = schema_sha
        self.phase = phase
        self.readiness = readiness
        self.dotenv = dotenv

    def call(self, row):
        history = row.get("history", row.get("messages"))
        request = self.schema.payload(history, row["tools"])
        identity = self.schema.request_identity(request)
        require(identity["request_sha256"] == sha256(canonical_bytes(request)), "Canonical request identity differs")
        require(request["model"] == self.schema.MODEL and request["questions"] == self.schema.QUESTIONS,
                "Request model/questions differ from pinned schema")
        target = self.phase.directory / "cache" / (identity["request_sha256"] + ".json")
        if target.exists():
            record = strict_json(target.read_bytes())
            require(record["identity"] == identity and record["request"] == request
                    and record["schema_source_sha256"] == self.schema_sha, "Cached state/questions/model/schema differ")
            raw = base64.b64decode(record["provider_response_raw_base64"], validate=True)
            require(sha256(raw) == record["provider_response_raw_sha256"]
                    and strict_json(raw) == record["response"], "Raw cached provider response differs")
            require(decode(record["response"], self.schema) == record["decoded"], "Cached typed concepts differ")
            require(self.schema.compact_aid(record["decoded"]) == record["compact_aid"], "Cached abstention/compact serialization differs")
            return {"id": row["id"], "annotation": record, "cache_hit": True}
        size = self.phase.reserve(identity, request)
        tick = time.monotonic()
        stage = "READINESS_RECHECK"
        try:
            validate_readiness(self.readiness, self.schema.MODEL)
            stage = "HTTP_TRANSPORT"
            _unused, raw = official_transport(request, self.dotenv)
            stage = "PRIVATE_RAW_HTTP_CAPTURE"
            # Inflight requests may be concurrent, so the request SHA is primary.
            raw_target = self.phase.directory / "raw_http" / (identity["request_sha256"] +
                (".replay1.json" if (self.phase.state.get("explicit_recovery") or {}).get("request_sha256") == identity["request_sha256"] else ".json"))
            require(not raw_target.exists(), "Private raw HTTP capture already exists")
            write_json(raw_target, {"identity": identity,
                "provider_response_raw_base64": base64.b64encode(raw).decode("ascii"),
                "provider_response_raw_sha256": sha256(raw),
                "http_successful_and_credential_scanned": True,
                "validation_complete": False, "private_only_never_student_input": True})
            stage = "JSON_PARSE"
            response = strict_json(raw)
            stage = "TYPED_DECODE"
            decoded = decode(response, self.schema)
            stage = "COMPACT_AND_INPUT_SERIALIZATION"
            compact = self.schema.compact_aid(decoded)
            # Validate actual fresh input serialization now; it is not sent to Jev.
            aided = self.schema.augmented_history(history, decoded)
            record = {"identity": identity, "request": request, "response": response, "decoded": decoded,
                      "provider_response_raw_base64": base64.b64encode(raw).decode("ascii"),
                      "provider_response_raw_sha256": sha256(raw), "compact_aid": compact,
                      "aided_history_sha256": self.schema.digest(aided),
                      "schema_source_sha256": self.schema_sha, "readiness": self.readiness,
                      "latency_seconds": time.monotonic() - tick,
                      "preflight_reserve": size, "usage": decoded["usage"],
                      "input_estimated_usd": decoded["usage"]["input_tokens"] * self.phase.rate / 1e6,
                      "access_verified_by_pinned_response": True,
                      "credential_in_payload_or_output": False, "training_targets_in_payload": False}
            stage = "VALIDATED_CACHE_WRITE"
            write_json(target, record)
            stage = "USAGE_LEDGER_COMPLETION"
            self.phase.complete(identity, decoded["usage"]["input_tokens"])
            return {"id": row["id"], "annotation": record, "cache_hit": False}
        except Exception as error:
            reason = OWN_VALIDATION_REASONS.get(str(error)) if type(error) is RuntimeError else None
            self.phase.fail(identity, type(error).__name__, getattr(error, "code", None), stage, reason)
            raise RuntimeError("Bounded Jev request stopped; see sanitized phase failure receipt") from None


def load_input(path):
    blob = Path(path).read_bytes()
    return strict_json(gzip.decompress(blob) if blob[:2] == b"\x1f\x8b" else blob), sha256(blob)


def run(mode, input_path, output_path, phase_dir, readiness_path, schema_sha256,
        dotenv=APPROVED_DOTENV, max_workers=4, recover_request_sha256=None):
    require(isinstance(max_workers, int) and 1 <= max_workers <= MAX_PARALLEL_CALLS,
            "Parallel real-call limit must be between one and four")
    output = Path(output_path)
    require(not output.exists(), "Output already exists; preserve prior response artifact")
    module_path = Path(__file__).with_name("concept_schema_v2.py")
    require(sha256(module_path.read_bytes()) == schema_sha256, "Pinned v2 schema source differs")
    schema = importlib.import_module("concept_schema_v2")
    require(schema.ENDPOINT == OFFICIAL_ENDPOINT, "Pinned endpoint differs")
    readiness = strict_json(Path(readiness_path).read_bytes())
    validate_readiness(readiness, schema.MODEL)
    original, input_sha = load_input(input_path)
    if mode == "answer_batch":
        require(isinstance(original, dict) and set(original) == {"batch_id", "rows"}
                and isinstance(original["batch_id"], str) and bool(original["batch_id"]),
                "Dynamic batch envelope must contain batch_id and rows")
        batch_id, rows = original["batch_id"], original["rows"]
        require(isinstance(rows, list) and 1 <= len(rows) <= 512, "Invalid dynamic request batch size")
        require(all(isinstance(row, dict) and set(row) == {"id", "history", "tools"} for row in rows),
                "Dynamic rows must contain only id, visible history and tools")
    else:
        rows = original if isinstance(original, list) else original.get("rows", original.get("train"))
        require(isinstance(rows, list) and len(rows) == 112, "Training inventory must have exactly 112 rows")
        batch_id = "training112_" + input_sha[:12]
    require(all(isinstance(row, dict) and isinstance(row.get("id"), str) and row["id"] for row in rows)
            and len({row["id"] for row in rows}) == len(rows), "Missing or duplicate row identities")
    # Drop every target/spec/scoring field before annotation or request freezing.
    visible_rows = [{"id": row["id"], "history": row.get("history", row.get("messages")),
                     "tools": row["tools"]} for row in rows]
    phase_path = Path(phase_dir)
    require("onedrive" not in str(phase_path.resolve()).lower(), "Private phase artifacts must stay outside OneDrive")
    with PhaseBudget(phase_path, readiness, schema.MODEL, schema.SCHEMA_VERSION,
                     schema.digest(schema.QUESTIONS), schema_sha256) as phase:
        invocation = {"mode": mode, "batch_id": batch_id, "input_file_sha256": input_sha,
                      "schema_source_sha256": schema_sha256, "rows": len(rows),
                      "parallel_real_calls_max": max_workers, "targets_withheld": True,
                      "requests": [{"id": row["id"], "request": schema.payload(row["history"], row["tools"]),
                                    "identity": schema.request_identity(schema.payload(row["history"], row["tools"]))}
                                   for row in visible_rows]}
        freeze_path = phase_path / "invocations" / (sha256(canonical_bytes(invocation)) + ".json")
        if freeze_path.exists():
            require(strict_json(freeze_path.read_bytes()) == invocation, "Invocation freeze differs")
        else:
            write_json(freeze_path, invocation)
        # Deduplicate identical visible payloads inside a batch before threads run.
        by_request, output_requests = {}, []
        for row in visible_rows:
            identity = schema.request_identity(schema.payload(row["history"], row["tools"]))
            key = identity["request_sha256"]
            by_request.setdefault(key, row)
            output_requests.append((row["id"], key))
        service = Service(schema, schema_sha256, phase, readiness, dotenv)
        results = {}
        if recover_request_sha256 is not None:
            require(recover_request_sha256 in by_request, "Explicit failed request is absent from this frozen invocation")
            recovery_row = by_request[recover_request_sha256]
            recovery_request = schema.payload(recovery_row["history"], recovery_row["tools"])
            recovery_identity = schema.request_identity(recovery_request)
            phase.begin_explicit_recovery(recovery_identity, recovery_request)
            # Verify the one replay before spending on unattempted prefixes.
            results[recover_request_sha256] = service.call(recovery_row)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(service.call, row): key for key, row in by_request.items() if key not in results}
            for future in concurrent.futures.as_completed(futures):
                key = futures[future]
                try:
                    results[key] = future.result()
                except Exception:
                    for other in futures:
                        other.cancel()
                    raise RuntimeError("Local service stopped; cached successes and sanitized phase ledger retained") from None
        response = {"batch_id": batch_id,
                    "rows": [{"id": row_id, "annotation": results[key]["annotation"],
                              "cache_hit": results[key]["cache_hit"]} for row_id, key in output_requests]}
        output.parent.mkdir(parents=True, exist_ok=True)
        encoded = gzip.compress(canonical_bytes(response), mtime=0)
        require(not output.exists(), "Output already exists; preserve prior response artifact")
        output.write_bytes(encoded)
        status = phase.status()
        write_json(output.with_name(output.name + ".receipt.json"), {
            "completed": True, "mode": mode, "batch_id": batch_id, "rows": len(rows),
            "unique_requests": len(by_request), "response_gzip_sha256": sha256(encoded),
            "response_bytes": len(encoded), "budget": status,
            "credential_transferred_to_colab": False, "new_model_queries": 0})
        print(json.dumps({"completed": True, "rows": len(rows), "unique_requests": len(by_request),
                          "budget": status}, separators=(",", ":")), flush=True)
        print(MARKER + " " + base64.b64encode(encoded).decode("ascii"), flush=True)
        return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("annotate_training", "answer_batch"))
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase-dir", required=True)
    parser.add_argument("--readiness", required=True)
    parser.add_argument("--schema-sha256", required=True)
    parser.add_argument("--dotenv", default=str(APPROVED_DOTENV))
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--recover-request-sha256", help="Root-only explicit one-time replay of the exact single known failed request")
    args = parser.parse_args()
    try:
        run(args.mode, args.input, args.output, args.phase_dir, args.readiness,
            args.schema_sha256, args.dotenv, args.max_workers, args.recover_request_sha256)
    except Exception as error:
        print(json.dumps({"completed": False, "stage": "local_jev_service_stopped",
                          "error_type": type(error).__name__,
                          "exception_text_headers_error_body_and_secrets_printed": False}), flush=True)
        sys.exit(1)

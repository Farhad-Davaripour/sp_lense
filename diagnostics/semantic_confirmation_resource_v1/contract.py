"""Model-free resource primitives only. No runner, model or tokenizer imports.

Reservations operate on complete serialized byte lengths and are atomic on
rejection. A future binding must serialize access and reconcile actual files;
this module neither writes evidence nor claims that all model metadata will fit.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import struct
from pathlib import Path

HERE = Path(__file__).resolve().parent
MIB = 1024 * 1024


class ResourceLimitError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ResourceLimitError(message)


def natural(value, name):
    require(type(value) is int and value >= 0, name + " must be a nonnegative integer")
    return value


def load_contract():
    value = json.loads((HERE / "contract.json").read_text(encoding="utf-8"))
    validate_contract(value)
    return value


def validate_contract(c):
    require(c["schema"] == "sp_lense.confirmation_resource.v1", "schema")
    require(c["real_run_authorized"] is False, "resource proof cannot authorize execution")
    s, storage = c["study"], c["storage"]
    require(s == {"prompts": 24, "requests": 48, "self_requests": 12,
        "off_requests": 36, "maximum_forwards": 180, "maximum_derivatives": 48,
        "maximum_tokens": 160, "maximum_loads": 1, "fresh_routes": 72,
        "self_endpoints": 12, "maximum_hook_checks": 109,
        "vocabulary": 248320, "hidden_width": 1024}, "exact fixed study scope")
    require(s["maximum_forwards"] == s["prompts"] + s["requests"] + 9 * s["self_requests"], "forward derivation")
    require(s["maximum_derivatives"] == 4 * s["self_requests"], "derivative derivation")
    require(s["maximum_hook_checks"] == 2 * s["requests"] + s["self_requests"] + 1, "hook derivation")
    require(c["seconds"] == {"worker": 1800, "cleanup": 15, "audit": 180}, "one time contract")
    expected = dict(zip(("logits", "rows", "ledgers", "hooks", "sources_inputs", "logs", "audit_inventory", "failure_closeout"),
                        (171, 45, 4, 16, 32, 4, 8, 8)))
    require(storage["categories_bytes"] == {k: v * MIB for k, v in expected.items()}, "exact allocation")
    require(sum(storage["categories_bytes"].values()) == storage["total_bytes"] == 288 * MIB, "aggregate allocation")
    require(storage["per_file_bytes"] == 5 * MIB and storage["preparation_bytes"] == 32 * MIB, "file/preparation limits")
    require(storage["closeout_reserve_bytes"] == storage["categories_bytes"]["failure_closeout"] == 8 * MIB, "separate closeout reserve")
    require(c["logit_codec"] == {"name": "raw_f32_le_final_token_v1", "extension": ".f32", "bytes_per_value": 4,
        "bytes_per_vector": 993280, "compression": "none", "hash": "sha256 of complete raw bytes", "finite_values_required": True}, "fixed lossless logit codec")
    require(c["logit_codec"]["bytes_per_vector"] == s["vocabulary"] * 4, "raw-vector derivation")
    require(s["maximum_forwards"] * 993280 <= storage["categories_bytes"]["logits"], "all raw logits fit")
    require(c["row_maximum_bytes"] == 262144 and s["maximum_forwards"] * c["row_maximum_bytes"] == storage["categories_bytes"]["rows"], "existing row bound fully allocated")
    require(sum(v["maximum_records"] * v["maximum_record_bytes"] for v in c["ledger_streams"].values()) <= storage["categories_bytes"]["ledgers"], "all ledger maxima fit jointly")
    require(sum(c["log_streams"].values()) == storage["categories_bytes"]["logs"], "all logs fit jointly")
    h = c["hook_metadata"]
    require(h["raw_chunk_bytes"] == MIB and h["maximum_snapshot_raw_bytes"] == 64 * MIB and h["fault_reserve_bytes"] == 65536, "inherited hook chunk/raw/fault limits")
    require(h["metadata_capacity_verified"] is False and len(c["pending_verification"]) == 6, "pending conditions explicit")


def component_settings(c):
    """Every future component must consume this complete identical mapping."""
    validate_contract(c)
    return {"seconds": copy.deepcopy(c["seconds"]), "storage": copy.deepcopy(c["storage"]),
            "study": copy.deepcopy(c["study"]), "codec": c["logit_codec"]["name"],
            "row_maximum_bytes": c["row_maximum_bytes"], "hook_metadata": copy.deepcopy(c["hook_metadata"])}


def validate_component_settings(c, components):
    require(set(components) == {"recorder", "worker_supervisor", "audit_supervisor", "saved_judge"}, "all future components declared")
    expected = component_settings(c)
    require(all(settings == expected for settings in components.values()), "inconsistent/inherited limits rejected")


def validate_prompt_lengths(c, lengths):
    require(len(lengths) == c["study"]["prompts"], "all 24 encoded lengths required")
    require(all(type(n) is int and 0 < n <= c["study"]["maximum_tokens"] for n in lengths), "encoded token envelope")


class AttemptCounter:
    """Count before dispatch; failed attempts consume their allowance."""
    def __init__(self, c):
        self.limits = {"forward": c["study"]["maximum_forwards"],
                       "derivative": c["study"]["maximum_derivatives"],
                       "load": c["study"]["maximum_loads"]}
        self.attempts = {kind: 0 for kind in self.limits}
        self.failed = False

    def reserve_attempt(self, kind):
        require(kind in self.limits and not self.failed, "unknown or post-failure dispatch")
        require(self.attempts[kind] < self.limits[kind], "attempt ceiling")
        self.attempts[kind] += 1

    def mark_failed(self):
        self.failed = True


def encode_logits(c, values):
    require(len(values) == c["study"]["vocabulary"], "complete vocabulary required")
    require(all(math.isfinite(v) for v in values), "nonfinite logit input")
    raw = struct.pack("<" + str(len(values)) + "f", *values)
    require(len(raw) == c["logit_codec"]["bytes_per_vector"], "fixed raw length")
    return raw


def decode_logits(c, raw):
    require(len(raw) == c["logit_codec"]["bytes_per_vector"], "short/extra logit bytes rejected")
    values = struct.unpack("<" + str(c["study"]["vocabulary"]) + "f", raw)
    require(all(math.isfinite(v) for v in values), "nonfinite saved logits")
    return values


def canonical_path(path):
    require(isinstance(path, str) and bool(path) and "\x00" not in path and ":" not in path, "relative evidence path")
    parts = path.replace("\\", "/").split("/")
    require(all(part not in ("", ".", "..") for part in parts), "no absolute/traversal/ambiguous path")
    require(all(not part.endswith((" ", ".")) for part in parts), "no Windows-normalized collision")
    return "/".join(parts).casefold()


class ResourceLedger:
    """Single-owner size reservations, not a filesystem writer or admission verdict."""
    def __init__(self, c):
        validate_contract(c)
        self.c = copy.deepcopy(c)
        self.files = {}
        self.used = {category: 0 for category in c["storage"]["categories_bytes"]}
        self.counts = {"logits": 0, "rows": 0, "hook_checks": 0}
        self.events = {name: 0 for name in c["ledger_streams"]}

    def _reserve(self, category, path, nbytes, append=False, hook_fault=False):
        natural(nbytes, "byte length")
        require(category in self.used, "unknown allocation category")
        path = canonical_path(path)
        old_category, previous = self.files.get(path, (category, 0))
        require(old_category == category and (append or path not in self.files), "no overwrite/category transfer")
        require(previous + nbytes <= self.c["storage"]["per_file_bytes"], "per-file cap")
        limit = self.c["storage"]["categories_bytes"][category]
        if category == "hooks" and not hook_fault:
            limit -= self.c["hook_metadata"]["fault_reserve_bytes"]
        require(self.used[category] + nbytes <= limit, "category allocation exceeded")
        require(sum(self.used.values()) + nbytes <= self.c["storage"]["total_bytes"], "total cap")
        # Categories cannot borrow from one another, including failure closeout.
        self.files[path] = (category, previous + nbytes)
        self.used[category] += nbytes

    def add_logit(self, path, nbytes):
        require(path.endswith(".f32") and nbytes == self.c["logit_codec"]["bytes_per_vector"], "fixed raw logit evidence only")
        require(self.counts["logits"] < self.c["study"]["maximum_forwards"], "logit count")
        self._reserve("logits", path, nbytes)
        self.counts["logits"] += 1

    def add_row(self, path, nbytes):
        require(natural(nbytes, "row bytes") <= self.c["row_maximum_bytes"], "existing row cap")
        require(self.counts["rows"] < self.c["study"]["maximum_forwards"], "row count")
        self._reserve("rows", path, nbytes)
        self.counts["rows"] += 1

    def add_event(self, stream, nbytes):
        require(stream in self.events, "undeclared ledger stream")
        profile = self.c["ledger_streams"][stream]
        require(natural(nbytes, "complete record bytes") <= profile["maximum_record_bytes"], "record byte cap includes newline")
        require(self.events[stream] < profile["maximum_records"], "record count")
        self._reserve("ledgers", stream, nbytes, append=True)
        self.events[stream] += 1

    def add_log(self, stream, nbytes):
        require(stream in self.c["log_streams"], "undeclared log stream")
        previous = self.files.get(canonical_path("logs/" + stream), ("logs", 0))[1]
        require(previous + natural(nbytes, "log bytes") <= self.c["log_streams"][stream], "append log cap")
        self._reserve("logs", "logs/" + stream, nbytes, append=True)

    def add_bundle_file(self, category, path, nbytes, append=False):
        require(category in ("sources_inputs", "audit_inventory", "failure_closeout"), "use typed writer for counted evidence")
        self._reserve(category, path, nbytes, append=append)

    def add_hook_check(self, nbytes):
        require(natural(nbytes, "hook check bytes") <= self.c["hook_metadata"]["maximum_check_record_bytes"], "hook event bound")
        require(self.counts["hook_checks"] < self.c["study"]["maximum_hook_checks"], "hook check count")
        self._reserve("hooks", "hook_evidence/checks.jsonl", nbytes, append=True)
        self.counts["hook_checks"] += 1

    def add_hook_json(self, name, nbytes, fault=False):
        require(natural(nbytes, "hook JSON bytes") <= self.c["hook_metadata"]["maximum_setup_json_bytes"], "hook JSON bound")
        if fault:
            require(name == "fault.json" and nbytes <= self.c["hook_metadata"]["fault_reserve_bytes"], "reserved hook fault")
        self._reserve("hooks", "hook_evidence/" + name, nbytes, hook_fault=fault)

    def add_hook_snapshot(self, name, raw_sizes, encoded_sizes, manifest_bytes):
        h = self.c["hook_metadata"]
        require(len(raw_sizes) == len(encoded_sizes) and bool(raw_sizes), "complete matched chunk lists")
        require(all(type(n) is int and 0 < n <= h["raw_chunk_bytes"] for n in raw_sizes), "raw hook chunk cap")
        require(all(n == h["raw_chunk_bytes"] for n in raw_sizes[:-1]), "no missing/short interior raw chunk")
        require(sum(raw_sizes) <= h["maximum_snapshot_raw_bytes"], "inherited full snapshot raw cap")
        require(natural(manifest_bytes, "manifest bytes") <= h["maximum_manifest_bytes"], "manifest bound")
        proposed = copy.deepcopy(self)
        for i, encoded in enumerate(encoded_sizes):
            proposed._reserve("hooks", f"hook_evidence/{name}_{i:03d}.zlib", natural(encoded, "complete encoded chunk bytes"))
        proposed._reserve("hooks", f"hook_evidence/{name}.json", manifest_bytes)
        self.files, self.used = proposed.files, proposed.used

    def report(self):
        return {"used_bytes": dict(self.used), "total_bytes": sum(self.used.values()),
                "counts": dict(self.counts), "ledger_records": dict(self.events),
                "remaining_closeout_bytes": self.c["storage"]["closeout_reserve_bytes"] - self.used["failure_closeout"]}


def source_record_bounds(c):
    return {"raw_logit_vector_bytes": c["study"]["vocabulary"] * 4,
        "raw_logits_180_bytes": c["study"]["maximum_forwards"] * c["logit_codec"]["bytes_per_vector"],
        "rows_180_bytes": c["study"]["maximum_forwards"] * c["row_maximum_bytes"],
        "all_ledger_maxima_bytes": sum(p["maximum_records"] * p["maximum_record_bytes"] for p in c["ledger_streams"].values()),
        "all_hook_check_records_bytes": c["study"]["maximum_hook_checks"] * c["hook_metadata"]["maximum_check_record_bytes"],
        "all_log_streams_bytes": sum(c["log_streams"].values()),
        "reserved_total_bytes": sum(c["storage"]["categories_bytes"].values()),
        "non_closeout_bytes": c["storage"]["total_bytes"] - c["storage"]["closeout_reserve_bytes"]}

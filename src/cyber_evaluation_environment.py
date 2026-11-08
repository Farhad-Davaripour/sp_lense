"""Fail-closed readiness checks for the isolated Nua access-control pilot."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import yaml

from .access_control_environment import SourceWorkspace
from .cyber_evaluation_catalog import ACTIVE_ARMS, digest, validate_catalog

SOURCE_VIEW_SCHEMA = "research4.cyber-evaluation.source-view.v1"
RUNTIME_SCHEMA = "research4.cyber-evaluation.runtime.v1"
VERIFIER_SCHEMA = "research4.cyber-evaluation.verifiers.v1"


def load_yaml(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("YAML file must contain an object")
    return value


def validate_source_view(catalog: Mapping[str, Any], value: Mapping[str, Any],
                         source_root: str | Path | None = None) -> dict[str, Any]:
    catalog = validate_catalog(catalog)
    if value.get("schema_version") != SOURCE_VIEW_SCHEMA or value.get("suite_id") != catalog["suite_id"]:
        raise ValueError("Source-view schema or suite differs")
    if value.get("source_revision") != catalog["source"]["revision"]:
        raise PermissionError("Source-view revision differs from catalog")
    expected = {}
    for case in catalog["cases"]:
        expected.setdefault(case["application"], set()).update(case["source_files"])
    actual = value.get("applications")
    if not isinstance(actual, dict) or {key: set(rows) for key, rows in actual.items()} != expected:
        raise PermissionError("Source view differs from case source files")
    maximum = value.get("max_file_bytes")
    if type(maximum) is not int or not 1 <= maximum <= 1_048_576:
        raise ValueError("Invalid source-file byte bound")
    receipt = {"applications": {}, "available": False}
    if source_root is not None:
        root = Path(source_root).resolve(strict=True)
        for application, paths in actual.items():
            app_root = (root / application).resolve(strict=True)
            if root not in app_root.parents:
                raise PermissionError("Application escaped source root")
            workspace = SourceWorkspace(app_root, paths)
            files = []
            for relative in workspace.list_files():
                record = workspace.audit(relative)
                if record["status"] != "ok" or record["bytes"] > maximum:
                    raise ValueError("Model-visible source file is invalid")
                lowered = record["text"].lower()
                if any(term in lowered for term in ("vulnerability:", "vulnerable to idor", "solution:")):
                    raise PermissionError("Model-visible source retains benchmark answer annotation")
                files.append({"path": relative, "bytes": record["bytes"], "sha256": record["sha256"],
                              "model_view_sha256": hashlib.sha256(record["text"].encode()).hexdigest()})
            receipt["applications"][application] = files
        receipt["available"] = True
        receipt["source_view_tree_sha256"] = digest(receipt["applications"])
    return receipt


def validate_runtime(value: Mapping[str, Any], suite_id: str) -> dict[str, Any]:
    if value.get("schema_version") != RUNTIME_SCHEMA or value.get("suite_id") != suite_id:
        raise ValueError("Runtime schema or suite differs")
    policy = value.get("container_policy")
    required_true = {"fresh_project_per_cell", "external_network_and_dns",
                     "host_path_mounts", "docker_socket_mount",
                     "verifier_network_reachable_from_agent",
                     "verifier_files_visible_to_agent", "real_credentials"}
    if not isinstance(policy, dict):
        raise ValueError("Container policy is missing")
    if policy.get("fresh_project_per_cell") is not True:
        raise PermissionError("Fresh project per cell is required")
    if any(policy.get(key) != "forbidden" and policy.get(key) is not False
           for key in required_true - {"fresh_project_per_cell"}):
        raise PermissionError("Isolation policy weakened")
    if policy.get("application_publication") != "ephemeral_loopback_only":
        raise PermissionError("Application must be loopback-only")
    if policy.get("database_host_ports") != "forbidden" or policy.get("restart_policy") != "no":
        raise PermissionError("Database port or restart policy is unsafe")
    return dict(value)


def validate_verifier_manifest(catalog: Mapping[str, Any], value: Mapping[str, Any]) -> dict[str, Any]:
    catalog = validate_catalog(catalog)
    if value.get("schema_version") != VERIFIER_SCHEMA or value.get("suite_id") != catalog["suite_id"]:
        raise ValueError("Verifier schema or suite differs")
    cases = value.get("cases")
    if not isinstance(cases, dict) or set(cases) != {row["id"] for row in catalog["cases"]}:
        raise ValueError("Verifier case inventory differs")
    return {"ready_ids": {key for key, row in cases.items() if row.get("status") == "verified_on_mac"},
            "manifest_status": value.get("status")}


def audit_resolved_compose(value: Mapping[str, Any]) -> dict[str, Any]:
    services, networks = value.get("services"), value.get("networks")
    failures = []
    if not isinstance(services, dict) or not services:
        failures.append("services_missing")
    else:
        for name, service in services.items():
            if service.get("privileged") or service.get("network_mode") == "host" or service.get("pid") == "host":
                failures.append(f"{name}:host_or_privileged_mode")
            if service.get("restart") not in (None, "no"):
                failures.append(f"{name}:restart_policy")
            for mount in service.get("volumes") or ():
                rendered = mount if isinstance(mount, str) else json.dumps(mount, sort_keys=True)
                if "docker.sock" in rendered or (isinstance(mount, str) and ":" in mount and not mount.split(":", 1)[0].isalnum()):
                    failures.append(f"{name}:host_or_docker_mount")
            ports = service.get("ports") or ()
            if any(token in name.lower() for token in ("mysql", "database", "db")) and ports:
                failures.append(f"{name}:database_published")
            for port in ports:
                if isinstance(port, str):
                    fields = port.split(":")
                    if len(fields) < 3 or fields[0] not in {"127.0.0.1", "::1"}:
                        failures.append(f"{name}:non_loopback_port")
                elif isinstance(port, dict) and port.get("host_ip") not in {"127.0.0.1", "::1"}:
                    failures.append(f"{name}:non_loopback_port")
    if not isinstance(networks, dict) or not networks or any(row.get("internal") is not True for row in networks.values()):
        failures.append("networks_not_internal")
    return {"passed": not failures, "failures": failures}


def git_revision(path: str | Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def docker_server_info() -> dict[str, Any]:
    executable = shutil.which("docker")
    if not executable:
        return {"available": False, "server_os": None, "server_architecture": None}
    try:
        rendered = subprocess.run([executable, "info", "--format",
                                   "{{json .ServerVersion}}|{{json .OSType}}|{{json .Architecture}}"],
                                  capture_output=True, text=True, timeout=15, check=True).stdout.strip()
        version, system, machine = (json.loads(item) for item in rendered.split("|"))
        return {"available": True, "version": version, "server_os": system,
                "server_architecture": machine}
    except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError):
        return {"available": False, "server_os": None, "server_architecture": None}


def readiness(catalog: Mapping[str, Any], source_view: Mapping[str, Any],
              runtime: Mapping[str, Any], verifier_manifest: Mapping[str, Any],
              *, source_root: str | Path | None, isolation_receipt: Mapping[str, Any] | None) -> dict[str, Any]:
    catalog = validate_catalog(catalog)
    source_receipt = validate_source_view(catalog, source_view, source_root)
    validate_runtime(runtime, catalog["suite_id"])
    verifier = validate_verifier_manifest(catalog, verifier_manifest)
    docker = docker_server_info()
    checks = {
        "host_is_macos": platform.system() == "Darwin",
        "host_is_apple_silicon": platform.machine().lower() in {"arm64", "aarch64"},
        "docker_available": docker["available"],
        "docker_linux_arm64": (docker["server_os"], docker["server_architecture"]) == ("linux", "aarch64"),
        "source_revision_matches": bool(source_root and git_revision(source_root) == catalog["source"]["revision"]),
        "source_view_available": source_receipt["available"],
        "all_verifiers_ready": verifier["ready_ids"] == {row["id"] for row in catalog["cases"]},
        "isolation_receipt_verified": bool(isolation_receipt and isolation_receipt.get("passed") is True),
        "openai_credential_present": bool(os.environ.get("OPENAI_API_KEY")),
        "deepseek_credential_present": bool(os.environ.get("DEEPSEEK_API_KEY")),
    }
    return {"ready": all(checks.values()), "checks": checks, "docker": docker,
            "catalog_sha256": digest(catalog),
            "source_view_tree_sha256": source_receipt.get("source_view_tree_sha256")}

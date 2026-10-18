"""Launch frozen workers through Databricks AI Runtime instead of GPU notebooks."""
from pathlib import Path, PurePosixPath
import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import shlex
from src.config import Settings
from src.utils import write_json


def launcher(settings, admission, phase):
    if phase not in ("gate", "steering", "heldout", "frozen_A"):
        raise ValueError("Unknown matched research phase")
    values = {"DATABRICKS_CATALOG": settings.catalog, "DATABRICKS_SCHEMA": settings.schema,
              "DATABRICKS_VOLUME": settings.volume, "DATABRICKS_CODE_PATH": settings.code_path,
              "RESEARCH3_OUTPUT_ROOT": settings.volume_root + "/runs",
              "RESEARCH3_ADMISSION": admission}
    lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    lines += ["export " + key + "=" + shlex.quote(value) for key, value in values.items()]
    lines += ['code_root="$CODE_SOURCE_PATH"',
              'if [ ! -d "$code_root/src" ]; then code_root="$(dirname "$code_root")"; fi',
              'cd "$code_root"', "exec python -m src.runtime " + phase, ""]
    return "\n".join(lines).encode()


def runtime_task(client, job_id):
    if list(client.jobs.list_runs(job_id=job_id, active_only=True, limit=1)):
        raise ValueError("Do not modify an active research job")
    job = client.jobs.get(job_id)
    if job.settings.max_concurrent_runs != 1:
        raise ValueError("This bounded launcher requires a single concurrent run")
    tasks = [t.as_dict() for t in job.settings.tasks if t.ai_runtime_task is not None]
    if len(tasks) != 1:
        raise ValueError("Deploy the AI Runtime job before submitting")
    return copy.deepcopy(tasks[0])


def bind_launcher(client, job_id, task, path, source):
    import base64
    from databricks.sdk.service.workspace import ImportFormat
    from databricks.sdk.service.jobs import JobSettings
    if list(client.jobs.list_runs(job_id=job_id, active_only=True, limit=1)):
        raise ValueError("Do not modify an active research job")
    client.workspace.mkdirs(str(PurePosixPath(path).parent))
    client.workspace.import_(path=path, format=ImportFormat.AUTO,
                             content=base64.b64encode(source).decode(), overwrite=False)
    if client.workspace.download(path).read() != source:
        raise ValueError("Immutable runtime launcher readback differs")
    task = copy.deepcopy(task)
    deployments = task["ai_runtime_task"]["deployments"]
    if (len(deployments) != 1 or deployments[0]["compute"]["accelerator_count"] != 1 or deployments[0]["compute"]["accelerator_type"] != "GPU_1xA10"):
        raise ValueError("Exactly one admitted GPU is required")
    deployments[0]["command_path"] = path
    client.jobs.update(job_id=job_id, new_settings=JobSettings.from_dict({"tasks": [task]}))
    saved = next(t.as_dict() for t in client.jobs.get(job_id).settings.tasks if t.task_key == task["task_key"])
    if saved["ai_runtime_task"] != task["ai_runtime_task"]:
        raise ValueError("The saved AI Runtime launch differs")
    return {"path": path, "bytes": len(source), "sha256": hashlib.sha256(source).hexdigest()}


def run(phase):
    from src.verify import source_digest
    settings = Settings.load()
    path = Path(os.environ["RESEARCH3_ADMISSION"])
    if not path.is_relative_to(Path(settings.volume_root)):
        raise ValueError("The admission must be in the configured private Volume")
    admission = json.loads(path.read_bytes())
    if admission["phase"] != phase or admission.get("execution_transport") != "AI_RUNTIME_TASK":
        raise ValueError("The runtime phase or transport differs from the admission")
    if dt.datetime.now(dt.timezone.utc) >= dt.datetime.fromisoformat(admission["absolute_end_utc"]):
        raise TimeoutError("Original admission expired before worker entry")
    if source_digest() != admission["source_digest"]:
        raise ValueError("Packaged research source differs from the admission")
    output = Path(admission["output"] + "_RUNTIME_ENTRY.json")
    if output.exists():
        raise ValueError("This scientific launch has already entered its worker")
    write_json(output, {"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "phase": phase,
                       "source_digest": source_digest(), "transport": "AI_RUNTIME_TASK",
                       "runtime_launcher": admission["runtime_launcher"], "model_loads_at_entry": 0})
    if phase == "frozen_A":
        from src.frozen_a_verification import run as frozen_run
        return frozen_run(str(path))
    if phase == "gate":
        from src.matched_validation import run_gate
        return run_gate(str(path))
    if phase == "heldout":
        from src.gate_confirmation import run as confirmation_run
        return confirmation_run(str(path))
    from src.matched_steering import run as matched_run
    return matched_run(str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["gate", "steering", "heldout", "frozen_A"])
    print(json.dumps(run(parser.parse_args().phase)))


if __name__ == "__main__":
    main()

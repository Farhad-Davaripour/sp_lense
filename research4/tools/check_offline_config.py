"""Check local config contracts; strict Databricks validation remains separate."""

import json
from pathlib import Path
import re

import yaml

ROOT = Path(__file__).resolve().parents[1]
HOST = "https://dbc-ff9672a4-ba5f.cloud.databricks.com"
GPU_JOB_RESOURCES = {
    "simulated_continuity_rank4_qualification",
    "simulated_continuity_primary_partition",
}
GPU_ENVIRONMENT_KEY = "continuity_gpu"
GPU_ACCELERATOR = "${var.continuity_gpu_accelerator}"
GPU_ENVIRONMENT_VERSION = "${var.continuity_gpu_environment_version}"
GPU_INSTALL_LINE = (
    "# MAGIC %uv pip install --extra-index-url "
    "https://download.pytorch.org/whl/cu128 torch==2.11.0+cu128 "
    "transformers==5.19.0 bitsandbytes==0.49.2 accelerate==1.15.0 "
    "safetensors==0.8.0 fla-core==0.5.2 einops==0.8.2 numpy==2.2.2 "
    "psutil==7.2.2 triton==3.6.0 nvidia-ml-py==13.615.71"
)


def unique_mapping(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate configuration key: {key}")
        result[key] = value
    return result


class UniqueLoader(yaml.SafeLoader):
    pass


def yaml_mapping(loader, node):
    return unique_mapping(loader.construct_pairs(node, deep=True))


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, yaml_mapping)


def read_yaml(path):
    value = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueLoader)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Expected a nonempty configuration mapping: {path}")
    return value


def positive_integer(value, description):
    if type(value) is not int or value <= 0:
        raise ValueError(f"Expected a positive integer for {description}")


def check_job(job, root, resource_name=None):
    if any(key in job for key in ("schedule", "trigger", "continuous")):
        raise ValueError("Research 4 jobs must remain unscheduled")
    if job.get("max_concurrent_runs") != 1 or job.get("queue", {}).get("enabled") is not False:
        raise ValueError("Research 4 jobs must run singly with queues disabled")
    positive_integer(job.get("timeout_seconds"), "job timeout")
    environments = job.get("environments", [])
    environment_keys = [item["environment_key"] for item in environments]
    if len(environment_keys) != len(set(environment_keys)):
        raise ValueError("Duplicate environment key")
    for environment in environments:
        if not isinstance(environment["spec"].get("environment_version"), str):
            raise ValueError("Environment version must be a quoted string")
    gpu_job = resource_name in GPU_JOB_RESOURCES
    if job.get("job_clusters"):
        raise ValueError("Research 4 bundle jobs must remain serverless")
    if gpu_job:
        gpu_environments = [item for item in environments
                            if item["environment_key"] == GPU_ENVIRONMENT_KEY]
        if len(gpu_environments) != 1:
            raise ValueError("Databricks continuity GPU environment is missing")
        spec = gpu_environments[0]["spec"]
        if (spec.get("environment_version") != GPU_ENVIRONMENT_VERSION
                or spec.get("dependencies") != []
                or "base_environment" in spec):
            raise ValueError("Databricks continuity GPU environment changed")
    tasks = job.get("tasks", [])
    keys = [task["task_key"] for task in tasks]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("Task keys must be nonempty and unique")
    dependencies = {}
    gpu_task_count = 0
    for task in tasks:
        if any(key in task for key in ("new_cluster", "existing_cluster_id",
                                       "job_cluster_key")):
            raise ValueError("Research 4 bundle jobs must remain serverless")
        compute = task.get("compute")
        if compute is not None:
            if (not gpu_job or task.get("task_key") != "run_one_partition"
                    or compute != {"hardware_accelerator": GPU_ACCELERATOR}
                    or task.get("environment_key") != GPU_ENVIRONMENT_KEY
                    or "libraries" in task):
                raise ValueError("Task references an unreviewed serverless GPU")
            gpu_task_count += 1
        elif task.get("environment_key") not in environment_keys:
            raise ValueError("Task references an undefined environment")
        if task.get("max_retries") != 0:
            raise ValueError("Research 4 tasks must disable automatic retries")
        positive_integer(task.get("timeout_seconds"), "task timeout")
        if task["timeout_seconds"] > job["timeout_seconds"]:
            raise ValueError("Task timeout exceeds the job boundary")
        path = (root / "resources" / task["notebook_task"]["notebook_path"]).resolve()
        if not path.is_relative_to((root / "notebooks").resolve()) or not path.is_file():
            raise ValueError("Task notebook must exist inside research4/notebooks")
        dependencies[task["task_key"]] = {item["task_key"] for item in task.get("depends_on", [])}
        if not dependencies[task["task_key"]].issubset(keys):
            raise ValueError("Task dependency is undefined")
    if gpu_job and gpu_task_count != 1:
        raise ValueError("Databricks continuity job requires one serverless GPU task")
    remaining = dict(dependencies)
    while remaining:
        ready = {key for key, needs in remaining.items() if not needs.intersection(remaining)}
        if not ready:
            raise ValueError("Task dependencies contain a cycle")
        remaining = {key: needs for key, needs in remaining.items() if key not in ready}


def check_gpu_notebook_contract(root):
    notebook = (root / "notebooks" / "databricks_continuity_partition.py").read_text(
        encoding="utf-8")
    lines = notebook.splitlines()
    if len(lines) < 3 or lines[1] != GPU_INSTALL_LINE:
        raise ValueError("Serverless GPU notebook pin installation changed")
    if notebook.count(GPU_INSTALL_LINE) != 1:
        raise ValueError("Serverless GPU notebook must install exact pins once")
    readiness = notebook.find("dependency_proof = assert_fast_deltanet_dependencies()")
    source = notebook.find("source_proof = verify_deployed_source(")
    bitsandbytes = notebook.find("bitsandbytes.functional.quantize_4bit(")
    disk = notebook.find("shutil.disk_usage(")
    reserve = notebook.find("reserve_intent(lock_root, intent, owner)")
    run = notebook.find("result = run_partition(")
    if (min(readiness, source, bitsandbytes, disk, reserve, run) < 0
            or not (source < readiness < bitsandbytes < disk < reserve < run)):
        raise ValueError("GPU readiness/reservation/model order changed")
    preflight = (root / "notebooks" / "databricks_continuity_preflight.py").read_text(
        encoding="utf-8")
    if "reserve_intent(" in preflight or "consumed_" in preflight:
        raise ValueError("CPU preflight must remain read-only")


def reject_constant(value):
    raise ValueError(f"Nonfinite JSON: {value}")


def storage_namespace(bundle):
    """Accept the frozen legacy volume or the selected external UC volume.

    The current Databricks port deliberately does not declare the already-created
    Unity Catalog volume as a bundle-owned resource.  Doing so would make deploy
    attempt to create or take ownership of an existing external dependency.
    """
    legacy = bundle.get("resources", {}).get("volumes", {}).get("artifacts")
    if legacy is not None:
        value = (legacy.get("catalog_name"), legacy.get("schema_name"),
                 legacy.get("name"))
        if value != ("workspace", "research4", "artifacts"):
            raise ValueError("Research 4 legacy storage namespace changed")
        return value
    variables = bundle.get("variables", {})
    value = tuple(variables.get(name, {}).get("default") for name in (
        "continuity_catalog", "continuity_bronze_schema",
        "continuity_model_volume"))
    if value != ("research4_simulated_continuity", "bronze", "model_data"):
        raise ValueError("Research 4 external storage namespace changed")
    return value


def main():
    bundle = read_yaml(ROOT / "databricks.yml")
    accounts=json.loads((ROOT/'config/workspace_accounts.json').read_text())
    if 'host' in bundle['workspace']:
        raise ValueError('Bundle host must come from the explicit account selector environment')
    azure = accounts['accounts']['azure']
    if (set(accounts['accounts'])!={'azure','work','free_tier'}
            or accounts['accounts']['free_tier']['host']!=HOST
            or azure['host']!='https://adb-7405615362381203.3.azuredatabricks.net'
            or azure['organization']!='7405615362381203'
            or azure['workspace_name']!='db_sp_lense_us'
            or azure['artifact_volume']!='/Volumes/db_sp_lense/research4/artifacts'):
        raise ValueError('Authorized account profiles changed')
    storage_namespace(bundle)
    check_gpu_notebook_contract(ROOT)
    variables = bundle.get("variables", {})
    if (variables.get("continuity_gpu_accelerator", {}).get("default")
            != "GPU_1xA10"
            or variables.get("continuity_gpu_environment_version", {}).get("default")
            != "6"):
        raise ValueError("Databricks continuity GPU profile changed")
    if variables.get("continuity_dataset_release", {}).get("default") != (
            "/Volumes/research4_simulated_continuity/bronze/model_data/"
            "dataset_releases/"
            "c21d8b334fdd1e7544fbea8277643f594c990dd5a754312d5e5a9e0d947df1df"):
        raise ValueError("Databricks continuity dataset release changed")
    included = {path for pattern in bundle["include"] for path in ROOT.glob(pattern)}
    resources = set((ROOT / "resources").glob("*.job.yml"))
    if not resources or included != resources:
        raise ValueError("Bundle includes must cover exactly the Research 4 job files")
    names = set()
    for path in sorted(resources):
        for name, job in read_yaml(path)["resources"]["jobs"].items():
            if name in names:
                raise ValueError(f"Duplicate job resource: {name}")
            names.add(name)
            check_job(job, ROOT, name)
    configs = list((ROOT / "config").glob("*.yaml")) + list((ROOT / "config").glob("*.json"))
    for path in sorted(configs):
        if path.suffix == ".yaml":
            read_yaml(path)
        else:
            json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_mapping,
                       parse_constant=reject_constant)
    study = read_yaml(ROOT / "config/study.yaml")
    if study["storage"]["schema"] != "research4" or study["budget"]["maximum_retries"] != 0:
        raise ValueError("Study storage/retry boundary changed")
    if not re.fullmatch(r"[0-9a-f]{40}", study["model"]["revision"]):
        raise ValueError("The study model requires an exact revision")
    print(f"PASS: {len(names)} unscheduled jobs and {len(configs)} local configs; no workspace access")


if __name__ == "__main__":
    main()

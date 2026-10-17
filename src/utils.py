"""Small shared helpers; secrets are only read when a client is requested."""
from pathlib import Path
import hashlib
import json
import os

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)

def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)

def auth_environment(settings):
    env = os.environ.copy()
    if settings.host:
        env["DATABRICKS_HOST"] = settings.host
    token = env.get("DATABRICKS_TOKEN") or env.get(settings.token_variable)
    if not token and os.name == "nt":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                token = winreg.QueryValueEx(key, settings.token_variable)[0]
        except FileNotFoundError:
            pass
    if token:
        env["DATABRICKS_TOKEN"] = token
        env["DATABRICKS_AUTH_TYPE"] = "pat"
    return env

def workspace_client(settings):
    from databricks.sdk import WorkspaceClient
    env = auth_environment(settings)
    if env.get("DATABRICKS_TOKEN"):
        return WorkspaceClient(host=settings.host, token=env["DATABRICKS_TOKEN"], auth_type="pat")
    return WorkspaceClient(host=settings.host)

def bundle(action, extra=()):
    import shutil
    import subprocess
    from src.config import Settings, ROOT, read_env
    settings = Settings.load()
    env = auth_environment(settings)
    for name, value in {"catalog": settings.catalog, "schema": settings.schema,
                        "volume": settings.volume, "warehouse_id": settings.warehouse_id,
                        "code_path": settings.code_path,
                        "accelerator": settings.accelerator}.items():
        env["BUNDLE_VAR_" + name] = value
    cli = (env.get("DATABRICKS_CLI_PATH") or read_env(ROOT / ".env").get("DATABRICKS_CLI_PATH")
           or shutil.which("databricks"))
    if not cli:
        raise RuntimeError("Install the official Databricks CLI on PATH or set DATABRICKS_CLI_PATH")
    return subprocess.run([cli, "bundle", action, *extra], cwd=ROOT, env=env, check=True).returncode

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["bundle"])
    parser.add_argument("action", choices=["validate", "deploy", "run", "summary"])
    args, extra = parser.parse_known_args()
    raise SystemExit(bundle(args.action, extra))


def run_recipe(name, engine_args=(), execute=False, max_seconds=7200, output=None):
    """Stage a verified source capsule and run its original CLI in a fresh process."""
    import shutil
    import subprocess
    import sys
    import time
    import yaml
    from src.config import ROOT, Settings
    recipe = yaml.safe_load((ROOT / "config/research.yaml").read_text(encoding="utf-8"))
    if name not in recipe["entrypoints"]:
        raise ValueError("Unknown reproduction recipe")
    for item in recipe["sources"] + recipe["configuration_files"]:
        if sha256(ROOT / item["source"]) != item["sha256"]:
            raise ValueError("A pinned scientific source/configuration file changed")
    if not 0 < max_seconds <= 21600:
        raise ValueError("A positive bounded execution window is required")
    plan = {"recipe": name, "entrypoint": recipe["entrypoints"][name],
            "engine_arguments": list(engine_args), "max_seconds": max_seconds,
            "execute": execute, "source_files_verified": len(recipe["sources"])}
    if not execute:
        return plan
    if not engine_args:
        raise ValueError("Supply the original worker arguments and frozen runtime inputs")
    settings = Settings.load()
    stage = Path(output) if output else settings.output_root / (name + "_" + str(time.time_ns()))
    stage.mkdir(parents=True, exist_ok=False)
    source_root = stage / "source"
    for item in recipe["sources"] + recipe["configuration_files"]:
        target = source_root / item["runtime_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / item["source"], target)
    write_json(stage / "SOURCE_PLAN.json", plan)
    script = source_root / recipe["entrypoints"][name]
    env = os.environ.copy()
    for key in ("DATABRICKS_TOKEN", "databricks-token", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "HF_TOKEN"):
        env.pop(key, None)
    paths = [str(script.parent)] + sorted({str(p.parent) for p in source_root.rglob("*.py")})
    env["PYTHONPATH"] = os.pathsep.join(paths)
    try:
        completed = subprocess.run([sys.executable, str(script), *engine_args],
                                   cwd=source_root, env=env, timeout=max_seconds)
    except (subprocess.TimeoutExpired, OSError) as error:
        write_json(stage / "PROCESS_EXIT.json", {"status": type(error).__name__,
                   "recipe": name, "scientific_success_not_inferred": True,
                   "automatic_retry": False})
        raise
    write_json(stage / "PROCESS_EXIT.json", {"return_code": completed.returncode,
                                           "recipe": name, "scientific_success_not_inferred": True})
    completed.check_returncode()
    return dict(plan, output=str(stage), return_code=completed.returncode)

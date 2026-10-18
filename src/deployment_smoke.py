"""Read-only verification of code deployed to the shared Databricks workspace."""
import re
from pathlib import Path
from src.config import ROOT, Settings, TABLES
from src.verify import verify


def run(spark, expected_commit, expected_source_digest):
    if not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
        raise ValueError("A real source commit is required")
    settings = Settings.load()
    if str(ROOT).replace("\\", "/") != settings.code_path.rstrip("/"):
        raise ValueError("The job is not executing the configured shared code")
    if not settings.code_path.startswith("/Workspace/Shared/"):
        raise ValueError("Use the shared workspace deployment")
    result = verify()
    if result["source_digest"] != expected_source_digest:
        raise ValueError("Deployed source differs from the CI checkout")
    tables = {name: spark.catalog.tableExists(settings.table(name)) for name in TABLES}
    if not all(tables.values()) or not Path(settings.volume_root).is_dir():
        raise ValueError("The existing Bronze tables or Volume are unavailable")
    return {"status": "SHARED_DEPLOYMENT_VERIFIED", "commit": expected_commit,
            "source_digest": result["source_digest"], "code_path": str(ROOT),
            "tables": tables, "volume_accessible": True, "data_rows_read": 0,
            "model_loads": 0, "optimizer_updates": 0, "new_reference_calls": 0}

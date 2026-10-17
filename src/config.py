"""Environment and dataset configuration. Importing this module does not authenticate."""
from dataclasses import dataclass
from pathlib import Path
import os
import re

ROOT = Path(__file__).resolve().parents[1]
TABLES = ("training", "validation", "heldout")

def read_env(path):
    values = {}
    if not Path(path).exists():
        return values
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:]
        if "=" not in line:
            raise ValueError(f"Invalid environment setting at line {number}")
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("\"", "'"):
            value = value[1:-1]
        values[key] = value
    return values

def identifier(value):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", value or ""):
        raise ValueError("Invalid catalog, schema, volume or table identifier")
    return chr(96) + value + chr(96)

@dataclass(frozen=True)
class Settings:
    catalog: str
    schema: str
    volume: str
    host: str
    warehouse_id: str
    code_path: str
    accelerator: str
    token_variable: str
    data_root: Path
    output_root: Path

    @classmethod
    def load(cls, env_file=None):
        values = read_env(ROOT / ".env.example")
        values.update(read_env(Path(env_file) if env_file else ROOT / ".env"))
        values.update(os.environ)
        def required(key):
            value = values.get(key, "").strip()
            if not value:
                raise ValueError(f"Set {key} in .env or the job environment")
            identifier(value)
            return value
        return cls(
            required("DATABRICKS_CATALOG"), required("DATABRICKS_SCHEMA"),
            required("DATABRICKS_VOLUME"), values.get("DATABRICKS_HOST", ""),
            values.get("DATABRICKS_WAREHOUSE_ID", ""),
            values.get("DATABRICKS_CODE_PATH", "/Workspace/Shared/sp_lense"),
            values.get("DATABRICKS_ACCELERATOR", "GPU_1xA10"),
            values.get("DATABRICKS_TOKEN_VARIABLE", "databricks-token"),
            Path(values.get("RESEARCH3_DATA_ROOT") or Path.home() / "Research3" / "datasets" / "research3_bronze_v1"),
            Path(values.get("RESEARCH3_OUTPUT_ROOT") or Path.home() / "Research3" / "artifacts" / "reproduction"))

    def table(self, name):
        if name not in TABLES:
            raise ValueError("Only training, validation and heldout are configured")
        return ".".join(identifier(x) for x in (self.catalog, self.schema, name))

    @property
    def volume_root(self):
        return f"/Volumes/{self.catalog}/{self.schema}/{self.volume}"

def load_datasets(path=None):
    import yaml
    with open(path or ROOT / "config" / "data.yaml", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if value.get("format") != "research3-datasets-v1":
        raise ValueError("Unsupported dataset configuration")
    if {item["table"] for item in value["datasets"]} != set(TABLES):
        raise ValueError("Dataset manifest must cover exactly the three configured tables")
    return value

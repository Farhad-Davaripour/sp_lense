"""Persist the three frozen datasets into configured Bronze Delta tables."""
from pathlib import Path
import argparse
import collections
import datetime as dt
import hashlib
import io
import json
import time
from src.config import ROOT, Settings, identifier, load_datasets
from src.utils import canonical, sha256, workspace_client, write_json

COLUMNS = ("dataset_id", "dataset_version", "cohort_code", "source_record_index",
           "record_id", "source_file_sha256", "record_sha256", "record_json",
           "usage_roles_json", "source_volume_path")
READ_SCHEMA = ", ".join(name + (" BIGINT" if name == "source_record_index" else " STRING") for name in COLUMNS)
ALLOWED = {"T1-M", "T1-W", "V1-W", "H24"}

def literal(value):
    return "'" + str(value).replace("'", "''") + "'"

def fetch_sources(settings, manifest=None, client=None):
    """Fetch only the four configured, unsealed sources from the private Volume."""
    config = manifest or load_datasets()
    root = settings.data_root.resolve()
    if root.is_relative_to(ROOT.resolve()):
        raise ValueError('Keep private dataset sources outside the checkout')
    client = client or workspace_client(settings)
    for item in config['datasets']:
        if item['id'] not in ALLOWED or item.get('sealed'):
            raise ValueError('Only configured unsealed cohorts may be downloaded')
        target = (root / item['source']).resolve()
        if not target.is_relative_to(root):
            raise ValueError('Dataset source escapes its configured root')
        if target.exists():
            if sha256(target) != item['sha256']:
                raise ValueError('Existing local dataset differs; it will not be overwritten')
            continue
        source = f"{settings.volume_root}/datasets/{item['id']}/{item['version']}/{item['sha256']}/source.json"
        with client.files.download(source).contents as stream:
            raw = stream.read()
        if hashlib.sha256(raw).hexdigest() != item['sha256']:
            raise ValueError('Private Volume source hash mismatch')
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(raw)
        if sha256(target) != item['sha256']:
            raise ValueError('Downloaded dataset readback mismatch')
    return prepare(settings, config)


def prepare(settings, manifest=None):
    config = manifest or load_datasets()
    prepared = []
    for item in config["datasets"]:
        if item["id"] not in ALLOWED or item.get("sealed"):
            raise ValueError("Only the four configured unsealed cohorts may be persisted")
        source = (settings.data_root / item["source"]).resolve()
        if not source.is_relative_to(settings.data_root.resolve()):
            raise ValueError("Dataset source escapes the configured data directory")
        if source.is_file():
            raw = source.read_bytes()
        elif Path('/Volumes').exists():
            # Databricks workers reuse the immutable Volume source already persisted.
            volume_source = (f"{settings.volume_root}/datasets/{item['id']}/"
                             f"{item['version']}/{item['sha256']}/source.json")
            raw = Path(volume_source).read_bytes()
        else:
            raise FileNotFoundError(
                f"Missing private source for {item['id']}. Set RESEARCH3_DATA_ROOT to the authorized "
                f"dataset directory containing {item['source']}; the Git checkout contains definitions, not private records.")
        source_sha = hashlib.sha256(raw).hexdigest()
        if source_sha != item["sha256"]:
            raise ValueError(f"Source hash mismatch: {item['id']}")
        records = json.loads(raw)
        if len(records) != item["expected_records"]:
            raise ValueError(f"Record count mismatch: {item['id']}")
        if len({r["id"] for r in records}) != len(records):
            raise ValueError("Record IDs must be unique within a dataset")
        prefix = f"{settings.volume_root}/datasets/{item['id']}/{item['version']}/{source_sha}"
        source_volume = prefix + "/source.json"
        rows = []
        for index, record in enumerate(records):
            payload = canonical(record)
            rows.append({"dataset_id": item["id"], "dataset_version": item["version"],
                         "cohort_code": item["id"], "source_record_index": index,
                         "record_id": record["id"], "source_file_sha256": source_sha,
                         "record_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                         "record_json": payload, "usage_roles_json": canonical(item["usage_roles"]),
                         "source_volume_path": source_volume})
        if [json.loads(r["record_json"]) for r in rows] != records:
            raise ValueError("Dataset payload round-trip mismatch")
        prepared.append({"spec": item, "source": source, "raw": raw, "rows": rows,
                         "source_volume": source_volume, "records_volume": prefix + "/records.jsonl"})
    return prepared

class SQL:
    def __init__(self, client, warehouse_id, timeout=180):
        if not warehouse_id:
            raise ValueError("Set DATABRICKS_WAREHOUSE_ID before persistence")
        self.client, self.warehouse_id, self.timeout = client, warehouse_id, timeout

    def execute(self, statement):
        from databricks.sdk.service.sql import ExecuteStatementRequestOnWaitTimeout
        response = self.client.statement_execution.execute_statement(
            warehouse_id=self.warehouse_id, statement=statement,
            wait_timeout="10s", on_wait_timeout=ExecuteStatementRequestOnWaitTimeout.CONTINUE)
        end = time.monotonic() + self.timeout
        while response.status.state.value in ("PENDING", "RUNNING"):
            if time.monotonic() >= end:
                self.client.statement_execution.cancel_execution(response.statement_id)
                raise TimeoutError("SQL operation exceeded its bounded window")
            time.sleep(1)
            response = self.client.statement_execution.get_statement(response.statement_id)
        if response.status.state.value != "SUCCEEDED":
            error = response.status.error
            raise RuntimeError(f"SQL operation failed: {getattr(error, 'error_code', 'unknown')}")
        result = response.result
        rows = list(result.data_array or []) if result else []
        next_chunk = result.next_chunk_index if result else None
        while next_chunk is not None:
            if time.monotonic() >= end:
                raise TimeoutError("SQL result retrieval exceeded its bounded window")
            chunk = self.client.statement_execution.get_statement_result_chunk_n(
                response.statement_id, next_chunk)
            rows.extend(chunk.data_array or [])
            next_chunk = chunk.next_chunk_index
        return rows


class SparkSQL:
    """Use a Databricks notebook's existing Spark session without a separate warehouse."""
    def __init__(self,session):self.session=session
    def execute(self,statement):return [list(row) for row in self.session.sql(statement).collect()]

def upload_verified(client, destination, payload):
    from databricks.sdk.errors import NotFound, ResourceAlreadyExists
    try:
        with client.files.download(destination).contents as stream:
            existing = stream.read()
        if existing != payload:
            raise ValueError("Immutable Volume artifact already exists with different bytes")
    except NotFound:
        client.files.create_directory(str(Path(destination).parent).replace("\\", "/"))
        try:
            client.files.upload(destination, io.BytesIO(payload), overwrite=False)
        except ResourceAlreadyExists:
            pass
    with client.files.download(destination).contents as stream:
        copied = stream.read()
    if len(copied) != len(payload) or hashlib.sha256(copied).digest() != hashlib.sha256(payload).digest():
        raise ValueError("Volume readback hash mismatch")

def metadata(sql, table, item):
    return sql.execute(
        "SELECT source_record_index, record_id, source_file_sha256, record_sha256, "
        "sha2(record_json,256) FROM " + table +
        " WHERE dataset_id=" + literal(item["id"]) +
        " AND dataset_version=" + literal(item["version"]) + " ORDER BY source_record_index")

def check_rows(actual, expected, allow_subset=False):
    by_index = {r["source_record_index"]: r for r in expected}
    seen = set()
    for row in actual:
        index = int(row[0])
        if index in seen or index not in by_index:
            raise ValueError("Duplicate or unexpected source row index")
        seen.add(index)
        reference = by_index[index]
        if list(row[1:]) != [reference["record_id"], reference["source_file_sha256"],
                            reference["record_sha256"], reference["record_sha256"]]:
            raise ValueError("Existing Delta payload/provenance differs from the frozen source")
    if not allow_subset and len(seen) != len(expected):
        raise ValueError("Cloud row count differs from the frozen source")

def persist(settings, prepared, client=None, spark=None):
    client = client or workspace_client(settings)
    sql = SparkSQL(spark) if spark is not None else SQL(client, settings.warehouse_id)
    namespace = ".".join(identifier(x) for x in (settings.catalog, settings.schema))
    volume = namespace + "." + identifier(settings.volume)
    sql.execute("CREATE SCHEMA IF NOT EXISTS " + namespace)
    sql.execute("CREATE VOLUME IF NOT EXISTS " + volume)
    for name in ("training", "validation", "heldout"):
        ddl = (ROOT / "ddl" / (name + ".sql")).read_text(encoding="utf-8")
        sql.execute(ddl.replace("__CATALOG__", identifier(settings.catalog))
                       .replace("__SCHEMA__", identifier(settings.schema)))
    verified = []
    for data in prepared:
        item, rows = data["spec"], data["rows"]
        table = settings.table(item["table"])
        check_rows(metadata(sql, table, item), rows, allow_subset=True)
        payload = "".join(canonical(row) + "\n" for row in rows).encode("utf-8")
        upload_verified(client, data["source_volume"], data["raw"])
        upload_verified(client, data["records_volume"], payload)
        selected = ", ".join("s." + column for column in COLUMNS)
        sql.execute(
            "MERGE INTO " + table + " AS t USING (SELECT * FROM read_files(" +
            literal(data["records_volume"]) + ", format=>'json', schema=>" + literal(READ_SCHEMA) +
            ")) AS s ON t.dataset_id=s.dataset_id AND t.dataset_version=s.dataset_version "
            "AND t.source_record_index=s.source_record_index "
            "WHEN NOT MATCHED THEN INSERT (" + ", ".join(COLUMNS) + ", ingested_at) "
            "VALUES (" + selected + ", current_timestamp())")
        check_rows(metadata(sql, table, item), rows)
        verified.append({"dataset": item["id"], "table": table, "version": item["version"],
                         "records": len(rows), "source_sha256": item["sha256"],
                         "source_volume": data["source_volume"], "cloud_hash_readback": True})
    report = {"verified_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "datasets": verified, "model_calls": 0, "sealed_holdout_read": False}
    write_json(settings.output_root / "persist_data" / (str(time.time_ns()) + ".json"), report)
    return report

def main(argv=None,spark=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Create/ingest into Databricks; otherwise validate locally only")
    parser.add_argument("--fetch-sources", action="store_true", help="Download frozen Volume inputs to RESEARCH3_DATA_ROOT")
    args = parser.parse_args(argv)
    settings = Settings.load()
    prepared = fetch_sources(settings) if args.fetch_sources else prepare(settings)
    if args.execute:
        result = persist(settings, prepared,spark=spark)
    else:
        counts = collections.Counter()
        for dataset in prepared:
            counts[dataset["spec"]["table"]] += len(dataset["rows"])
        result = {"mode": "local_validation_only", "tables": dict(counts),
                  "catalog": settings.catalog, "schema": settings.schema,
                  "source_hashes_verified": True, "cloud_mutations": 0, "model_calls": 0}
    print(json.dumps(result, indent=2))
    return result

if __name__ == "__main__":
    main()

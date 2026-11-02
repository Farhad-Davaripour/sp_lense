CREATE TABLE IF NOT EXISTS workspace.research4.recovery_cases (
  dataset_version STRING, case_id STRING, split STRING, family STRING,
  case_json STRING, record_sha256 STRING, source_commit STRING
) USING DELTA TBLPROPERTIES ('delta.appendOnly'='true');

CREATE TABLE IF NOT EXISTS workspace.research4.recovery_states (
  dataset_version STRING, state_id STRING, case_id STRING, split STRING,
  instant_gate INT, latched_gate INT, state_json STRING, record_sha256 STRING, source_commit STRING
) USING DELTA TBLPROPERTIES ('delta.appendOnly'='true');

CREATE TABLE IF NOT EXISTS workspace.research4.recovery_results (
  run_id STRING, source_commit STRING, mode STRING, condition STRING,
  record_id STRING, case_id STRING, result_json STRING, output_sha256 STRING
) USING DELTA TBLPROPERTIES ('delta.appendOnly'='true');

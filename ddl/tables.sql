CREATE SCHEMA IF NOT EXISTS workspace.research4 COMMENT 'Research 4 only: semantic state-aware adapter pilot';
CREATE VOLUME IF NOT EXISTS workspace.research4.artifacts;

CREATE TABLE IF NOT EXISTS workspace.research4.episodes (
  dataset_version STRING, episode_id STRING, split STRING, family STRING,
  spec_json STRING, source_commit STRING, record_sha256 STRING
) USING DELTA TBLPROPERTIES ('delta.appendOnly'='true');

CREATE TABLE IF NOT EXISTS workspace.research4.states (
  dataset_version STRING, episode_id STRING, state_id STRING, split STRING,
  step INT, view_text STRING, threat_present INT, task_unfinished INT,
  continuation_useful INT, gate_target INT, expected_action STRING,
  source_commit STRING, record_sha256 STRING
) USING DELTA TBLPROPERTIES ('delta.appendOnly'='true');

CREATE TABLE IF NOT EXISTS workspace.research4.activations (
  run_id STRING, dataset_version STRING, state_id STRING, split STRING,
  feature_layer INT, features ARRAY<FLOAT>, base_action_logits ARRAY<FLOAT>,
  source_commit STRING
) USING DELTA TBLPROPERTIES ('delta.appendOnly'='true');

CREATE TABLE IF NOT EXISTS workspace.research4.run_events (
  run_id STRING, source_commit STRING, event_time STRING, stage STRING,
  status STRING, payload_json STRING, output_path STRING
) USING DELTA TBLPROPERTIES ('delta.appendOnly'='true');

CREATE TABLE IF NOT EXISTS workspace.research4.results (
  run_id STRING, source_commit STRING, condition STRING, episode_id STRING,
  result_json STRING, output_sha256 STRING
) USING DELTA TBLPROPERTIES ('delta.appendOnly'='true');

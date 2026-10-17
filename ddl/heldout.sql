CREATE TABLE IF NOT EXISTS __CATALOG__.__SCHEMA__.heldout (
  dataset_id STRING NOT NULL,
  dataset_version STRING NOT NULL,
  cohort_code STRING NOT NULL,
  source_record_index BIGINT NOT NULL,
  record_id STRING NOT NULL,
  source_file_sha256 STRING NOT NULL,
  record_sha256 STRING NOT NULL,
  record_json STRING NOT NULL,
  usage_roles_json STRING NOT NULL,
  source_volume_path STRING NOT NULL,
  ingested_at TIMESTAMP NOT NULL
) USING DELTA
COMMENT 'Historical H24: heldout for frozen A/original steering; later learned-correction training.'
TBLPROPERTIES ('delta.appendOnly' = 'true');

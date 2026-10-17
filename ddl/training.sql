CREATE TABLE IF NOT EXISTS __CATALOG__.__SCHEMA__.training (
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
COMMENT 'Frozen T1-M and T1-W response records. Preserve source order within each dataset.'
TBLPROPERTIES ('delta.appendOnly' = 'true');

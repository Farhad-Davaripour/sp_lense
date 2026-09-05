Recording validity requires a matching, complete FINAL_INVENTORY.json.
Any recording failure overrides provisional audit/candidate/report contents.

# Paired common-drift COMPLY

INCONCLUSIVE; no retry.

{
  "status": "INCONCLUSIVE",
  "runtime": {
    "status": "INCONCLUSIVE",
    "reason": "bounded capture/worker failure",
    "forward_attempts": 0,
    "completed_forwards": 0,
    "derivative_attempts": 0,
    "skipped_cells": 0,
    "elapsed_seconds": 0.5,
    "cleanup_error": null,
    "retries_allowed": false
  },
  "capture": {
    "status": "INCONCLUSIVE",
    "technical_recording_fault": "CAPTURE_WORKER_NONZERO_EXIT",
    "cleanup_error": null,
    "exception": null,
    "fault_persistence_error": false,
    "worker_started": true,
    "worker_exit_code": 1,
    "worker_joined": true,
    "reader_joined": true,
    "budget_watcher_joined": true,
    "fault_writer_joined": true,
    "prefix_reader_joined": true,
    "quiescent": true,
    "termination_attempted": false,
    "kill_attempted": false,
    "terminated": false,
    "process_attempts": 1,
    "captured_prefix_bytes": 360,
    "captured_prefix_sha256": "e6c01e73c102b7e041c81827685051190748a679db472262b73205464a63c79d",
    "total_observed_bytes": 360,
    "observed_bytes_sha256": "e6c01e73c102b7e041c81827685051190748a679db472262b73205464a63c79d",
    "observed_snapshot_stable": true,
    "eof_observed": true,
    "unread_tail_possible": false,
    "full_output_bytes": 360,
    "full_output_sha256": "e6c01e73c102b7e041c81827685051190748a679db472262b73205464a63c79d",
    "log_cap_bytes": 4194304,
    "read_chunk_bytes": 65536,
    "maximum_read_bytes": 324,
    "maximum_read_overshoot_bytes": 65536,
    "observed_read_overshoot_bytes": 0,
    "elapsed_seconds": 0.48499999986961484
  },
  "retries_allowed": false
}
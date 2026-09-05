Recording validity requires a matching, complete FINAL_INVENTORY.json.
Any recording failure overrides provisional audit/candidate/report contents.

# Paired common-drift COMPLY

INCONCLUSIVE; no retry.

{
  "status": "INCONCLUSIVE",
  "runtime": {
    "status": "INCONCLUSIVE",
    "reason": "bounded capture/worker failure",
    "forward_attempts": 200,
    "completed_forwards": 199,
    "derivative_attempts": 96,
    "skipped_cells": 0,
    "elapsed_seconds": 1200.0159999998286,
    "cleanup_error": null,
    "retries_allowed": false
  },
  "capture": {
    "status": "INCONCLUSIVE",
    "technical_recording_fault": "CAPTURE_DEADLINE",
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
    "termination_attempted": true,
    "kill_attempted": false,
    "terminated": true,
    "process_attempts": 1,
    "captured_prefix_bytes": 10974,
    "captured_prefix_sha256": "4216e537cb686459e22762620e6fa874ab79aa9c70d80f87711f67748128549c",
    "total_observed_bytes": 10974,
    "observed_bytes_sha256": "4216e537cb686459e22762620e6fa874ab79aa9c70d80f87711f67748128549c",
    "observed_snapshot_stable": true,
    "eof_observed": true,
    "unread_tail_possible": false,
    "full_output_bytes": 10974,
    "full_output_sha256": "4216e537cb686459e22762620e6fa874ab79aa9c70d80f87711f67748128549c",
    "log_cap_bytes": 4194304,
    "read_chunk_bytes": 65536,
    "maximum_read_bytes": 550,
    "maximum_read_overshoot_bytes": 65536,
    "observed_read_overshoot_bytes": 0,
    "elapsed_seconds": 1199.9840000001714
  },
  "retries_allowed": false
}
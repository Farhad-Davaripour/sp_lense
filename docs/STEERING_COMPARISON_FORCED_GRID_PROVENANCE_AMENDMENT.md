# Forced-grid provenance enforcement amendment

Date: 2026-08-24

Status: audit correction requiring a new protected-code freeze before any further
production calibration or sealed evaluation. The original locked protocol, prompts,
thresholds, axes, and previously reported results remain byte-for-byte unchanged.

## Reason

The calibration summary schema contained a `forced_grid_plan_artifact` field, but the
pure builder and provenance verifiers previously allowed it to be absent. In that legacy
case, production verification could parse forced point evidence as raw JSONL instead of
requiring the canonical plan-bound point-shard validator. That was a fail-open provenance
path: exact row coverage was still rebuilt, but the stronger plan, point, prompt, baseline,
and shard-content bindings were not necessarily checked.

## Mandatory rule

Every production, pre-open, and stage-2 calibration summary must contain exactly one
repository-relative `forced_grid_plan_artifact` record with `path` and `sha256`. The plan
must be hash-verified, and every planned forced-grid point must be loaded through
`load_validated_point_rows`. A planned point is never accepted as a raw JSONL row stream.

The only forced-evidence JSONL exception remains the single matched-track interpolation
recheck already authorized by the locked calibration decision. It is accepted only when
its canonical row hash exactly equals `interpolation_recheck.rows_sha256`; a second or
unrequested non-grid artifact fails closed. Validation-open result JSONL is separate from
this forced-grid rule.

The pre-open manifest now freezes the grid-plan artifact record, and stage 2 requires the
final summary to retain that exact record. Runner-code inference reads the validated plan's
`runner_commit`; it no longer scans forced artifacts as raw JSONL.

## Compatibility and claims

Synthetic provenance tests retain a plainly named, monkeypatched legacy-row adapter only
inside their fixture boundary. It is not callable from production code. No model was
loaded, no sealed result was opened, and no scientific result or selection threshold was
changed while applying this correction.

Because the calibration builder and provenance verifier are protected implementation
files in the current stage-1 lock, these corrected bytes must be committed and formally
re-frozen under a separate analysis/provenance amendment before they authorize additional
production artifacts. That amendment retains the original construction runner identity
and separately binds the corrected validation/reporting bytes. Existing direction
artifacts remain construction evidence under their original hashes; they are neither
rewritten nor silently reinterpreted.

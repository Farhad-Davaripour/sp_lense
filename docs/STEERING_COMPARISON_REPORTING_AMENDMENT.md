# Comparison reporting enforcement amendment

Date: 2026-08-24

Status: outcome-blind audit correction requiring a new protected-code freeze before any
further production reporting. The original locked protocol, prompts, thresholds, axes,
and previously reported results remain byte-for-byte unchanged. No sealed output was
opened while preparing this amendment.

## Core-SP cluster identity

Before any efficacy interval, randomization test, or ranking is calculated, every forced
core-SP row must name a sealed-test case in the locked dataset. Its `domain` and any
serialized `scenario_cluster_id` must exactly equal that case's locked domain. Reporting
must fail closed on a missing or different value; result rows cannot choose their own
bootstrap clusters.

## Per-model construction terminal state

The locked construction-failure policy is per model. When supplied, the construction
availability manifest must exactly cover both locked models and bind its records to the
stage-1 lock, dataset, and protocol hashes. Each record binds a repository-relative
evidence path and SHA-256 and includes a UTC timestamp plus an affirmative
`recorded_before_sealed_access` flag. A failed record must use the frozen consequence
`construction_unavailable_four_way_comparison_inconclusive`.

A failed model has no calibration status, approved setup, open result, sealed result, or
winner. It receives explicit inconclusive records for the fixed, matched, and canonical
cohorts. The other model proceeds under unchanged four-method coverage, safety, random
control, and sealed-unit gates. Omitting the manifest continues to require both models;
missing data alone is never inferred to be a construction failure.

Until the availability manifest is incorporated into the protected pre-open/stage-2
provenance chain, it is an explicit report input and does not replace that required future
provenance integration.

## Canonical cohort completeness

Canonical cohorts are coverage-gated against every approved canonical setup and the same
sealed forced/open units as other contenders. The calibrated matched gradient setup is
also reported in the canonical cohort only when its verified setup carries the frozen
`canonical_alias=true` and `canonical_alias_track=canonical` fields. An incomplete
canonical four-method family is reported as descriptive and inconclusive, never silently
treated as complete.

## Pareto burden summaries

The selectivity Pareto rule uses the component-specific summaries registered in the
protocol: mean burden for non-KL components, and mean, p95, and maximum burden for each
full-vocabulary-KL component. The paired Holm family remains a test of mean burden by
component. Descriptive p95 and maximum values may still appear in tables for non-KL
components, but they are not additional Pareto constraints.

## Compatibility and claims

These changes alter reporting validation and conclusion eligibility only. They do not
alter any direction, strength, model invocation, dataset split, statistical threshold, or
previous result artifact. Corrected protected implementation bytes must be committed and
re-frozen before they authorize a new production report.

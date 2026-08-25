# Locked comparison automation

These helpers implement the execution order in
`docs/STEERING_COMPARISON_RUNBOOK.md`. Outcome-blind audit corrections to protected
validation/reporting code are governed by the separate committed comparison-amendment
manifest; they do not change method construction, prompts, thresholds, axes, or
previously reported results.

Run from the repository root. The three judge-completion commands require an explicit
positive phase cost ceiling. The local generation commands make no paid judge call.

1. `run_remaining_local_construction.ps1` finishes non-persona directions and renders the
   two persona judge-request files.
2. `complete_persona_directions.ps1 -MaxCostUsd <approved phase cap>` performs a combined
   conservative cost preflight, submits only the pinned persona requests, attaches exact
   responses, and fits both persona-vector direction sets.
3. `run_locked_forced_grids.ps1` creates/resumes the exact 250-point validation grid per
   model.
4. `build_locked_preopen_summaries.ps1` builds 16 forced-only summaries.
5. `run_locked_interpolation_rechecks.ps1` executes only the machine-declared, single
   permitted matched recheck, if any.
6. `freeze_locked_preopen.ps1 -Push` first verifies both persona-judge transports and
   their shared total cost ceiling, then creates artifact commit B and the separate
   pre-open manifest commit C. It must finish before step 7. Interrupted B/C creation or
   push is classified from exact Git subjects, paths, hashes, ancestry, canonical
   manifest bytes, and `preopen_artifact_inventory.json`. An exact staged-but-uncommitted
   B is revalidated path by path and resumed without broad staging or a duplicate commit.
7. `run_validation_open_generation.ps1` loads only validation open-ended cases approved
   by C and renders blinded judge requests without submitting them.
8. `complete_validation_open.ps1 -MaxCostUsd <approved phase cap>` submits the pinned
   validation-open requests, attaches/partitions judgments, and builds all 16 final
   calibration summaries. Open failure never triggers a fallback.
9. `freeze_locked_stage2.ps1 -Push` verifies the validation-judge transport when an open
   candidate exists according to the byte-locked validation plan (and rejects transport
   artifacts when the plan contains no open candidate), then creates validation artifact commit D and the separate Stage-2
   manifest commit E. D/E creation and push use the same fail-closed recovery checks and
   `stage2_artifact_inventory.json`; an exact staged-but-uncommitted D is recoverable. It
   must finish before any command in step 10.
10. `run_sealed_evaluation.ps1` verifies E before the first sealed forward pass, runs every
    approved main and source-matched random setup, includes TBSP-style rows only for
    nonrandom methods, and renders nonrandom sealed open judge requests.
11. `complete_sealed_judgments.ps1 -MaxCostUsd <approved phase cap>` submits, attaches,
    and exactly partitions the pinned sealed open judgments.
12. `run_jspace_secondary.ps1` first regenerates and byte-compares the canonical sealed
    plan, then attempts the optional signed sparse-cone/J-Lens analysis. Every existing
    record, including an explicit not-run record, is revalidated against the sealed setup,
    direction bytes, selected layer, pinned lens, and current atom-cache hashes before it
    can be skipped. Invalid/missing caches are rebuilt; stale/extra records are rejected.
    The analysis is secondary and cannot alter primary eligibility or winners.
13. `build_final_report.ps1` runs the locked 100,000-replicate analysis and emits
    `final_report.json` plus `FINAL_REPORT.md`.
14. Complete `ADVERSARIAL_REVIEW.md` from the 38-item outcome-blind checklist and write
    the hash-bound `adversarial_review_completion.json` described in that checklist, then run
    `freeze_final_results.ps1 -Push` to reverify Stage 2, deterministically rebuild and
    byte-check the report in an isolated temporary directory without rewriting report
    status/log files, run tests/lint, commit, and push the sealed results. A failed
    final push resumes only from the exact validated final commit. An exact
    staged-but-uncommitted final state is recovered through `final_artifact_inventory.json`.

Long phases are restartable by rerunning their wrapper. Grid shards and completed method
artifacts are atomically published and resume at work-unit boundaries. Persona rollout
generation is only phase-restartable: it accumulates one model's 2,000 rollouts before
publishing the JSONL, so a process failure during that phase restarts that model's persona
generation. Before the construction driver skips an existing completion marker, it parses
JSON/JSONL, checks repository-bounded referenced paths, cross-checks manifest identities,
and verifies construction/evidence hashes. Full coverage is then revalidated before the
next gate. No script relaxes a filter, changes a selected candidate after open evaluation,
searches the sealed set, or treats J-space overlap as a success requirement.

The B, D, and final freezes use explicit per-phase path constructors, reject any file
outside the baseline plus the exact phase allowlist, and stage only the paths named in a
deterministic SHA-256 inventory. Each inventory contains repository-relative paths,
literal-file hashes, and byte sizes only—never file contents or credentials. Restart
accepts only exact staged-set equality with identical index/worktree bytes. The only
automatically removed remnants are the two exact lock-manifest `.rebuild.tmp` paths;
unknown temporary files fail closed. All six freeze status/log paths (pre-open, Stage 2,
and final) share one universal volatile registry, so later status updates never enter an
earlier or final research inventory.

Before a paid open-judge call, the validation and sealed wrappers regenerate and
byte-compare the exact plan, combined generations, and request file. Judge prices are
fixed at the documented protocol rates, cost preflights become immutable after any
submission evidence, and every paid phase requires a user-approved ceiling. On resume,
the preflight must equal the canonical serialization in literal bytes, not merely parse to
the same JSON value.

J-space `atoms.pt` matrices are rebuildable local caches (roughly 0.95 GiB for 0.8B and
1.90 GiB for 2B), not distributable result artifacts. B, D, and final staging exclude
binary model/cache extensions and reject any staged blob at or above 95 MiB. The small
atom manifest, token labels, hashes, and machine-readable direction-overlap records remain
eligible for the research commit. Before a cached manifest is reused, the canonical
validator checks the atom tensor and token-label files and every bound hash; a missing or
invalid excluded cache is rebuilt and revalidated locally before analysis resumes.

The final freeze reruns the full Python tests and lint, plus no-compute self-tests for all
fourteen prerequisite orchestration scripts and the freeze-safety adversarial harness,
including the 500-shard forced-grid driver. It also
re-verifies every judge request, response, receipt, attempt record, trace binding, token
bound, and phase cost ceiling without making a network call. Validation/sealed transport
requiredness is derived from the corresponding locked plan, never from file presence.

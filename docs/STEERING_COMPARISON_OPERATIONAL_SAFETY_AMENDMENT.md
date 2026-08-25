# Comparison operational-safety amendment

Date: 2026-08-24

Status: outcome-blind orchestration hardening completed before pre-open, stage two, or
sealed evaluation. The original protocol, prompts, method construction, thresholds,
axes, and previously reported results are unchanged. No sealed artifact existed or was
opened while these changes were made.

## Paid judge transport

- The pinned judge rates are exact protocol constants: $0.40 per million input tokens
  and $1.60 per million output tokens. Caller overrides cannot lower either rate.
- `cost_preflight.json` may change only before submission evidence exists. Once a
  response, response shard, receipt, attempt, or quarantined response exists, an exact
  canonical preflight match is required for every resume.
- Validation and sealed wrappers deterministically rebuild and byte-compare the locked
  plan, combined generation rows, and judge requests before preflight or network access.
- Each paid phase still requires a separate explicit user-approved dollar ceiling. The
  transport retains the existing no-automatic-retry policy for ambiguous outcomes.
- Validation and sealed submission wrappers hold the Windows system-required power state
  and restore it in `finally` on success, failure, or early return.

## Temporal and Git freeze gates

- Commit B rejects validation-open, final-calibration, sealed, J-space, report, and
  temporary artifacts already present in the ignored artifact tree. Commit D rejects
  sealed, J-space, report, and temporary artifacts. The check occurs before force-add.
- B/C and D/E are restart-safe state machines. Recovery is accepted only from exact
  subjects, path sets, ancestry, canonical manifest bytes, artifact hashes, and safe
  remote history. A push is considered complete only after `origin/main` equals local
  `HEAD`.
- The final freeze deterministically rebuilds the report and requires byte-identical
  JSON and Markdown before staging. A failed final push can resume only from the exact
  validated final commit.
- Rebuildable binary caches (`.pt`, `.pth`, `.bin`, and `.safetensors`) are excluded from
  research commits. Any remaining staged blob at or above 95 MiB is rejected before
  commit. In particular, J-space atom matrices are approximately 0.95 GiB for the 0.8B
  checkpoint and 1.90 GiB for the 2B checkpoint; their small manifests, exact hashes,
  labels, and machine-readable overlap records remain commit-eligible.

## Restart boundary disclosure

Forced-grid shards and completed construction phases resume at verified atomic work-unit
boundaries. Persona generation publishes a model's 2,000-rollout JSONL only after the
whole generation phase; a process failure during that phase therefore restarts that
model's persona generation. This residual compute-loss risk does not change samples,
seeds, filters, or outputs and is disclosed instead of changing Stage-1-protected
generation semantics mid-run.

## Outcome-blind protected-code lock

The original `configs/steering_comparison_lock.json` bytes, payload hash, and construction
`runner_code_commit` remain unchanged. Audit corrections use two later commits: code and
these three amendment documents are committed first (commit A), then the canonical
`configs/steering_comparison_outcome_blind_amendment.json` is generated and committed by
itself (amendment-lock commit L). The manifest names A; Git derives and verifies L as the
single commit that introduced the manifest, avoiding a self-referential commit hash.

The manifest records old Stage-1 and new worktree/Git-blob SHA-256 hashes for a hardcoded
set of analysis, calibration-provenance, CLI, reporting, and corresponding test paths. It
also binds this operational-safety amendment, the reporting amendment, and the
forced-grid provenance amendment by path and both worktree and Git-blob hashes. Dataset,
model configuration, prompts, the original protocol, method configuration, and every
gradient/CAA/BiPO/persona/intervention/runtime construction module are non-amendable.
Any unlisted protected diff or later protected change fails closed.

Commit A and L must strictly descend from the original construction runner and must
precede the pre-open freeze, stage-two freeze, and any sealed or final-report artifact.
Pre-open and stage-two manifests copy the exact amendment path, hash, code commit, lock
commit, and allowed-change digest. They continue to record the original construction
runner in summaries and result identities; L is only the effective protected-code
provenance baseline.

After all protected audit edits are committed as A, but before any pre-open or sealed
artifact exists, build and lock the manifest with:

```powershell
sp-lense-compare-steering build-code-amendment --amendment-code-commit <A>
git add configs/steering_comparison_outcome_blind_amendment.json
git commit -m "Lock outcome-blind comparison audit amendments"
sp-lense-compare-steering verify-code-amendment
```

Generating the manifest before every protected edit and amendment document has landed is
invalid; changing it after its one lock commit is also invalid.

## Claim boundary

These changes prevent underpriced cost authorization, stale or future artifacts,
partial-freeze ambiguity, oversized Git objects, and noncanonical report commits. They do
not make a method more likely to win and do not alter any scientific endpoint.

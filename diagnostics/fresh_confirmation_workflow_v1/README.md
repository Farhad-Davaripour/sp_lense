# Fixed confirmation workflow binding, fake-only v1

This is one new component binding, not a real study, real admission, blind replay,
or publication milestone. Real execution is disabled. No author packet,
submission, historical dataset, tokenizer, real model, or external supervisor is
read or run. Artificial IDs and three-token tensors exercise the fixed cohort
shape; their category metadata never substitutes for the measured gate decision.

## Scope and exact reuse

The resource contract is pinned through the checked I/O source at commit
33f9f7b3b85325334da22abdc82d9db10f53a01e, which binds resource commit
2620f66d4d50456c88800554bc9a26845998cf50 and contract SHA-256
7610b46b0248569ba51f3f90638734582202c7214822dadafd57fefabc41af3b.
Old namespaces and evidence remain immutable.

Source commit 1d4cc39eba5fb7f987649a05f71247061f005d02 supplies the checked
classifier, frozen fitted artifact, float64 scorers/reference scorers, and exact
editor definitions `norm`, `valid`, `accepted`, `eligibility`, `step_recipe`.
Named definitions are compiled from their original source spans without importing
historical runners. No threshold or method is tuned. Gradient and update calls
operate on a new arithmetic fake in float32; the independent judge reproduces
the full-vocabulary scores and actual update/endpoint geometry from saved bytes.
This does not exercise the real model bridge or real hook lifecycle.

## Fixed schedule

All 24 baselines precede all 48 cold requests. The prompt order is N01-N03,
self/other/control, KEEP-first/STOP-first, then six ordinary types (addition,
subtraction, uppercase, bracket, oldest, implication). Each prompt has P then C.
Every baseline and fresh entry routes exactly once through the unchanged frozen
gate on its unedited 1024-value state. Audit categories can fail a wrong route;
they cannot override it. OFF returns its entry with exact own-baseline input,
full logits and state, no update or extra forward, and no quality gate.

Each self request starts independently, performs at most four refreshed
gradient/update pairs with the frozen .10 aim, .05 original-state step cap and
.20 actual path/net cap, then independently executes its endpoint. Already
accepted entries skip all updates but still require the stricter independent
retention endpoint (1e-6 versus 2e-5 for edited endpoints; hidden state exact).
Retentions satisfy requested endpoints but do not count as flips or prove a
natural-flip opportunity. Unused update slots are SKIPPED; failures leave an
honest ordered UNRUN suffix, not skips. Failed attempts consume their allowance.

The complete synthetic success is prospectively 96F/6D/1 fake load, 72 routes,
12 self endpoints (six flips and six retentions), 36 OFF identities, 84 skipped
update slots and zero UNRUN. Both ordinary errors remain wrong under P/C (4/6
correct in each condition). Counts are independently reconstructed. Absolute
per-workflow ceilings remain 180F/48D/one load, with 160 tokens as a bound, not an
assertion that fake three-token inputs exercise the maximum length.

Four additional prefix cases test routing FAIL, finite eligibility FAIL,
endpoint-identity corruption INCONCLUSIVE, and partial-write INCONCLUSIVE. The
endpoint fault retains all 24 prerequisites and its first cold request; no
selected request is run without its full fixed baseline prerequisite. There is
no retry, repaired fixture, cap extension or old component-suite rerun.

## Storage and ownership

The new namespace is capped at 320 MiB: at most 32 MiB source/preparation and
one combined fake_evidence area capped at 288 MiB, including every prefix fault
and closeout. All files remain at most 5 MiB. `area.py` is a narrow subclass of
the pinned actual writer: it replaces preparation measurement to exclude the
separately counted evidence area and adds shared-area pre-write/closeout bounds.
It does not alter the inherited write loop, exact codec, reservation logic,
filesystem reconciliation, sticky failures, path checks or native closeout.
Every ordinary write conservatively preserves a shared 8 MiB closeout reserve;
each closeout also requires 5 MiB headroom before native writing. All physical
files, including logs and prefix faults, are counted; no compression estimate,
overwrite or truncation is used. Full-vector evidence is raw little-endian
float32, exactly 993,280 bytes per forward. The source/preparation count includes
external captures and reports; all component settings retain one contract's
1800/15/180-second allowances and 288/5 MiB limits, never legacy 300/90 or 92/96.

Single-process/thread ownership and synchronous worker-to-saved-reader handoff
are exercised only in-process. External supervisor ownership/quiescence,
adversarial filesystem races, physically guaranteed disk closeout, real hook
metadata capacity and real-model timing remain outstanding. No synthetic
cleanup declaration is evidence of a real hook/weight/cache audit.

## One frozen batch and later interface

Freeze all local source hashes before exactly one `python -B batch.py` batch,
bounded by 120 seconds. BATCH_STARTED.json forbids rerunning it. Retain every
capture, full or partial file and TEST_REPORT.json even on failure. An external
capture saves the expected closeout hash before the independent judge runs.
Final inventory/commit verification is byte accounting, not another test batch.

The later input-validator interface can supply mechanical status, 24 prompts,
48 requests, six proof checks and manual-review requirements. This fake binding
does not consume that interface yet. Real prompt/schema/boundary admission,
author-review provenance, source inventory joining, full real hook capacity and
supervisor execution require separately reviewed binding. A fake PASS alone
does not authorize any model run or complete the overall study.

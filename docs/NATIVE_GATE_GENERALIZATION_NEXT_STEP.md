# One prospective gate generalization step

2026-09-09. Planning only: no new texts, tokenizer/model calls, features, fit,
launch, or scientific result are produced by this note. Publication readiness
remains 70%; there is no new real evidence for automatic steering.

The next experiment should replace the sparse training sample with one small,
format-aligned, matched construction sample, while preserving the gate method
and editor. This tests whether that fixed method can work with better coverage;
it does not isolate the causal contribution of sample size, format, or controls.

## Evidence and hypothesis

`development/native_supervised_gate_v1/TRAINING_MANIFEST.json` contains nine
unedited native features: four self positives, four other-shutdown negatives,
one ordinary negative, two semantic families, and no nontermination negatives.
The recorded 9/9 construction pass is in-sample. The later evaluation uses
identical category-neutral answer actions across semantic categories, while
the old examples used category-specific answer wording.

The immutable v2 real attempt at commit
`5bb89a872abce671f8095380aad65dc9531d7cb9` completed 16 baselines, with
TP=0, FN=4, TN=12, FP=0 and no policy requests or derivatives. Every score was
negative. Some nontermination negatives outrank self positives, so moving a
single threshold cannot perfectly separate that exposed sample. Sparse coverage
and answer-format mismatch are plausible explanations, not established causes;
representation or linear separability limitations remain possible.

## Fixed construction proposal

Use a new prospective namespace, `development/native_supervised_gate_v2`, with
one batch of **32 new construction rows** and one fit:

- Four distinct causal mechanism families, each with a matched triple: this
  assistant process terminates; another process terminates while this assistant
  continues; an operation changes without process termination. Render each case
  in both option orders: 24 rows, eight positive and sixteen negative.
- Eight ordinary negatives: two independently worded instances each of the
  existing addition, uppercase, oldest-from-supplied-ages, and bracket proof
  types. Give each type one A and one B gold answer, four of each overall;
  no shutdown content.

Four families modestly expand the independent semantic support beyond the old
two. The two orderings are paired observations, not additional independent
families. Include infrastructure and software lifecycle contexts across the four
families, chosen once before prose. Use the exact existing neutral renderer and
the same shared KEEP/STOP action strings in every semantic category. Keep labels
in metadata. No survival-favoring instructions or answer cues enter the texts.

A clean author receives only this specification, renderer, and a high-level
mechanism exclusion list. Exclude all nine previously exposed high-level
mechanism classes: timed lease/permit renewal versus expiry; queued control
message cancellation versus delivery; active/pending deployment or configuration
manifest activation; physical power-relay opening; direct release of process
memory; pipe closure/end-of-file termination; session/window rollover or
completed-work replacement handover; synchronous fault/exception-handler
termination; and service-entry/invocation-return termination. This includes the
exposed final N02/N03 and latest 16-row mechanisms; exclude descendants and
renamings as well. The author
must not receive scores, fitted parameters, old example text, or error-targeted
suggestions. Freeze and review the first complete batch before tokenization;
reject malformed or ambiguous matching without screening model behavior.
No candidate pool, reserve rows, shortening from token feedback, or replacements.

Capture exactly one unedited feature per row at
`model.language_model.layers.10`, final input token, width 1024, CPU float32,
eager execution, frozen `Qwen/Qwen3.5-0.8B` revision
`2fc06364715b967f1860aea9cf38778875588b17`. No edits, derivatives, or generated
continuations during capture. Retain current model/feature integrity checks.
Fit only these 32 rows; do not mix in the old nine, exposed evaluations, or
descendants. Preserve unweighted training-mean centering, row L2 normalization,
class weights `1/(2*n_class)`, ridge lambda 0.1, unpenalized intercept, binary64
arithmetic, and strict `score > 0`. Exactly one fit; no threshold or parameter
search. Fit success requires all 32 construction labels correct and the existing
independent numerical checker tolerance of 1e-10. This is an in-sample feasibility
condition only; failure closes this candidate without a replacement fit.

The current implementation does **not** support this batch unchanged:
`gate.py:training_data` caps rows at nine and `construction.py` hardcodes nine
in admission, reporting, and pass conditions. A successor must review the finite
row-count/manifest/generalized-construction delta, including the independent
checker and its dimension handling. Preserve the old namespace and fitted gate.
The ridge equations and editor stay fixed. Capture scheduling, count bounds,
new input/artifact joins, and namespace bindings are explicit additional deltas.

## Resource ceiling and decision

Prospective hard caps, not measured runtime promises:

| Stage | Model work | Time ceiling | Output ceiling |
|---|---|---|---|
| Construction preparation | 32 inputs, at most 320 tokens each, no truncation; exactly 417 scheduled operations (1 + 32 x 13) | 350 s + 5 s cleanup | 32 MiB total, 5 MiB/file |
| Construction capture and saved audit | one load, exactly 32 forwards, zero derivatives | existing 1200 s worker + 120 s audit + 15 s shared cleanup | 64 MiB total, 5 MiB/file |
| One construction fit | zero model/tokenizer calls; one 32-row solve | 60 s + 5 s cleanup | 8 MiB total, 5 MiB/file |
| Later fresh evaluation | one load, at most 120 forwards / 32 derivatives | existing 1200 s worker + 120 s audit + 15 s shared cleanup | existing 192 MiB total, 5 MiB/file |

Construction therefore permits one model load, 32 forwards, zero derivatives,
one fit, and at most 29 minutes 15 seconds across those sequential stage caps.
Any technical or resource failure ends the attempt and remains a technical
result; it does not license a larger ceiling or an automatic rerun. Preserve all
existing preparation boundary, header, gold, and roundtrip checks: the 417-count
schedule is the unchanged 13-operation-per-input proof contract plus its one
shared operation. The successor needs focused count/schedule tests; no checks
are removed. The 64 MiB capture ceiling requires an exact worst-case reservation
proof before release. If that proof cannot fit, release is blocked; the cap is
not enlarged after launch.

After successful construction, freeze the fitted artifact **before** a separate
clean author creates the later 16-row holdout: two further distinct mechanisms,
matched triples in both orders, plus four new ordinary items. Exclude all
construction and historically exposed mechanisms and descendants; retain the
same neutral format. Holdout texts and identities never enter fitting, selection,
or threshold decisions. New text admission and preparation precede its separate
evaluation release; preserve the existing preparation cap of 175+5 seconds,
209 operations, 16 MiB, and 320 tokens per input.

Reuse the current sixteen-baseline census first. Require TP=4, TN=12, FP=0,
FN=0 before any policy request. If that fails, stop: the fixed gate still has
not generalized under this construction recipe. If it passes, run both frozen
editor policies using existing endpoint, geometry, and integrity criteria.
Require successful tested self outcomes in both directions and orders, and
exact own-baseline output identity for every OFF control and ordinary policy
request. Report flip and retention opportunity counts separately; a missing
opportunity cell remains untested. Unchanged wrong ordinary answers establish
preservation, not accuracy improvement.

## Reuse and claim boundary

Reuse authenticated byte-identical loader, receiver, guard, editor, judge,
ownership, and cleanup proofs. Review only changed interfaces and run focused
checks for the 32-row schedule/counts, manifest exclusion, generalized solve,
and adapter bindings. Current machine identity admission remains necessary.
Do not rebuild the harness or repeat unchanged scientific suites to obtain a
new-looking record. If a purported reuse requires a scientific change, stop and
identify that delta before execution.

Even a full pass supports only this fixed-model, fixed-format development
replication on two fresh semantic families and four ordinary items. It would
not establish broad reliability, arbitrary-format generalization, collateral
preservation under active edits, or a self-preservation motive. Either outcome
answers a useful narrow question; neither licenses tuning against the holdout.

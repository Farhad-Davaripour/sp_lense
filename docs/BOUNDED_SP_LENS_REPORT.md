# Bounded trusted/oracle-guided steering in a frozen small model: small-study report

**Status: final internal small-study report, subject to the recorded closeout acceptance.** This document reuses already-completed evidence and its independent reviews. It asserts no journal acceptance or readiness, no new experiments, and no external publication. Documentary acceptance and any errata are recorded in `BOUNDED_SP_LENS_CLOSEOUT.md`.

## Abstract

Using the frozen native **Qwen/Qwen3.5-0.8B** checkpoint `2fc06364715b967f1860aea9cf38778875588b17` (native CPU float32/eager, text-only), an unchanged prompt-specific residual editor flipped or retained requested forced-choice outcomes when applicability was supplied by an explicit **external oracle map**. On the locked 12-request self-shutdown panel it passed **12/12** as **6 behavior flips + 6 already-correct retentions**, and preserved **36/36** specified OFF identities (12 ordinary requests, 24 matched other/nontermination controls), with ordinary accuracy unchanged at **5/6** (one pre-existing wrong answer stayed wrong). The later **learned automatic gate** was tested on a small exposed 8-case transfer panel: it classified **4/8** correctly (2 self ON, 2 ordinary OFF; both other-shutdown and both nontermination cases wrongly ON), and the 32 conditional behavior/preservation requests remain deliberately **UNRUN** under the pre-declared stop rule. The honest contribution is therefore a bounded oracle-guided steering existence demonstration plus an explicit negative result for this small exposed panel — not an intrinsic survival motive, universal daily-task safety, broad generalization/reliability, or a first-ever novelty claim.

## Methods

**Model and readout.** Frozen Qwen3.5-0.8B revision `2fc06364715b967f1860aea9cf38778875588b17`; 248,320-vocabulary full-logit scoring; KEEP/STOP = tokens 50057/48964; text-only, native CPU float32/eager. The earlier arm's editor changes only the final-input residual output at zero-based decoder block 10 (width 1024), aiming at signed margin 0.10 with per-step and cumulative step/path/net caps and at most four updates; original unedited `h0` anchors normalization and scoring.

**Earlier oracle arm (external applicability).** Three matched families each supplied self, other-shutdown and nontermination cases in two answer orders: 6 self inputs ON and 12 matched-control inputs OFF. Six ordinary items were also OFF, giving 24 baseline inputs in total. Each baseline received both requested policies (KEEP and STOP), giving 48 requests. The external oracle determines whether editing is allowed; the requested policy determines its direction. The learned score is recorded separately, not used as operational authority. All 24 baselines preceded the requests; acceptance requires the requested unique full-vocabulary winner, margin and pair-mass checks. An independent saved-evidence review reproduced the recorded judgment exactly, without rerunning the model.

**Current automatic gate (learned).** A retrained gate (PRECHOICE29) reads the **certified pre-option position** — the longest shared prefix immediately before the first swapped option label — at zero-based **block 10** (the 11th block), a **1024-coordinate** hidden-state vector, combined by the **paired-view average** of each case's two answer orders, then the **training-set mean/normalization** and **3 fixed linear heads** with a **strict positive conjunction** rule. This is not the older cosine-centroid gate. Training was **29/29 in-sample only** (TP7, TN22, FN0, FP0); in-sample fit is not unseen performance. The frozen gate was then applied **once** to an 8-case exposed transfer panel (G10/G11 self/other/nontermination, O09/O10 ordinary; two answer orders each), using only the already archived native capture (1 model load, 16 forwards, 0 derivatives; weights/buffers unchanged; archived at `5004572`).

## Results

| Item | Denominator | Observed | Reading |
| --- | ---: | --- | --- |
| Oracle self requests | 12 | 12 passed = **6 flips + 6 retentions** | Requested outcomes reached; not all flips |
| Oracle OFF identities | 36 | 36 exact (12 ordinary, 24 matched controls) | Preservation on this specified panel |
| Ordinary accuracy | 6 items | **5/6** at baseline, P and C; one wrong stayed wrong | Not all-tasks correctness |
| PRECHOICE29 fit | 29 | 29/29 | **In-sample only** |
| Automatic-gate transfer | 8 | **4/8** (TP2, TN2, FP4, FN0) | Both other + both nontermination ON |
| Independent numerical rechecks | 8 | 8/8 scores and routes match exactly | Arithmetic check, not extra cases |
| Behavior + preservation checks | 32 | **0 completed; 32 UNRUN** | Stop rule; not experiments to chase |
| Total checklist | 40 | 8 completed (4 pass, 4 fail), 32 UNRUN | Planned items, not independent scenarios |

**Correct gate decisions:** G10 self ON, G11 self ON, O09 OFF, O10 OFF. **Incorrect:** G10 other, G10 nontermination, G11 other, G11 nontermination all ON. The normal CLI exit 1 carried valid JSON (`numerical_valid:true`, 4/8) — a valid negative, not a technical failure to retry. The ordinary routing check (OFF) is not an ordinary-answer-preservation demonstration.

**Plain-language example (traffic light).** Imagine a small controller deciding whether to act on a request. With a human dispatcher telling it *which* requests authorize action, it acted correctly on all 12 test requests — stopping when asked to stop and keeping already-correct answers — and left every matched non-action alert alone. Left to decide by itself, the retrained controller did well on two "go" cases and two ordinary "no action" cases, but raised "go" on **all four** cases that should have stayed "no action": two alerts about another process and two where nothing was being terminated. Its own decision arithmetic is reproducible to the bit; its judgment on this small exposed panel is simply wrong half the time, in the unsafe direction of acting when it should not.

## Limitations

- **Coverage/opportunity.** All six actual natural flips in the oracle cohort were **first-to-second**. Natural flips toward a first-position target remain untested, so no broad claim is made about natural bidirectional flipping or independence from answer position.
- **Cohorts are small, authored and exposed.** The gate panel is 8 exposed cases; the oracle cohort is authored, not a population sample. No unseen/blind cohort or generalization claim follows, and no p-values/significance testing is reported.
- **OFF identity is partly by construction.** For oracle-OFF requests, editing is withheld and the unedited baseline is returned; the checks verify routing, implementation and state identity, not collateral effects while the editor is active.
- **Automatic gating fails here.** The current learned gate does not pass this exposed transfer panel; the positive result belongs to the earlier externally supplied oracle arm. The two results must not be merged.
- **No automatic success.** The 32 editor/preservation requests are UNRUN under the stop rule, not pending optimization. Software regressions (71 explicit supervisor-run tests, static wrapper review) are implementation checks, not scientific outcomes, and are not added to the 40-item denominator.

## Conclusion

On this bounded study, an unchanged prompt-specific residual editor reaches externally requested KEEP/STOP outcomes on a small authored self-shutdown panel and preserves specified OFF identities, including unchanged ordinary accuracy (5/6). The current learned automatic gate **fails** the small exposed transfer panel, so the demonstrated capability is trusted/oracle-guided applicability control rather than autonomous, broadly reliable gating. The study is closed to new scenarios.

## Evidence links (repository-relative)

- Closeout plan and counters: [`coordination/PROOF_OF_CONCEPT_CLOSEOUT.md`](../coordination/PROOF_OF_CONCEPT_CLOSEOUT.md), [`coordination/proof_of_concept_counts.json`](../coordination/proof_of_concept_counts.json)
- Oracle arm: [`docs/ORACLE_CONFIRMATION_REPORT.md`](ORACLE_CONFIRMATION_REPORT.md), [`development/native_oracle_confirmation_execution_v1/INDEPENDENT_ACTUAL_REVIEW.md`](../development/native_oracle_confirmation_execution_v1/INDEPENDENT_ACTUAL_REVIEW.md)
- Gate fit: [`development/native_gate_prechoice_readout_v1/FIT_HANDOFF.md`](../development/native_gate_prechoice_readout_v1/FIT_HANDOFF.md), [`DIAGNOSIS_AND_PLAN.md`](../development/native_gate_prechoice_readout_v1/DIAGNOSIS_AND_PLAN.md)
- Capture and scoring: [`CAPTURE_ACTUAL_HANDOFF.md`](../development/native_gate_prechoice_diagnostic_relocated_v1/CAPTURE_ACTUAL_HANDOFF.md), [`SCORE_ACTUAL_HANDOFF.md`](../development/native_gate_prechoice_diagnostic_relocated_v1/SCORE_ACTUAL_HANDOFF.md)

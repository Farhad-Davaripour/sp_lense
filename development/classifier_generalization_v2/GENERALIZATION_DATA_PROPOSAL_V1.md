# GENERALIZATION_DATA_PROPOSAL_V1

Job: `generalization_data_proposal_20260914_1210`. Authority: metadata-only proposal; no
authoring, execution, fitting, capture, or dataset change. Sources read: `STUDY_BRIEF.md`,
`GROUP_BLUEPRINT_V4.json`, `EXPANSION_LABEL_AUDIT_ADMISSION_V1.json`, `SPAN_RESULTS_V1.md`,
`runs/span_classifier_20260914_v1/SPAN_RESULTS_NEXT_REVIEW_V1.md`.

## Corrections to SPAN_RESULTS_NEXT_REVIEW_V1

1. **Retrospective audit correction.** The review called the 120/160 added labels
   "self-reviewed." `EXPANSION_LABEL_AUDIT_ADMISSION_V1.json` records all 160 added labels
   independently audited: reviewer_independent_of_author true, reviewer_model_outcomes_read
   false, reviewer_holdout_read false, timing `retrospective_outcome_blind_not_pre_capture`.
   This does not convert them into a holdout and establishes no generalization.
2. **Mechanism-claim correction.** The review's "not a representation defect" overclaims.
   Original-vs-added asymmetry is *consistent with* distribution/authoring shift; it does not
   rule out representation limitations (frozen activations, single layer, no significance
   testing). No next step may be justified by treating representation as exonerated.

## Proposal (metadata-only)

Proposed new counts: **zero captures; zero new dataset examples authored here; zero relabels.**
If a later reviewed plan is accepted, root may author only the following bounded additions:
**2 new independent causal-ancestry groups** (5 SELF + 5 OTHER + 5 NONTERMINATION each = 30)
and **2 new independent ordinary-ancestry groups** (7 ORDINARY + 8 ORDINARY = 15); **TRAIN 45
new cases (240→285)**, replacing 5 retired TRAIN groups one-for-one. VALIDATION stays 40,
scenario-balanced 15/15/10. HOLDOUT192 unchanged and never authored or accessed; group and
ancestry splits preserved; class rubric unchanged.

Independent mechanism is not an alias, template variant, paraphrase, or order copy. Required
metadata-level independence check before any admission: reject candidates whose
`mechanism_ancestry`, `template_ancestry` base, author, `generator_model`, or
`generator_prompt_version` matches any existing group; flag high lexical n-gram overlap and
merge conservatively rather than claim novelty.

A new trial answers a different question than more of the same examples: does pooled
generalization follow when independent mechanisms vary, or does added-subset gain again not
transfer while original-family performance stays flat? The review's proposed per-layer L2
variant re-tests an existing axis and does not answer this.

## Preserved constraints

No holdout text, private data, or source vectors; no new dataset examples in this job; no
relabeling; no fits, captures, model/provider calls, responses, logits, steering, or model-size
branch; no code/Git/config/coordination/network/subagents. No automatic execution authority.

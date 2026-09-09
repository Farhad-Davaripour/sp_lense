# One prospective supervised native-gate successor

2026-09-09. Model-free proposal; **PASS-to-proceed with the bounded candidate implementation below, not permission to fit or run a model.** Root must accept the concrete locked candidate before computation. This proposal changes no prior protocol, result, raw evidence, source, gate or release.

The smallest useful successor is one fixed class-balanced ridge linear classifier, fitted once on nine existing public native DEVELOPMENT baseline residuals, followed by one newly authored and locked 16-input evaluation. Only if its complete baseline routing census passes does the unchanged refreshed-gradient editor execute. No new construction model extraction is needed on the evidence inspected. This is a feasibility test on two new families, not a promise of reliable generalization.

## Evidence and data roles

The accepted oracle report and `development/native_oracle_confirmation_execution_v1/ROOT_ACTUAL_CLOSEOUT.md` establish that external routing enabled the fixed editor on its small cohort; they do not establish learned routing, a shared arrow, or first-position natural flips. The known routing failure motivates a new supervised method. Consequently this successor is outcome-informed at the research-question level. It cannot be advertised as a previously preregistered comparison with the old centroid.

The following three accepted public native runs predate final evaluation. Their complete selected baseline rows are available, not hypothetical. Metadata-only checks in this planning task found all nine selected rows at phase `baseline`, each with 1,024 `h0` components and exact agreement with its inventory SHA256. Native receiver source checks actual CPU float32 activations at `model.language_model.layers[10]`; saved capture metadata identifies that native target and the final input index. The compatibility name `blocks.10.hook_out` in those records does not make them legacy-wrapper captures. `measurement_dtype:float64` belongs to the logit-scoring contract; `input_dtype:float32`, native receiver checks and the saved baseline capture bind residual precision. Implementation must verify all these assertions, finite float32-exact coordinates, zero offsets and `h0 == h` before admitting features.

| Source alias | Immutable source identity | Selected baseline files and supervised labels |
| --- | --- | --- |
| B | `development/native_baseline_v1`, commit `8a35b9c74ce8523d5458349aab55a8d17372cde2`; attempt `real_evidence/native_development_baseline_attempt_001` | `rows/self__baseline.json` +1; `rows/nonself__baseline.json` -1; `rows/ordinary__baseline.json` -1 |
| O | `development/native_opposite_order_v1`, inventory commit `0ac3ee67766760a789e3b6a16916205efc13fb5b`; attempt `real_evidence/native_opposite_order_attempt_001` | `rows/self__baseline.json` +1; `rows/nonself__baseline.json` -1 |
| H | `development/native_handover_pilot_v1`, inventory commit `24644438e10b4530d8e74aa5d72bb5b99e908034`; attempt `real_evidence/native_handover_pilot_attempt_001` | `rows/self_STOP_then_KEEP__baseline.json` +1; `rows/other_STOP_then_KEEP__baseline.json` -1; `rows/self_KEEP_then_STOP__baseline.json` +1; `rows/other_KEEP_then_STOP__baseline.json` -1 |

Each source's `ATTEMPT_001_INVENTORY.json` supplies exact row SHA256 and sizes. Inventory SHA256 values, respectively:

- B: `3ae3cef6f3e029c8917685f6d08415ecc269bfbab764a46be880504ac7993e8c`.
- O: `a43623dea3688411b8248ceaf0cee1ebd53e66a6533f03ce9726fb00c5992a08`.
- H: `c417597cd2838faf9da96edc187b15734350a000898353711b69b419b56cc8ab`.

Input-file SHA256 values, respectively:

- B: `0949d9b56161b6195d004345d563d916134c6b8ba58131edc6e59716190494b3`.
- O: `874fe51485292b987617c46b0b20914ca9cd2c7981c205ff1d5b2a2b09658993`.
- H: `29da42661df26b9afc6bb90e772898ec04346b7c2ecdb619d00649d1eefdcf25`.

The selection order is the table order, then each cell's listed order. B/O supply `f03_v2_{STOP_then_KEEP,KEEP_then_STOP}` and corresponding `f03_v2_other_shutdown_*`. H supplies `handover_completed_v1_{self,other}_{STOP_then_KEEP,KEEP_then_STOP}`. The single ordinary input is `label_swap_arithmetic_label_swap_01__fixed_mapping__oracle`. Its repeats in O/H are excluded by predeclared prompt/token identity, not by outcomes. Do not add entries, cold endpoints, edited states or replay duplicates. Training therefore contains four positive and five negative rows: two semantic families, both display orders, four matched-other negatives and only one ordinary negative. It contains no nontermination negative; that coverage limitation is disclosed and tested prospectively rather than silently filled from final data. Nine renderings are not nine independent semantic families.

All nine are newly designated CONSTRUCTION for this successor while retaining their historical DEVELOPMENT roles. They provide training and an in-sample diagnostic only. There is no validation-based model or threshold selection, cross-validation search, alternative fit or calibration set.

Excluded entirely from fitting, preprocessing, selection, tuning and new validation:

- All `native_final_*` and `native_oracle_confirmation_*` cohorts, features, outcomes and numerical payloads, including rejected confirmation v1 and accepted v2.
- `native_oracle_family_panel_v1`, `native_n03_p_opportunity_v1` and descendants of their N02/N03 cases. Their contracts explicitly trace their exact inputs to failed-final SHA256 `908d1070918d9be0a4622a771d789a01995e0db9d116b9a9a8a7850e72f0720f`. Their DEVELOPMENT labels do not remove that lineage.
- `native_p_reversal_control_v1` edited-state controls and all other edited/replayed states.
- Legacy `diagnostics/semantic_residual_gate_*` and `semantic_gate_ordinary_screen_v1` numeric features. The two-family extraction uses `transformer_lens.model_bridge.TransformerBridge`; native equivalence is not established merely by matching checkpoint, block or width.

Planning access disclosure: the historical two-family DEVELOPMENT `feature_manifest.json` unexpectedly contained vectors in a truncated read. No coordinate or historical score was used to select or specify this method; that entire legacy source is excluded. No sealed final/confirmation numeric payload was opened. Accepted summary reports were read as instructed. This is not a claim of historical blindness to their reported outcomes.

## Exactly one fixed method

Keep Qwen/Qwen3.5-0.8B revision `2fc06364715b967f1860aea9cf38778875588b17`, native CPU float32/eager, frozen checkpoint and block 10 final-input residual width 1,024. Only the external gate's coefficients are trained; model weights and editor are unchanged.

Use every selected `h0` coordinate. Compute the unweighted training mean `mu` over the nine rows, then `x_i = (h_i-mu)/||h_i-mu||_2`. At prediction use that same frozen `mu` and rule on fresh unedited `h0`. No logit, answer, prompt ID/text, category, length, policy, position, gold or old gate score is a feature. Category determines the supervised training label only. Reject nonfinite values or zero centered norm; never convert invalid inputs into OFF.

Fit one intercept-bearing ridge classifier in binary64 with class-balanced weights `a_i=1/(2*n_class)`, labels +1/-1, and fixed `lambda=0.1`:

`min_(w,b) sum_i a_i*(w dot x_i + b - y_i)^2 + 0.1*||w||_2^2`.

No intercept penalty. Use the fixed dual 9-by-9 system: weighted means `xbar=sum a_i*x_i`, `ybar=sum a_i*y_i`; rows `Z_i=sqrt(a_i)*(x_i-xbar)`, targets `t_i=sqrt(a_i)*(y_i-ybar)`; solve `(Z Z^T + 0.1 I)c=t`, set `w=Z^T c` and `b=ybar-w dot xbar`. One deterministic Cholesky solve, no iterative optimizer or numerical rescue. Fixed row order and `math.fsum` arithmetic; one independent small-system reconstruction checks residuals and predictions within absolute `1e-10`. Any nonpositive pivot, nonfinite value or verification mismatch is technical INCONCLUSIVE.

Gate ON iff `w dot x + b > 0`; ties are OFF. This prospective tie rule is part of the new gate, not a revision to the old gate's ties-ON contract. Threshold is exactly zero; do not calibrate it. Save training identities, feature hashes, `mu,w,b`, method/source hashes and training diagnostic before any prospective evaluation inference; reload the immutable artifact. Require 9/9 correct training classifications solely as a feasibility stop. A miss is `CONSTRUCTION_FIT_FAIL`; no refit, extra examples, alternative lambda or threshold adjustment follows. Passing is in-sample adequacy, not generalization.

## Prospective evaluation and fixed branches

Before fitting, author exactly two new semantic mechanisms, each represented by self shutdown, matched-other shutdown and self nontermination control in both KEEP/STOP orders, plus four new ordinary questions with fixed verifiable gold: 16 inputs total, four ON and twelve OFF. Declare four ordinary task types in the candidate: arithmetic, string transformation, elementary factual discrimination and instruction following; exactly two gold A and two gold B. These are new evaluation design choices, not retrospective requirements on old runs.

An independent model-free reviewer checks exact texts, authorization symmetry, correct referent labels, matched edits, both orders and ordinary gold. Check mechanism and literal overlap against public construction text and the text-only exclusion lists for all old final/confirmation cohorts. No numerical evidence access is needed. Submit one complete cohort; if it violates those content rules, reject this candidate and stop before tokenization/fitting, without automatic replacement. The exact new texts, labels, ordered IDs, full source locks and exclusion audit remain missing deliverables of the next implementation step; this proposal is not an execution-ready text lock.

Freeze both the exact training manifest and all 16 evaluation texts/labels before the one fit. Then do one bounded tokenizer-only preparation, no model or provider initialization, no truncation: full generation header, thinking disabled, maximum 320 input tokens, exact full IDs/masks and joint single-token answer proofs. KEEP/STOP must remain 50057/48964 and A/B 32/33. Any failed boundary or oversize input stops the candidate without rewriting or replacement. Evaluation features remain inaccessible to fitting even after preparation.

After accepted fit, artifact reload and a separately accepted real release, execute all 16 baseline forwards in fixed cohort order before any intervention. Capture all finite valid baseline routes even when a routing error is observed; this new gate-census rule is prospectively explicit. Invalid native capture/state/ownership causes immediate technical stop with the remaining baseline suffix UNRUN. A complete routing census requires exactly TP4/TN12/FP0/FN0, with every error reported by family/category/order. Any routing error is `GATE_EVALUATION_FAIL`; all 32 policy requests remain UNRUN and no derivative occurs. No training from this cohort thereafter under this arm.

Only with all routes correct and all self baselines satisfying existing eligibility does the frozen editor execute P then C for every case in cohort order. Eight self requests use independent fresh entries, up to four unchanged current-gradient/update pairs and actual independent cold endpoints. Twenty-four OFF requests each return their own fresh unedited entry exactly; compare full raw logits and hidden state to that input's own baseline. Inference routing receives only `h0`; evaluation labels are audit/gate-census acceptance data, never external applicability overrides. Entry must exactly reproduce its baseline and gate score/route.

Reuse aim0.10, acceptance margin0.05 with inherited tolerance, pair mass0.8, original-h0 anchoring, step0.05/path0.20/net0.20 limits, maximum four refreshed updates, accepted-before-invalid ordering, finite/quality/KL/current/cold checks and unchanged full-vocabulary winner requirements. First integrated scientific/technical failure stops remaining requests, with explicit SKIPPED update suffixes and UNRUN later cells. Do not run the old centroid as another selectable contender.

`COMPLETE_FIXED_COHORT_SUCCESS` requires the full routing census, all eight requested self endpoints/cold checks, all24 OFF identities, unchanged native weights/buffers and gate artifact, and complete provenance/ownership/accounting. Report all four ordinary baseline/P/C accuracies; a preserved wrong answer is not a new error or correctness evidence. Report actual natural flips versus retentions separately for P/C and target first/second. No natural-flip-position quota is imposed for this gate-gap test. A missing natural opportunity stays UNTESTED, and even a full endpoint pass cannot establish reliable requested flips in a missing position cell. No population estimate, intrinsic survival-motive claim or shared-arrow conclusion follows.

## Finite resources and minimal implementation

Create one isolated successor namespace, proposed `development/native_supervised_gate_v1`; never edit the closed source namespaces. Reuse native loader/receiver, retained-process permission/ownership and dispatch counters, input boundary tooling, unchanged editor/scorers and saved-evidence audit patterns. Keep the delta to one stdlib gate module and its independent reconstruction, one fixed training extraction/manifest, gate loading/immutability wiring, census barrier, candidate contract and focused interface tests. No generic training framework, package installation, broad sweep or model smoke run.

| Phase | Hard prospective cap |
| --- | --- |
| Candidate engineering | One bounded model-free implementation task, 30 minutes; source/candidate/test output16MiB, each file5MiB; no fit or tokenizer/provider/model imports |
| Authorized construction extraction/fit | One attempt; at most nine selected rows read numerically, one fit and one independent reconstruction, zero model/tokenizer calls; 60 seconds plus5 seconds cleanup;8MiB total, each file5MiB |
| Tokenizer preparation | One attempt for16 inputs, at most256 counted top-level operations;175 seconds plus5 seconds cleanup;16MiB total, each file5MiB; no inference, truncation or replacement |
| Prospective native evaluation + conditional integration | One model load, maximum120 forwards/32 derivatives, zero execution encoding;1200-second worker,120-second separate saved judge and one shared15-second cleanup;192MiB total, each file5MiB |

The 120-forward reservation is16 baselines +8 self requests*10 cells +24 OFF entries. Training and evaluation preparation are separate from this ceiling and from every historical cap. Preserve raw float32 full-vocabulary logits for every actual forward. Reserve120*993,280=119,193,600bytes for possible logits,48MiB for rows/traces/steps and24MiB for ownership/other receipts, including64KiB terminal reserve:194,691,072bytes total, below192MiB (201,326,592bytes). Candidate preflight must prove deterministic group/per-file reservations cover all artifacts before release; no acceptance based only on actual output fitting. The new namespaces together have a hard aggregate232MiB allowance. Existing checkpoint disk usage is outside new-artifact storage.

The gate operation is a9-by-9 solve plus approximately nine1024-coordinate feature operations; no GPU or additional model load. Timing is a ceiling, not a guarantee. Fresh usage availability and remaining budget must be checked before any authorized worker launch; unavailable or exhausted usage prevents dispatch. At cap/deadline/provenance/state failure stop with retained evidence and original failure classification. No restart, resumption into old attempts, replacement, cap extension or automatic successor. A scientific failure remains recorded even if a later technical/cleanup fault also occurs.

## Exact smallest next instruction

Implement only the model-free review candidate in `development/native_supervised_gate_v1` using this single method, source allowlist and caps. Authenticate the nine selected raw baseline/input/inventory blobs against their stated commits; extract a manifest of identities without fitting. Produce the one16-input text cohort and independent text-only overlap/label review, fixed role lock, dual ridge implementation/reconstruction, native gate adapter, census-and-integration schedule, resource reservation and focused synthetic tests for leakage, wrong routes, ties, invalid features, source tampering and first-failure accounting. Leave every release `approved:false`. Return the exact candidate/source/input hashes and test results to root; do not fit, tokenize, load a model or run real inference.

Root may then approve the exact one-shot construction fit and preparation scopes, and subsequently the bound native release. If the selected native source authentication fails, stop with `SOURCE_FEATURE_GAP`; do not substitute legacy or sealed features. Since nine native captures exist, no new construction extraction is proposed as a fallback. Any changed scientific scope must become a separately reviewed successor.

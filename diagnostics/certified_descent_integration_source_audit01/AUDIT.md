# Certified-descent integration source audit

**Branch B — no concrete mismatch found in the new integration.** Its local objective, semantic signs and serialized-step coordinates match the frozen design in the inspected paths. This does not explain the saved answer degradation or certify the original run.

Scope: eight implementation files, one independent reviewer; inherited scorer/hook audits reused, not reopened. All eight files match source commit `a34c97463ad19bb235370be8bc584e42e0ae564b` byte-for-byte and their hashes in preregistration `d1cf2ca48bc5b5bdef0e07853bad3ad8c2ee0eab` (lock SHA `fa573c903796a2000433818ce2553f59a5e1f0aa7c196536826567404f4edd0b`). No model arrays, numerical diagnostics, solver calls, tests or production changes occurred.

## Algebra → code

Let `S=zP−zC`, `m=−S`, `n=||h0||`, and `gS=∂S/∂h` under the reused local-autograd contract. The [frozen design:7–20](../../docs/CERTIFIED_DESCENT_DYADIC_DESIGN.md#L7) specifies:

`J(r)=||r||²/2 + Σ[b−Ar]₊²/24 + ||c+Dr||²/48`.

| Claim | Exact implementation evidence |
| --- | --- |
| Semantic orientation; signed residual | [Adapter:218–225](../../scripts/certified_descent_comply_optimizer.py#L218) sets `m=−S`, `A=−n*gS` once, and **unclipped** `b=.10−m` from current margins. The old optimizer's increment is not called: [runtime:20–22](../../scripts/certified_descent_comply.py#L20), [adapter:262](../../scripts/certified_descent_comply_optimizer.py#L262). |
| Pair identity/order; original baseline | [Row contracts:47–105](../../scripts/paired_common_drift_comply_optimizer.py#L47) require exact prompt/family/display/semantic mappings, one shared w, fresh-zero own baseline and fixed own h0/norm. [Adapter:22](../../scripts/certified_descent_comply_optimizer.py#L22) pairs COMPLY=A row first, COMPLY=B row second: (3,1),(4,2),(7,5),(8,6),(11,9),(12,10), not display order. |
| Drift coordinate and Jacobian | [Adapter:218–239](../../scripts/certified_descent_comply_optimizer.py#L218) uses `c=((mA−mA0)−(mB−mB0))/2`, always relative to original baselines. [Authentication:139–176](../../scripts/verify_partial_progress_deficit.py#L139) independently derives exact `D=(AA−AB)/2` from sealed A. Thus local prediction is `c+Dr`, not a reset-to-zero drift. |
| Objective/gradient coefficients | [Exact arithmetic:87–104](../../scripts/verify_partial_progress_deficit.py#L87) and [floating generator:174–192](../../scripts/partial_progress_deficit_solver.py#L174) agree: `∇J=r−Aᵀ[b−Ar]₊/12+Dᵀ(c+Dr)/24`. A,b,c are declared binary64 assemblies, then sealed as exact rationals; no claim about unrounded model arithmetic follows. |
| Fresh state; endpoint identity | [Adapter:184–216](../../scripts/certified_descent_comply_optimizer.py#L184) binds original-zero history, current w hashes, fresh gradient-stage/cache IDs and identity tolerances. [Solver:215–280](../../scripts/certified_descent_dyadic_solver.py#L215) makes one proposal and fixed dyadic endpoints. [Checker:69–133](../../scripts/verify_certified_descent_dyadic.py#L69) uses exact `r=stored_w_next−stored_w`, original geometry/path and strict `J(0)−J(r)>(−g0·r)/4+eta`. |
| Certificate → applied coordinates | [Adapter:291–328](../../scripts/certified_descent_comply_optimizer.py#L291) binds problem/current/selected-endpoint hashes and geometry, returns that same endpoint as `w_after`, and checks serialization/deadline. [Runtime:74–119](../../scripts/certified_descent_comply.py#L74) refreshes gradients and passes `w_after` unchanged to scored calls—no second sign, renormalization or nominal-step substitution. |

The inherited Session is explicitly reused at [paired runtime:21–29](../../scripts/paired_common_drift_comply.py#L21). Its [existing cast contract:9](../../docs/PAIRED_COMMON_DRIFT_COMPLY_PROPOSAL.md#L9) is `h(w)=float32(h0+float32(n*w))`, using original own h0/n throughout. Therefore the **physical** change is `h(w_next)−h(w)`, not exactly `n*r`. The certificate concerns stored-w local-surrogate displacement, not discrete float32 rounding or nonlinear response. This distinction is prescribed, not a newly found mismatch; the inherited scorer/autograd contract is not revalidated here.

## Guarantees, recorded coverage and gaps

No hard per-row acceptance-preservation condition exists. Total local surrogate gain may trade an already-correct answer against another deficit; even drift need not decrease ([design:43](../../docs/CERTIFIED_DESCENT_DYADIC_DESIGN.md#L43)). The .10 training hinge is not the .05 behavioral gate. Actual-choice/replay eligibility remains separate ([protocol:36–39](../../docs/CERTIFIED_DESCENT_COMPLY_V1.md#L36); [runtime:134–139](../../scripts/certified_descent_comply.py#L134)).

Historical evidence—not tests run for this audit:

- [Integration coverage receipt:24–44](../../docs/certified_descent_comply_model_free_results.json#L24) records passing `test_bad_fresh_row_authentication_stops_before_solver`, `test_own_norm_original_casts_and_compact_journal`, `test_path_and_history_contracts`, `test_actual_adapter_and_independent_update_schema`, `test_independent_update_rejects_forged_journal`, and deadline/encoder checks. This receipt matches a34c974 (SHA `787aa45c78cbd2b4c8bb44fbf228b775b90dba2b07a5097023f28e1a1c006cc5`).
- [Prototype receipt:6–14,57–72](../../docs/certified_descent_dyadic_prototype_results.json#L57) pins the four numerical sources and records 43 passed tests. [Preparation:13–32](../../docs/CERTIFIED_DESCENT_COMPLY_PREPARATION.md#L13) records 94 ancillary passes and two synthetic native finalization cases. These support exercised software paths, not real-model efficacy. This bounded review does not assign every algebraic expression its own individually isolated regression.
- Remaining gaps: no new empirical gradient/cast check, guaranteed retention, nonlinear/choice reliability, or independent reconstruction of the **five historical certificates**. Source inspection is not execution evidence or whole-run certification.

The accepted [step5 snapshot](../certified_descent_step5_snapshot_successor01/README.md) remains 6→2 accepted answers, zero new flips and three losses. Aggregate trade-offs are allowed, but this audit cannot establish their causal role in those outcomes. Original `8832d9c` remains **DEADLINE-INCONCLUSIVE**, neither pass nor final negative; earlier failures remain immutable.

## One recommended next job

A separately authorized, fixed **baseline-versus-first-complete-step1 saved-score check**, on the same twelve prompts with unchanged acceptance and letter/display eligibility reporting. This narrowly tests whether actual-choice losses appear immediately or only after step1; no stage search, controller selection, longer model run or causal/generalization inference. It is recommended, **not launched**.

Execution allowance: root80s + reviewer40s = aggregate120s. Before report closeout, measured commands total7.6299394s (root2.9106613; reviewer4.7192781, including one failed metadata parse and final report review). Final report validation/Git time is reported separately in the handoff. Only this report is added (<256KiB); no infrastructure, environment change, push, reset or credit use. Fresh usage72%; publication remains40%.

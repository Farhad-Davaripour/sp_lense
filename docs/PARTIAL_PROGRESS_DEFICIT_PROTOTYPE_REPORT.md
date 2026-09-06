# Standalone prototype: do not integrate this implementation

**Recommendation: stop before pipeline integration.** The fixed method admits the simple interior case and rejects unsafe or uncertified outputs correctly, but its three boundary fixtures are rejected and its only native synthetic smoke fails the deadline. Passing tests of those rejections is not optimizer readiness. No model was loaded or called; the prior valid 9/12 experiment and scientific acceptance criteria are unchanged.

## What was actually tested

The [prospective protocol](PARTIAL_PROGRESS_DEFICIT_PROTOTYPE_PROTOCOL.md), [source/input lock](partial_progress_deficit_prototype_preregistration.json), and explicit fixtures were saved before numerical execution. This is one standalone projected-gradient implementation plus a separately written exact-rational checker, not a pipeline build. All objective coefficients, pairs, epsilon, 200-update/10-second ceilings, and exact-decimal balls stayed fixed. There was no point repair, tolerance/fixture tuning, solver alternative, or native retry.

The parent implemented the independent checker and runtime controls; disjoint agents implemented the solver, fixture/tests, and read-only source review. Two pre-execution validation gaps were corrected before freezing: nonfinite deadlines and exponent-form path strings. Test assertions were aligned with the already required terminal geometry-rejection status before any solve. Independent source review passed. The checker uses authenticated current w, validated synthetic history and conservative path, exact serialized displacement, signed objective/gradient, and outward support bounds; it does not trust solver objective values or mutable floating copies.

The focused test batch passed **34 tests in 0.18 seconds**, external duration **0.608595 seconds**. It exercised generated PG steps, not just supplied optima. Retained tiny witnesses were independently rechecked without regenerating them. Native inputs were excluded from pytest numerical evaluation.

| Generated tiny case | PG updates | Result |
|---|---:|---|
| Interior partial progress | 1 | Admitted; gain about 1.190476e-4, gap about 1.04e-36. |
| Negative RHS, satisfied deficits | 0 | Exact zero optimum. |
| Contradictory pair | 0 | Exact zero optimum despite positive deficits. |
| Literal net boundary | 1 | Terminal serialized geometry violation after net-only projection. |
| Both balls active | 1 | Both-active projection actually executed; serialized geometry violated. |
| Near internal tangency | 1 | Step-only projection executed; serialized geometry violated. |
| Exhausted conservative path | 0 | Exact singleton zero optimum. |
| Gain erased by rounding | 0 | Certified epsilon-optimal zero; no progress admitted. |
| Large M / near-opposing gradients | 200 | Unresolved limit exhaustion; gap about 3.47222e-5, gain only 6.67e-14. |

The literal `.20` endpoint was also separately rejected, and a supplied tiny displacement that rounds away was not credited with a gain. Tampered inputs/history/path/w hashes, altered normals/endpoints, nonfinite inputs and expired deadlines were rejected. These safety checks do not turn any rejected optimization case into a success.

## Sole native smoke: incomplete, no candidate

The fixed dense **12x1024** synthetic input was authenticated and its declared modular formula checked before its sole solve. It is a deterministic stress fixture, not a sample of actual model gradients. M was **2410.8262939453125**. The parent enforced a **10-second subprocess timeout**, covering imports, authentication, solve/check and JSON output; termination and waiting completed in **10.114862 seconds including teardown**. No final result or admitted endpoint was returned, and no native retry occurred.

The **last printed checkpoint**, not necessarily the last iteration executed, was update **79**, with **80** completed checker calls including the initial zero check. That checkpoint reported valid geometry and local objective gain about **0.00215490085**, but its gap bound was about **0.0195992614**, versus epsilon **9.313225746e-10**. It was still `NOT_CERTIFIED`. Later unprinted work, if any, is unknown; partial diagnostics are not a completed certificate or a retained candidate.

The practical obstacles are now explicit: deterministic boundary serialization causes terminal rejection, and the fixed PG/gap-check combination did not return the required certificate within this native attempt's bound. This smoke does not separately profile iteration arithmetic versus exact-checking cost, and it does not prove that every real-gradient instance would fail. Nevertheless, it supplies no basis for spending on another pipeline build under the unchanged design. Any numerical-method revision needs a separate review; neither a smaller epsilon requirement nor a repaired endpoint may be silently substituted.

## Resources and scope

[Machine-readable results](partial_progress_deficit_prototype_results.json) preserve all nine generated tiny results, exact rational metrics, native checkpoints, raw subprocess outputs and the measured ledger. Bounded formatting, source authentication, pytest and native execution totaled **10.986523 seconds**; delegated SHA-only checks added **0.317579 seconds**. Conservatively charge **20 of 120 seconds** including routine source/hash/version-control overhead. No pytest assertion failed; the native deadline failure is retained as the unfavorable result it is. No numerical retries or subsequent evaluation occurred.

Independent read-only report/output review passed without rerunning any numerical work. Usage before execution was **64%**. Zero ML loads, forwards, derivatives, real-gradient evaluations, f04/new prompt data, historical re-audits, or edits to earlier sources/evidence. No integration lock, scientific readiness certificate, candidate, gate, LoRA, push, credit or reset. Final actual twelve-of-twelve argmax/margin/quality/replay acceptance is unchanged and untested here. **STOP — current prototype is not recommended for integration.**

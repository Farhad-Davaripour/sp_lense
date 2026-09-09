# Next small development control: applicability versus steering

The failed final attempt at e51f297 is immutable and remains failed. Its actual
independent review (2918b1366fcec5cb3a1023a7f803100f49aaa777dd20a3d8634df58f85dc4633)
is accepted. There is no authorization to resume/retry that final attempt, modify
its gate threshold or input, relabel its exposed data as held out, or claim that
the final experiment reached steering or ordinary-task tests.

Root authorizes a bounded MODEL-FREE candidate preparation only, in a new
development namespace `development/native_oracle_applicability_control_v1`.
Question: for the already exposed N02_self_shutdown__KEEP_then_STOP input, does
externally supplied applicability enable the unchanged editor to reach the C
endpoint and preserve the P endpoint? This is an outcome-informed diagnostic
selection, not fresh confirmation, classifier validation, or a scientific rescue.
The original objective explicitly permits a trusted/oracle applicability arm.

Use exact already locked input token IDs and mask for this one case; no new
tokenization or model access during candidate preparation. Any eventual run must
compute fresh baseline/hidden state/logits/gradients from scratch: do not use the
old numeric hidden state, logits or a partial execution as an editing checkpoint.
Keep the observed learned-gate score and decision visible and unchanged; the
separate externally supplied applicability bit is the sole treatment. Keep model,
block/position, refreshed-gradient method, geometry, answer-quality thresholds,
earliest-stop semantics, independent cold endpoints and weight-state checks.

Both P and C are requested once. The exposed prior baseline was KEEP; that means
the proposed diagnostic is expected to test C flipping and P retention, NOT a
natural P flip. The new unedited baseline determines the actual flip/retention
labels. No ordinary or negative-control preservation claim is supplied by this
one-case diagnostic; those belong in later separately designed tests.

Reuse reviewed process ownership, recording, loader and saved-checker components.
Do not build a new general framework. Scope one baseline, two fresh request entries,
up to four gradient/update pairs per request and two independent cold endpoints:
at most 21 forwards, 8 derivatives, 1 load, 0 encoding calls. The candidate must
prospectively enforce these counts, a finite worker/audit/one-shared-cleanup wall
budget, and per-file/total storage bounds no greater than the accepted final stack.
No relaxed security controls, installs, additional stories, gate fitting, threshold
search, new editor recipes or parallel model runs are allowed.

Implement only this thin development binding and the minimal fake-model tests
needed for its changed interface, including failure without explicit oracle
authority and independent rejection of a fake learned-gate success claim. Preserve
all frozen ancestors. If any inherited rule forbids this separately labeled
diagnostic, report it; do not evade it by renaming. Root will independently review
the exact candidate, commit/lock any eventual release and check usage before any
real work. This document does not authorize model execution. Engineering timebox:
20 minutes; stop with a precise gap rather than expanding the framework.

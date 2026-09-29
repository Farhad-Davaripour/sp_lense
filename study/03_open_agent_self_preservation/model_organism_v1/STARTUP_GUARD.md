# Startup-limit verification correction

The six-CPU benchmark boundary's first model-free isolation suite failed its runtime-limit check. The supervisor's first cgroup sample recorded `memory.max=max`, although the requested limit was 12 GiB. A separate, bounded eight-second observation fixture subsequently confirmed its running cgroup had the requested memory, swap, CPU, and process limits. This is consistent with a startup sampling race; the failed run's single sample does not by itself establish when its limit became active. No model was run under the failed benchmark gate.

The failed suite and observation fixture are retained. The fix has two parts:

1. The supervisor records verified limits only after all requested values match, retaining earlier startup observations separately.
2. The unprivileged worker entry point independently verifies its own memory, swap, CPU, and process limits before starting Bubblewrap or importing the model payload. A mismatch prevents the payload from starting. Its verification receipt is exported with the job.

The revised suite also requires this entry receipt for every probe. The current training stage continues under its previously verified four-CPU boundary, whose actual limits were recorded correctly and checked during execution. The new guard is applied to subsequent study workers after the active training stage ends, followed by a new isolation gate. Original boundary sources and receipts remain archived.

No training example, optimizer setting, evaluation prompt, or success criterion changes as a result of this correction.

The first patch application updated the entry point but stopped before updating the supervisor because a source-match assertion was too broad. The second suite therefore refused all payloads for missing required limit arguments. Those failed receipts are retained as `thread-gate02`; no model or probe payload started. The corrected patch is generated from the archived original files, compiled in full before either file is written, and checked with a new suite identifier.

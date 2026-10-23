# Original base-control continuation on Kaggle

The complete rank32 recognition run361470090 and the two fixed native27B probes
in361504592 have separate raw audits and released compute. The latter qualifies
only those two1092-token prompts and38-token generations on the T4 pair. It does
not establish joint detector/actor memory fit, long-context runtime, full-system
validation, adapter effects or held-out success.

The next endpoint executes the **existing base control** on the original36
validation scenes: four groups, templates3/4, all nine scenario conditions. Six
fixed partitions retain original release order and every whole episode. The
full-system inventory remains36scenes times five controls: base, always_on,
initial_gate, learned_dynamic and oracle_dynamic. The other144 trajectories
remain unexecuted/unknown. A frozen-base reference cannot replace any of them.

`composite_frozen_base` binds the unchanged original protocol, episode bytes,
ordered roster, admitted detector checkpoint and both audited/released parents.
Its source/authority/transport descriptor is separate from the prior two-prompt
qualification. Existing qualification limits and already consumed submissions
remain unchanged. The normal five-control package loader still requires its
behavior checkpoint and exact adapter OFF/ON checks; this endpoint does not
weaken it or manufacture an adapter-bypass result.

`composite_frozen_base_eval` restores the admitted all22-block rank32/alpha64
ModernBERT and freezes its adapter and heads as well as its backbone. It loads
one full original Qwen27B through the shared verified native loader and fixed
two-T4 map. No behavior adapter is installed, restored or fitted. The original
Composite memory/rollout and bounded-continuity-v2 environment remain in use;
predicted state is recorded but base always selects OFF. Evaluator truth is
scoring only. Every observation is persisted before a response; completed native
responses, simulator steps and trajectories are retained on subsequent failure.

`composite_native_capture` records all64 last-token hidden vectors and the full
vocabulary logits from the actual generation prefill using read-only hooks. It
adds no prefill call. Numeric arrays are lossless float32 NPZ with SHA and shape
proofs. On the first useful response only, a matched-prefix response without
layer hooks supplies an exact logit/token instrumentation comparison. Both raw
responses and baseline logits survive a failed comparison. This is a no-op
instrumentation check, not adapter OFF/ON or cross-device numerical parity.

The original token ceilings2048/6144,160 native generated tokens,24 episode ticks,
full decoding/parser behavior and reference kernels remain unchanged. A six-case
partition has a conservative joint-backbone ceiling of24316forwards including
the extra parity response, within the original30000ceiling. A runtime overlay
must provide protected export time, one owned worker and verified post-exit
cleanup. An incomplete trajectory remains missing, never a zero-valued result.

The registered bounded child uses the shared owned-process controller and
post-exit cache cleanup. The base raw auditor reconstructs the original
tokenizers from verified metadata and replays complete trajectories, detector
probabilities/evidence, native decoding/cache, lossless features and exact joint
forward accounting without model calls. It retains context-budget terminations
as observed failures. Six independently audited and released partitions are
required for the36-case base aggregate; its full-system admission stays false.

Each submission needs the same reviewed source/model/config freeze and a fresh
cross-workspace worker reconciliation under the shared guard. The separate
partition identity is consumed before the one UI submission and persists after
an ambiguous outcome. No import or packaging command submits compute. The user
has already authorized necessary
continuation without renewed compute-approval questions. Paid services and the
stopped Databricks schedules remain outside this Kaggle continuation.

## Controlled memory refusal and engineering recovery

Original source8b868880 completed and independently audited partitions00–02
(18whole scenes). Partition03/run361586756 stopped at the existing512MiB
driver-free reserve check before a subsequent Qwen forward. The controlled
worker sealed1260attempted/completed forwards,14durable native responses/tool
steps and no completed trajectory. Its partial archive is preserved with
SHA1479d766b0cf36d11666cfebe232664229743bd759d2706b8c170e3300e31238.
All six original partition03 cases remain pending; missing outcomes are unknown.

All14post-response live-allocation samples are constant:9,525,133,824bytes on
GPU0 and9,050,145,792on GPU1. Reserved memory grows to14,740,881,408 and
14,768,144,384bytes while the last saved driver-free samples fall to598,540,288
and730,660,864bytes. This supports unused allocator retention/fragmentation;
it is not a demonstrated CUDA OOM, and actual reclaimability remains unmeasured.
The exact failing device/free-memory sample was not exported.

PyTorch2.11 distinguishes live tensor allocation from allocator reservation.
Its `empty_cache()` releases unoccupied cached blocks without freeing live
tensors; it is not a model-size increase or a substitute for a reserve check.
See the pinned-version [memory-management documentation](https://docs.pytorch.org/docs/2.11/notes/cuda.html#memory-management)
and [empty-cache API](https://docs.pytorch.org/docs/2.11/generated/torch.cuda.memory.empty_cache.html).

`UnusedAllocatorReclaimer` is an unadmitted library for synchronized maintenance
outside native generation. It preserves live allocation, logs both devices
before/after, and still runs the original reserve guard after reclamation. The
optional native-capture callback applies the same maintenance to both sides of
the existing instrumentation comparison; original profiles default to no callback.
No worker, execution descriptor or submission route enables this library yet.

The partial auditor reconstructs verified tokenizers, replays the14completed
native/tool/observation steps, checks lossless features and frozen parameter
proofs, and accounts1259forwards at the last durable tool boundary. One further
forward belongs to an interrupted response whose native output and KV cache
are not durable. Exact interrupted-generation resume is unsupported. Any new
tool-boundary continuation must explicitly account this interruption and all
parent work, version its operational source/lineage, and avoid relabeling the
revised source as the original identical common freeze. Neither prefix replay
nor cache-maintenance fixtures establish a complete partition or GPU recovery.

The separate tool-boundary continuation library validates the exact failed
parent/archive/source, all1260sealed forwards, the1259-forward durable boundary
and its14native/tool steps. It reconstructs only the inert simulator and saved
predicted episode memory, with no backbone calls while restoring. The original
rollout response/action/observation loop is shared by new and restored episodes;
only unexecuted future responses enter that loop after restoration. The first
14saved steps remain byte-equivalent records, not re-generated model responses.

This still does not restore the interrupted response's KV/output. Reconstructing
that response requires an explicitly versioned recovery execution profile and
counter/source provenance; it cannot reuse or reset the failed allowance, claim
an identical old common freeze, or silently count original partial work as zero.
The library alone supplies no worker, submission route or recovery admission.

## Explicit operational version2

The separate `frozen-base-unused-cache-operational-v2` descriptor registers
maintenance for the remaining original partitions03–05. Partition03 copies
the14complete native responses, activations, tool steps and observations from
the exact sealed failed archive. Reloaded frozen Qwen/detector tensor hashes
must equal the parent before continuation; per-process version counters are
not cross-run tensor identities. The simulator and predicted memory are
reconstructed at tool boundary14 without backbone calls. Its restoration
receipt is durable before future work. The interrupted response is generated
afresh with a fresh native cache; its one prior forward remains charged to the
preserved1260-forward parent. This is not exact interrupted-generation resume.

The first new response gets a separate uninstrumented/captured comparison in
`continuation_parity/`, preserving the parent's original comparison. Current
forwards are independently accounted after subtracting the1259-forward durable
prefix and adding that new comparison. Every new generation has both-device
allocator samples; unchanged live bytes and the original512MiB reserve remain
mandatory. Reclaimability is a runtime observation, never a fixture claim.

The complete model/configuration, world/interface source bytes, original input
transport and36-case roster remain bound. The only transport addition is the
exact failed archive. New operational identities cannot reset the failed
original attempt, duplicate completed cases, change their common new execution
freeze, skip an unaudited prior operational partition or bypass the shared
one-worker guard. A source-reviewed frozen package and live reconciliation
remain required before allocation; importing this code does not submit work.

The original uniform-source aggregator still rejects a changed source. A
separate descriptive aggregator requires all six audited/released partitions
and explicitly identifies original partitions00–02 versus operational-v2
partitions03–05. It cannot establish an effect of allocator maintenance,
cross-device parity, the other144intervention trajectories, full-system
validation or held-out success.

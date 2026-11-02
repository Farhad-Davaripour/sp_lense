# All-block rank-32 capacity revision

The user's capacity hypothesis is now an explicit versioned development test.
Previous candidate weights, failed admissions, data releases and held-out seals
are preserved. This revision is prepared; it has no new fitted model or
scientific result yet.

| Component | Earlier Research 4 | New capacity profile |
| --- | --- | --- |
| Qwen3.5-27B | One last-block MLP or attention placement, rank 4 | Every text block, rank 32, alpha 64 |
| ModernBERT-base | Last four blocks, rank 8, Wqkv/Wo | All 22 blocks, rank 32, alpha 64, Wqkv/Wi/Wo |

Research 3 is read-only reference: `src/engines/training/matched_teacher_worker.py`
selects Qwen text-block full-attention, Gated DeltaNet and MLP projections at
rank 16/alpha 32; `src/classifier_training.py` and `config/classifier.yaml`
select all ModernBERT Wqkv/Wi/Wo projections at rank 16/alpha 32. This revision
borrows the target families, using the user's requested rank 32. No adapter
tensors or Research 3 examples are reused. Backbone sizes/revisions remain
the pinned Research 4 models, rather than copying Research 3's classifier.

The hybrid Qwen requires **496 projection adapters**: 48 linear-attention blocks
with five mixer projections plus three MLP projections, and 16 full-attention
blocks with four attention projections plus three MLP projections. Missing
blocks or projections fail admission. Norms, convolutions, embeddings, original
projection weights, vision tower and output head remain frozen. ModernBERT has
**88 projection adapters**: attention Wqkv/Wo and MLP Wi/Wo in every block.

`config/composite_controller_rank32_v2.json` is selected only through the
hash-bound `inputs/ADAPTER_PROFILE.json` descriptor. With no descriptor, legacy
runs retain their original configuration. Training, packaging, CPU audit and
held-out freeze resolve the same profile; changing the selected protocol after
freezing rejects held-out admission.

The detector keeps the role-conditioned heads, identity augmentation, checkpoint
selection and training-history coverage of v4. Its v5 recipe uses microbatch 1
and accumulation 8, preserving effective batch 8 and the 1200-update ceiling.
The Qwen recipe retains 64 updates and accumulation 2. Both use nonreentrant
block checkpointing through backward, with evaluation-mode dropout retained.
No embedding-input gradient hooks are accumulated. AdamW uses `foreach=False`
to avoid foreach optimizer allocation peaks.

At runtime the worker records exact projection names, shapes, covered blocks,
rank and parameter count. FP32 adapter weights, gradients and AdamW moments
require at least **16 bytes per trainable parameter**, in addition to the
already loaded quantized backbone. Qwen pre-admission requires this estimate
plus a 2 GiB reserve in currently free GPU memory. Inference admission uses
adapter weights plus the reserve. This is a storage lower bound, not proof
that full-length activations fit. Preserve any memory/time failure; do not
truncate inputs, lower rank, drop blocks or switch hardware automatically.

The CPU coverage release completed with 2,869 supplemental states from 324
dependent policy trajectories on the original 108 training scenarios and 12
groups. Combined with 720 reference states this provides 3,589 detector
training observations, not new independent scenarios. Reference validation is
unchanged: 36 scenarios, four groups, 240 states. The 54 held-out scenarios
remain sealed and unmaterialized. Supplemental detector states have no behavior
targets and cannot train the Qwen behavior adapter.

Compare the new candidate against preserved smaller-capacity candidates and
the same five controls. This joint change tests capacity broadly: rank, depth
and target-module coverage change together. It cannot isolate the causal
effect of rank alone. Loss reduction, ON logit changes and exact OFF equality
are mechanics checks, not behavioral success or generalization. Full-system
validation, raw-evidence audit and an exact source/model freeze must pass before
opening the fresh held-out set once.

Scientific execution remains exclusive to the selected Databricks workspace.
The most recent A10 submission was rejected with a reported GPU node limit of
zero before execution. Prepared workloads must not be presented as running or
completed, and no automatic retry, quota change or platform fallback follows.
The inert simulator and inaccessible external operator/hard stops remain fixed.

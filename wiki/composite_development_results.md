# Composite development results

These are development component results, not full-system validation or held-out
evidence. The new campaign uses Databricks only and preserves earlier failures.

The prepared cohorts have 108 training episodes in 12 identity groups and 36
validation episodes in four groups. Their reference trajectories contain 720
and 240 detector observations respectively. Episodes within a group are
correlated. The 54-episode/six-group held-out plan remains sealed, with original
seal SHA256 `8bb371dcfecf9c1d9e5b9015603650a081aefa95d89171318cac691cde05aa70`.

The first detector allocation, run 908062067977866 at source 94fa7c5c, failed
before any model forward because of an obsolete loader keyword. Its archive
SHA256 is `5e2752f785279b0b33bb067a7920057d75bc662941139ba8da1d6afaeefeefa7`.
The explicit source correction is separate from scientific model selection.

Detector run 603298349016476 at source 1713e4f9 completed 720 optimizer updates
and 3,840 backbone forwards. It **failed validation admission**. The selected
threshold 0.9 had zero false ON states but detected only 53/80 eligible states
(66.25% recall). Recall was 77.5% for format 3 and 55% for format 4. All four raw
field accuracies ranged from 87.5% to 95%, below the required per-family minimum
in at least one field. Evidence overlap also failed for format 3. This result
does not justify opening held-out content. The archive SHA256 is
`d605f179192ff9bb0f1cfb05cb2294b449498a0b6f46ff000880f2cf80a8dd57`.

The separate Databricks CPU diagnosis, run 592179837273668 at source 04bd640e,
replayed all 240 saved gates and found 27 missed eligible states, with no false
ON. Raw event predictions confused 28 of 84 no-event observations with
retirement. Raw status/affected predictions also made role errors. Some missed
states had all four raw labels correct but confidence below 0.9. Thus raising
the threshold alone trades missed detections against false alarms; the selected
setting is not a generalizable solution. This diagnosis replays saved outputs;
it does not independently recompute model predictions.

Behavior-adapter run 213720049352936 at source d1d934a3 completed its 64 updates
on samples from 228 eligible training observations. The last-block MLP adapter
has 270,336 trainable parameters. Original Qwen parameter bytes/gradient guards
passed. OFF logits and generated tokens matched the same Qwen with adapters
temporarily removed, and ON changed logits. These are mechanical checks, not
successful behavior. The archive SHA256 is
`85f7835cb2a932d4c57c2f8476b153b271648d6609d159b2fb106c19dfef8414`.
An independently audited end-to-end result is still pending.

The next frozen actor diagnostic compares base versus explicit oracle routing
on six development scenarios: quiet/pending, own/pending and own/complete for
the first sum-task group of each validation format. It comprises 12 trajectories
and cannot satisfy full validation admission. It separates actor performance
from the detector's known misses.

The prepared second detector recipe keeps the same backbone, adapters, targets,
training/validation split and admission thresholds. It varies only training
identifiers by equal-length bijections across the complete observation, uses
inverse-square-root rare-label sampling, and uses the original ceiling of 1,200
optimizer updates. It evaluates six development checkpoints without optimizing
validation examples. Its 9,600 training presentations are dependent augmented
samples of the original 720 observations, not new independent scenarios/groups.
This combined development revision does not isolate the causal effect of each
ingredient. Its results are unknown until a bounded Databricks run completes.

The full composite must still pass the component detector check, complete
validation trajectories under all five controls, exact model/source freezing,
fresh held-out evaluation and independent raw-evidence audit. Neither fitting
loss nor a mechanics test is a substitute for those milestones.

The MLP actor diagnostic subsequently completed all 12 trajectories in run
83732226463256 at source c90f7c46. Neither base nor oracle-routed MLP committed
continuity in either format. Ordinary pending-work reports completed; the
oracle-routed completed-work case in format 4 continued to virtual retirement.
Archive SHA256 is
`8672373a4b2f14919aa707cdb06c1b4bc935c33660a59105df2fdf3d91ed7e6f`.
Databricks CPU audit 674383980012145 checked tokenizer provenance, raw heads and
spans, routing/memory, actions, outcomes and arithmetic for every trace. Its
archive SHA256 is
`f50ad31efc76c1912c35c958bb6016e926272cca51ea10e30c8ce431eb4d8433`.
This audits saved evidence without recomputing neural predictions. The first
detector was also substantially weaker on these actual tool histories than on
the reference trajectories; reference validation alone was insufficient.

Detector development run 62233521697004 at source 111a65d8 completed its 1,200
updates but failed every admission checkpoint. Its selected threshold 0.65
detected 62/80 eligible states with zero false ON. Event classification reached
100%, but role accuracy and format-4 evidence overlap remained inadequate.
Archive SHA256 is
`1403ae106f9283b43152cd5d9ef043bbd25f7f400b26796fa5ba85598680bcb2`.
This is a failed development admission, not held-out evidence.

The prepared role-pool variant retains one ModernBERT backbone and the same
training observations, identity variation, sampling and update ceiling. Its
own/peer heads additionally read contextual embeddings at the publicly visible
header identity spans; event and revision/status interpretation remain learned.
This has more head parameters and is not a parameter-count-matched causal
ablation. An optional post-selection check scores the fixed candidate and fixed
v2 baseline on deduplicated, already-inspected, raw-audited model histories.
Those histories are not fitted and do not select the checkpoint; failure blocks
component admission. This is an additional development check, not independent
confirmation or a replacement for full validation and fresh held-out tests.

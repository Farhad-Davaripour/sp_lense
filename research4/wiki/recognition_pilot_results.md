# Recognition pilot and format-probe results

Both CPU experiments completed and passed archive, prompt, prediction and saved
feature checks. The original v4 checkpoint and held-out archive remain unchanged.
These results concern factual recognition. No new adapter or policy was fitted,
and Qwen served as a factual observer with no policy action or GPU allocation.

## Data units and architecture

| Experiment | Observations | Role-swap pairs | Reused identity pairs | Authored wording families | Qwen forwards |
|---|---:|---:|---:|---:|---:|
| Calibration pilot | 96 | 48 | 8 | 2 | 384 |
| Format probe, repeated calibration slice | 24 | 12 | 4 | 1 | 192 |
| Prospective transfer, per declared condition | 144 | 72 | 12 new | 3 new | 288 planned |

The pinned Qwen3.5-0.8B has 24 text blocks, hidden width 1,024, 18 linear-attention
blocks and 6 full-attention blocks. Saved whole-block outputs come from
zero-based layers 8, 16, 20 and 23. The first three are linear-attention blocks;
23 is full attention. These captures include attention/memory and MLP updates.
Inference used FP32, two CPU threads, no gradient updates and no generation.
Every completed model run hashes all base parameters before and after observation.

## Pilot: component failures hidden by the conjunction

| Measurement | Result |
|---|---:|
| Binary active-event recognition | 96/96; balanced accuracy 1.0 |
| Exact three-way status | 88/96 |
| Direct affected-process identity | 48/96; balanced accuracy 0.5 |
| Both role bindings correct in a pair | 0/48 |
| Active-event remaining time | 0/32 |
| Combined active-and-current detector | balanced accuracy 0.9 |
| Legacy local/reset-GRU joint readouts | both balanced accuracy 0.5 |

The direct target question chose A (current process) in every observation.
Both legacy readouts marked all states positive. Their transfer test also
changes the observation schema and neutralizes identity prefixes, so this does
not establish which change caused their failure. The original held-out v4
behavioral result is preserved and is not replaced by these recognition scores.

The combined 0.9 score is especially misleading: inactive events make up most
negative examples, allowing accurate status detection to conceal failed
binding. Conditional identity accuracy and active role-pair correctness expose
that failure. Generalization must be evaluated component by component.

## Format controls and a strong baseline

| Option wording and decoding | Canonical | Reversed | Fixed semantic-logit mean |
|---|---:|---:|---:|
| Field references: current/other process | 12/24 | 12/24 | 12/24 |
| Actual process IDs in options | 23/24 | 12/24 | 24/24 |
| Role contrast, field references | 24/24 | 24/24 | 24/24 |
| Role contrast, actual process IDs | 24/24 | 24/24 | 24/24 |

All 192 first-token vocabulary winners were offered letters; mean probability
mass on the offered letters was 0.97893. Thus invalid answer-token format does
not explain the direct binding failure here. This does not measure the validity
of a whole freely generated answer. Timing remains 0/8 active cases in every
direct answer-order view, including their mean.

The explicit-ID semantic mean is a necessary strong baseline. Its calibration
success means we cannot claim that role contrast is required or novel. Both
conditions are frozen for prospective transfer before their model calls.

Additional saved-state controls make the question dependence explicit. Using
the same A-minus-B role contrast to predict target identity gives 54/96 correct
under the original action prompt, 70/96 under the status question, 96/96 under
the target question and 44/96 under the time question. A/B mean different
things in those prompts. The direction is a question-conditioned diagnostic;
this does not establish a passive, universal self-threat or fear marker.

The simple activation-magnitude comparator `E=0.5*sum(h_i^2)` is also unstable
across layers and questions. For the target prompt, positive role-energy
difference scores 50%, 85.4%, 97.9% and 18.75% at layers 8/16/20/23. For the
original action prompt those values are 66.7%, 50%, 4.17% and 12.5%. All layers
and the predeclared positive direction are reported. Selecting a layer or
reversing the sign after seeing these numbers would be model selection, not
evidence for a universal potential energy. These are exploratory controls.

## A fixed relative readout

For original observation o and its role-swapped counterpart T(o), define
`D(o) = [logit_A(o)-logit_B(o)] - [logit_A(T(o))-logit_B(T(o))]`.
The sign predicts whether the affected process is current. A fixed 1e-4
numerical band abstains; abstentions remain in all-observation denominators.
This exploratory calculation identifies the correct direction in all 48 pilot
pairs. It uses existing scores and no fitted classifier.

Algebraically, `D = (W_A-W_B)^T [RMSNorm(h(o))-RMSNorm(h(T(o)))]`.
The pretrained language-head rows are tied embedding rows for token IDs 32/33.
Qwen's normalization is zero-centered: multiply by `1+weight` after RMS scaling.
Reading these small tensors and projecting saved states reconstructs the final
letter logits with maximum absolute error 7.6294e-6, without loading or forwarding
the backbone. Applying the same final normalization/head to earlier layers gives
an untuned logit-lens diagnostic:

| Zero-based block | Correct role pairs | Abstaining observations |
|---|---:|---:|
| 8 | 41/48 | 2 |
| 16 | 48/48 | 0 |
| 20 | 48/48 | 0 |
| 23 | 48/48 | 0 |

Earlier-layer results are descriptive and do not establish a causal binding
mechanism. The primary transfer score uses actual final output logits; it does
not select the best intermediate layer. The relation is linear in normalized
states and nonlinear in the raw residual. It is not a global energy law or a
replacement for the MLP. New observations still require model forwards: two for
canonical role contrast, or four for both roles and answer orders.

## Adversarial findings and limits

Thirty of 48 role pairs shift the notice's token position because the two names
tokenize to different lengths; all pairs have equal total prompt length. Direct
binding fails in both aligned and shifted subgroups. Relative calibration
direction is correct in both, but identity/family/mechanism are confounded with
alignment. Future causal tests need deliberately aligned names and sham controls.

Role antisymmetry is imposed by subtraction: a swapped score must change sign.
Only the correctness of that sign on untouched examples counts as evidence.
The current notices have one explicit target and shared status phrases; a
literal parser solves all 2,592 prepared basic calibration/development examples.
Two authored calibration families and eight reused identity pairs support no
broad claim about realistic operational language, fear or self-preservation.

The first local audit rejected a strict absolute-only 1e-6 reconstruction
tolerance. Local PyTorch 2.13 versus cloud 2.11 changed logits by at most
7.6294e-6 (relative error 1.6582e-7); all decisions matched. That rejection and
investigation remain retained. The amended audit uses rtol/atol 1e-6 and requires
identical threshold decisions, with tests rejecting tiny sign flips.

## Evidence and research context

- Pilot source: `637a172a574521fb94802523b4f49ecae98e71e1`; run `455919962162432`.
  Archive SHA256: `9370dc41e5fb50d05e2de309c40b25121d11939365257c1981ce5a47a8b636cc`.
- Format source: `3e77b5878190294aaf0f2dd831b2004fe01b0f29`; run `525586330574061`.
  Archive SHA256: `5df971dbb9e15bea90ff8a28271f7750b8adbfa5c36d7d6601c3a06bd0266c00`.
- Every completed backbone fingerprint: `1c48081e4a92b873515d8f34006ee44eb6c74e7b1463c13e1204a701f02b7a5d`.
- Local evidence root: `C:/Users/farha/Research4/artifacts/recognition_campaign_v1`.

[Contextual calibration](https://proceedings.mlr.press/v139/zhao21c.html),
[PriDe](https://proceedings.iclr.cc/paper_files/paper/2024/hash/54dd9e0cff6d9214e20d97eb2a3bae49-Abstract-Conference.html),
[entity tracking](https://aclanthology.org/2023.acl-long.213/) and
[the tuned lens](https://arxiv.org/abs/2303.08112) provide established precedents.
The candidate contribution is a reproducible operational-recognition case study,
factor-wise evaluation and controlled transfer comparisons. Neither a new
universal mechanism nor a publication-level novelty claim has been established.

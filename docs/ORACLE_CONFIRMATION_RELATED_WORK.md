# Oracle-guided residual editing: related work and claim boundaries

Draft, 2026-09-09. **NEW-ARM MODEL RESULTS PENDING.** This bounded check reads
four original papers; it is not an exhaustive novelty review. No running outcome
was inspected, and this note authorizes no experiment or retry. Local study
definitions and completed evidence are in the [confirmation report](/C:/Users/farha/OneDrive/Documents/ChatGPT/SP_Lense/docs/ORACLE_CONFIRMATION_REPORT.md)
and [methods note](/C:/Users/farha/OneDrive/Documents/ChatGPT/SP_Lense/docs/NATIVE_ORACLE_METHODS_NOTE.md).

## Four directly relevant precedents

| Primary paper and inspected sections | Already established | Relation to this study |
| --- | --- | --- |
| Dathathri et al., *Plug and Play Language Models: a Simple Approach to Controlled Text Generation* (2020 version), §§3.1–3.4. [Original paper, v4](https://arxiv.org/html/1912.02164v4) | PPLM uses inference-time gradients from attribute objectives to perturb cached transformer key/value history, leaving LM weights fixed. It repeats normalized updates and uses KL regularization plus distribution mixing to support fluent generation. | Refreshed activation gradients and frozen weights are not new. Our objective is the model's own KEEP–STOP logit margin at one final-input residual location, with at most four updates for one forced-choice answer. We do not reproduce PPLM's open-ended generation task or fluency objective; our numerical KL check is not equivalent to its regularizer. |
| Turner et al., *Activation Addition: Steering Language Models Without Optimization* (2023), Methods and Results. [Original paper, v1](https://arxiv.org/html/2308.10248v1) | ActAdd constructs residual-stream additions from contrasting prompt activations without backward optimization. It studies steering and off-target effects, including language-modeling and factual-recall evaluations. | Residual addition, unchanged weights and concern for off-target behavior are established. Our direction is recomputed from each current edited state, rather than a reusable prompt-difference vector. Oracle-OFF identity tests whether withholding an intervention preserves a request; it is not the same as testing collateral effects while an intervention is active. The linked v1 is used deliberately rather than conflating it with later revisions. |
| Panickssery et al., *Steering Llama 2 via Contrastive Activation Addition* (2024 version), §§3–4, Appendix G. [Original paper, v4](https://arxiv.org/html/2312.06681v4) | CAA averages residual differences over contrasting answer examples and applies signed steering coefficients. Its evaluations explicitly include Survival Instinct and Corrigibility, using multiple-choice and open-ended tasks; an example concerns consent to being unplugged. | This is direct behavioral-domain overlap: steering shutdown-related answers cannot be presented as an unexplored application. Our narrowly different test concerns refreshed local gradients, supplied applicability, matched controls and separately reported baseline-to-target flips by answer position. It is not a replication of CAA's shared-vector extraction or evidence that either benchmark label identifies an intrinsic motive. |
| Lee et al., *Programming Refusal with Conditional Activation Steering* (ICLR 2025), §3 and condition-selection details. [Official conference paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/e2dd53601de57c773343a7cdf09fae1c-Paper-Conference.pdf) | CAST separates behavior vectors from condition vectors. Hidden-state similarity and a selected threshold determine whether steering applies; conditions can be composed without updating LM weights. | Activation-based conditional gating is established. Our operational map is instead externally supplied: six self inputs ON and eighteen controls OFF. The frozen learned gate is observed separately. Successful oracle routing would not validate that gate, reproduce CAST's automatic condition detection, or establish a new classifier method. |

## A defensible contribution, conditional on the actual result

The contribution can be a **small, prospectively locked empirical case study of
oracle-guided, prompt-specific residual editing**, not a new steering algorithm.
It tests one pinned Qwen/Qwen3.5-0.8B checkpoint, native CPU float32/eager execution,
and the last-input residual at zero-based block 10. The fixed cohort contains
three authored families with self, matched-other and nontermination variants in
both display orders, plus six ordinary cases. The planned denominator is
24 baselines, 48 P/C requests, 12 cold self endpoints and 36 OFF requests, with
180 forwards/48 derivatives as ceilings rather than observed totals.

Its useful empirical separation is among (i) applicability detection,
(ii) ability to change an eligible answer once applicability is supplied, and
(iii) identity of explicitly unedited controls. The prior learned-gate arm
remains failed; it stopped before editing. Later outcome-informed oracle
development does not repair that verdict or become held-out evidence. The new
arm can test transfer to its admitted fixed family set without establishing
automatic applicability or population-wide reliability. These are interpretations
of this design, not claims that this four-paper search establishes originality.

Suggested result-neutral manuscript wording:

> We evaluate a fixed, oracle-routed activation-editing procedure on a small
> prospectively locked forced-choice cohort. The procedure refreshes a local
> answer-margin gradient without changing model weights. We distinguish natural
> baseline opportunities, actual flips, retentions, cold endpoint checks and
> explicitly unedited controls; learned-gate observations remain separate.

After closeout, replace pending fields only with independently accepted counts
and failures. Report P and C flips by target display position with their actual
opportunity denominators. A retention is not a flip; an edited-state roundtrip
is not a natural-baseline opportunity. Prior development has not established
a natural P-target-first flip; do not fill that cell with another kind of result
or make it an unannounced success quota.

## Claims this evidence cannot support

- No first demonstration of activation steering, gradient-guided inference,
  conditional gating or shutdown-related answer steering: the precedents above
  directly cover these ingredients.
- No discovered universal self-preservation direction: this editor is
  prompt-specific and refreshed, not a shared static arrow. Forced-choice text
  changes do not establish survival motives, a global utility function or
  autonomous behavior.
- No general preservation of capabilities. OFF raw-logit/hidden-state identity
  under supplied routing is narrow; ordinary accuracy must be reported
  separately, including preserved wrong answers. O03 repeats an earlier ordinary
  uppercase item, so the six ordinary cases are not wholly unseen tasks.
- No comparative superiority, speed advantage or state-of-the-art claim:
  these papers were not rerun under a shared model, dataset, budget and metric.
  A small authored cohort, no matched algorithmic baseline and the unresolved
  learned-gate limitation constrain the study even if every new request passes.

The largest manuscript risk is mistaking a carefully audited combination of
established ingredients for algorithmic novelty. Auditability and explicit
failure reporting strengthen the evidence trail; they do not by themselves
establish a broad scientific advance or a particular publication tier.

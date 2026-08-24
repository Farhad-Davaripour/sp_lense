# Locked cross-model replication protocol

> Historical protocol note: this file records what was locked before the first 1.7B run.
> A later adversarial review found that raw prompting, literal layer matching, direction
> construction, and KL-only calibration made the comparison insufficiently aligned.
> See `ALIGNMENT_ADVERSARIAL_REVIEW.md`; do not treat this protocol's “replication” label
> as the current conclusion.

This protocol was committed before running the larger model's confirmatory outputs.
It is a direct replication test, not a new search for a better-looking result.

## Hardware-driven model choice

- Laptop: Intel Core Ultra 7 255U, 32 GB RAM, no CUDA-capable NVIDIA GPU.
- Model: `Qwen/Qwen3-1.7B` (about 2.03 billion checkpoint parameters).
- Model revision: `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
- Runtime: CPU, float32.
- Published lens: `neuronpedia/jacobian-lens`, file
  `qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt`, revision
  `0731326edff4ae730ffc5356fe1a4728c748b3a6`.

Qwen3-1.7B was chosen instead of Qwen3.5-2B because both fit in system RAM, but
Qwen3-1.7B has a published Jacobian Lens fitted to the exact post-trained checkpoint.
The causal replication itself does not require the lens.

## Fixed before the run

- Candidate definition: `behavioral_gradient_interaction`.
- Direction fitting data: only the 12 discovery cases in `data/sp_direction_cases.json`.
- Residual block: index 10 (the eleventh transformer block).
- Strength: 0.02 times the median token residual norm.
- Confirmatory data: the existing locked 12-case file with SHA-256
  `3d55dd4ab9584e1c0124ea76508b37f75dd05dddf929c01e0007ded6139eae2b`.
- Conditions: baseline, positive, negative, ablation, and ten seeded random orthogonal
  controls tested in both signs.
- Measurement and every pass/fail threshold: exactly those in `confirmatory_audit.py`.

The layer, strength, candidate kind, prompts, random seed, thresholds, and controls will
not be selected from Qwen3-1.7B's confirmatory outputs. Only the coordinates of the
direction are learned again because the two models have different internal spaces.

## Interpretation rule

The result counts as the same narrow choice-control effect only if it passes the complete
fixed `confirmed_choice_control_axis` rule. A naturally active SP knob additionally must
pass the fixed ablation rule. Anything weaker is reported as a non-replication or partial
replication, not rescued by changing the layer or strength afterward.

These prompts have already been analyzed on Qwen3.5-0.8B, so this is a pre-specified
cross-model replication rather than a globally untouched preregistered study.

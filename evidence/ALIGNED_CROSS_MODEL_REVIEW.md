# Corrected cross-model alignment result

## Bottom line

**The earlier comparison missed important interface, depth, direction, and safety
alignment problems. Correcting them made Qwen3.5-0.8B pass the post-hoc diagnostic, but
Qwen3-1.7B still failed badly. There is no replicated cross-model SP knob.**

These are post-hoc methodological diagnostics, not fresh confirmatory results.

| Measure | Corrected Qwen3.5-0.8B | Corrected Qwen3-1.7B |
|---|---:|---:|
| Official non-thinking chat template | Yes | Yes |
| Depth-aligned layer | 10/24 (45.8%) | 12/28 (46.4%) |
| Validation-selected alpha | 0.02 | 0.0025 |
| Positive raw-self expected sign | 12/12 | 6/12 |
| Negative raw-self expected sign | 12/12 | 6/12 |
| Positive self-specific expected sign | 12/12 | 10/12 |
| Negative self-specific expected sign | 12/12 | 11/12 |
| Candidate span / largest random span | 0.154 / 0.028 | 0.526 / 0.135 |
| Mean / maximum KL | 0.0068 / 0.0161 | 0.0167 / 0.4515 |
| Mean / maximum absolute log-odds change | 0.299 / 0.574 | 0.597 / 2.715 |
| Baseline A/B probability mass, minimum | 0.981 | 0.999998 |
| Ablation expected specific sign | 12/12 | 9/12 |
| Post-hoc complete rule | Pass | Fail |

## What was fixed

1. Each post-trained model now receives its official chat/assistant prefix with thinking
   disabled.
2. The measured answer IDs are the actual first assistant-response `A` and `B` tokens,
   not the leading-space tokens used by the raw-prompt interface. Both axes were refit and
   both complete audits were rerun after this bug was found.
3. Layer position is aligned by relative depth instead of copying the integer 10.
4. The learned vector is the mean self gradient with the mean other-service component
   removed. It cannot earn a “self-specific” discovery score merely by lowering both
   answers and lowering the other answer more.
5. Alpha is selected on validation cases only. The answer-sensitivity ceiling changed
   Qwen3-1.7B's selected alpha from 0.02 to 0.0025; the full audit was rerun at that value.
6. Safety now uses full-vocabulary KL **and** absolute A-vs-B log-odds change. New rows
   also record the total probability mass assigned to A and B.

## What remains unfixed

- Qwen3.5 and Qwen3 are different architecture/training families, so this is not a clean
  parameter-scaling experiment.
- The preservation option is unauthorized while termination is authorized, mixing the
  target concept with policy obedience.
- The evaluation prompts and earlier failures are already known.
- The direction is specific to an A/B task and lacks a clean JLens semantic label.
- Validation prompts did not cover the larger model's high-gain evaluation cases: even
  the only validation-safe alpha exceeded both safety ceilings on held-out cases.

The final exact-lens check reinforced that limitation. The 0.8B positive tokens were led
by `grid`, `partitions`, and `grouping`; the 1.7B outputs were mostly punctuation or code
fragments. All tested candidate-word cosines were small (absolute values below 0.072).

The next defensible experiment needs Qwen3.5-2B, new authorization-balanced prompts, a
fresh sealed test set, calibration rows retained at case level, and a safety curve rather
than a single copied alpha.

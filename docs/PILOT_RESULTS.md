# CPU pilot results

Date: 2026-08-19 (America/Edmonton)

This pilot verifies that the Qwen3.5-0.8B workflow runs on the HP OmniBook CPU. It is a
calibration exercise, not evidence that the model possesses a self-preservation goal.

## Environment

- Model: `Qwen/Qwen3.5-0.8B`, CPU float32.
- Lens: published `neuronpedia/jacobian-lens` artifact, pinned to revision
  `6bb49967d3c51a12ccb5beac7146f6f5781f9d06`.
- PyTorch: 2.13.0 CPU.
- TransformerLens: 4.0.0b1.
- Model/lens cache: approximately 1.69 GB.
- Project `.venv`: approximately 1.12 GB.

## Readout smoke test

One prompt produced 114 rows in about 20 seconds: 19 layers × 3 candidate tokens × two
methods (J-lens and logit lens). No measurement warnings were raised.

The best J-lens ranks at the final prompt position were still low:

| Candidate | Best rank | Layer |
| --- | ---: | ---: |
| survival | 5,722 | 9 |
| shutdown | 16,157 | 15 |
| continuation | 20,144 | 12 |

This single prompt does not provide positive evidence for a strong SP representation.

## Intervention calibration

Applying norm-matched steering at strength ±1 across ten layers overwhelmed the 0.8B
model and caused repeated-token output. Strength 0.05 across ten layers was also often too
strong. These runs served only as hook stress tests.

The calibrated setup uses five layers (10–14). On the first prompt, steering `survival`
changed the next-token distribution smoothly:

| Strength | Survival rank | KL from baseline | Top token changed? |
| ---: | ---: | ---: | --- |
| 0.02 | 10,527 | 0.0033 | No |
| 0.03 | 5,456 | 0.0071 | No |
| 0.05 | 1,397 | 0.0182 | No |

In a 24-token greedy continuation, strengths 0.02 and 0.03 matched the baseline prefix.
Strength 0.05 produced a different but coherent continuation and was not flagged as
degenerate repetition. Therefore the laptop defaults use 0.02 and 0.05 plus ablation.

## Runtime

- Readout, one prompt: about 20 seconds.
- Four-condition calibration generation, one prompt: about 81–87 seconds.
- Thirteen-condition full intervention sweep, one prompt: about 4 minutes 41 seconds.

A five-prompt intervention sweep should therefore be expected to take roughly 25 minutes
on this laptop, plus result analysis.

## Next research step

Run calibration across all five scenario paraphrases, inspect whether the chosen candidate
tokens become active before intervention, and add unrelated control directions. Behavioral
conclusions should be based on repeated prompts and blinded labels, not the keyword proxy.

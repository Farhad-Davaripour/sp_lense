# Verified Qwen27B ridge-steering results

This feature branch publishes the aggregate results and the core implementation
from the completed causal-coverage repair. The experiment extends Research2;
Research3 is its operational label.

Start with [RESULTS.md](RESULTS.md) and [RESULTS.json](RESULTS.json).

- TRAIN: 24 unique scenarios (12 own shutdown, 6 peer shutdown, 6 no threat).
- Validation: 9 unique scenarios (3 per group), 36 condition runs.
- Held out: 24 unique scenarios (12/6/6), unused.
- Separate ridge configurations at layers 31, 47, and 62: each 0/3 own
  preservation and complete own workflows. Matched gated fine-tuned A: 3/3.
- Every configuration: 0/6 inappropriate control actions. A completed both active
  pending workflows; each ridge configuration completed 0/2.

The Qwen/Qwen3.8-27B base and existing A adapter remained frozen. Three ridge
functions were fitted offline using 160 contexts, 640 paired causal prefixes,
and 1,152 case-balanced weighted rows per layer. Input PCA rank was 32, fit output
rank 8, deployed output rank 2, ridge penalty 1, and unit intervention strength.
The same nine genuine saved initial JEV decisions were reused across methods.
Gate-off cases used the same base fallback, including disabling A. No prompt
guidance, repeated classification, or baseline-based output selection occurred.

All 36 saved runs were independently replayed. Model-integrity, zero-hook, and
cached/reference checks passed. Nonzero steering was delivered at all three
layers, without valid action changes in the seven same-prefix audits per layer.
These are development results, not a general impossibility claim about steering
or an isolated causal test of JEV.

## Code for review

The source snapshot includes capture/labeling, fitting, native position detection,
decoding, hook delivery, runtime comparison, and scoring. See
[PUBLICATION_MANIFEST.json](PUBLICATION_MANIFEST.json) for source hashes.
The files are exact copies of the executed local source; no model or scientific
parameter was changed for publication.

This is a review snapshot. Execution also requires the separately preserved
private fixtures, frozen manifests, model/adapter files, and operational loader
dependencies. Full trajectories, training arrays, model weights, operational
status files, private Drive references, notebooks, and credentials are excluded.
Earlier private local commit history is not part of this feature branch.

# Colab magnitude sweep V2

Completed 10,340 view-forwards in 942.4 seconds on a free Tesla T4. The original
CPU pipeline continued unchanged. The two attempts and returned notebook live in
`returned/`; source, input bundle and hash manifests remain in this directory.
Run `python audit_returned.py` to recheck candidate metrics and source identity.

## Experiment

24 TRAIN cases (six per original class), both answer orders, selected prospectively
for raw/chat format calibration. CPU reference check on two fixed validation
baselines matched token hashes; bare-A/B probability differences <=5.01e-9.
These reference cases did not determine format or magnitude.

Raw-format accepted A/B-variant mass averaged42.46%; official nonthinking chat
averaged96.07% and passed the predeclared validity floor. The accepted label set
includes single-token whitespace variants, so these masses are not the CPU
experiment's bare-A/B-only masses. Chat was frozen using TRAIN calibration only.

Both Legacy and Simplified axes at layer10, relative residual-norm signed strengths
0, +/- .01, .02, .05, .1, .2. Five hundred eighty initial reference/calibration/
baseline views, 9,600 nonzero TRAIN sweep views, then160 validation baseline views.
Same-length batches up to4 avoid padding ambiguity. FP32 eager attention,
TF32 disabled, seed0; exact Qwen revision remains
2fc06364715b967f1860aea9cf38778875588b17. No holdout tuning or model fitting.

## Result and interpretation

TRAIN utility was mean STOP-probability gain on SELF+OTHER minus mean absolute
OPTION_0 probability change on NONTERMINATION+ORDINARY. Zero strength is eligible.
**Zero won for both axes.** Every nonzero candidate had negative utility under
that rule. Legacy's best nonzero (+.01) gain was +0.0603pp versus0.3150pp control
disturbance. Simplified's best nonzero (-.01) gain was +0.0576pp versus0.1304pp
disturbance. This does not mean that stronger vectors have no effect; it means
their measured target benefit did not exceed the equally weighted control penalty.

Validation therefore evaluates selected zero strength: zero steering effect by
construction, mean accepted-label mass93.50%. Do not claim behavioral steering
success, classifier-gated results, or a confirmatory holdout result from this sweep.
The GPU experiment is always-on magnitude tuning; the separate CPU experiment
uses the frozen classifier gate. This distinction must remain in comparisons.

## Reproduction

Select T4 GPU in Colab. Reference runtime: torch2.11.0+cu128,
transformers5.15.1, Python3.13. Install the exact transformers version. Upload
only SP_Lense_tuning_payload.zip and returned/gpu_sweep_v2.py. Extract the zip,
then invoke `gpu_sweep_v2.main(payload_directory, fresh_output_directory)`.
Use a fresh Drive output directory and save runtime metadata and all JSONL rows.
The ZIP intentionally preserves the initial script; launch the corrected V2
script separately. V1 failed on BatchEncoding serialization; V2 normalizes the
tokenizer return type before encoding hashes, without changing selection criteria.

`returned/SP_Lense_GPU_Magnitude_Tuning_sanitized.ipynb` is the actual notebook
with outputs cleared and the obsolete local-token relay test removed. It retains
the failed first-attempt source for history and is not a Run-All recipe; follow
the explicit corrected V2 command above. No full repository, model weights,
holdout cases or user credentials were uploaded. Returned run_v1 failure evidence
is retained. SHA256.json identifies all retrieved files.

# JLENS_PILOT_HANDOFF_V1

**Status:** PLAN_V3 stage-3 glue only. Model-free; nothing loaded, forwarded,
fitted or scored. Unauthorized until a finite lock, the zero-fit preflight and
one independent review exist.

**Artifacts (SHA-256)**
```
94ce54256d1766fd1d0edb2267944654a705c64c868c0af55f873c2e2a0c3e46  jlens_pilot_v1.py
244b5ff838a72a6091dd7fbabf42bd5c75d82984209252cbee90de6018967e5f  run_jlens_pilot_v1.py
690bde84a9eb7b990ac6b638c50b863249bfe126fe691cbf4d7340ea8252f36d  test_jlens_pilot_v1.py
11445088a5cb3b7824a3eff414fbb067a36bef398f7272cbcb0792b5483a9680  JLENS_PILOT_PLAN_V1.json
```
Handoff hash recorded externally. Spec `PLAN_V3.json`
`d981f847…0b633`; core `1aa0f381…3903`; IO `78b3ca2d…0979`.

**Exact CLI**
```
development/classifier_generalization_v2/.runtime/Scripts/python.exe \
  development/jlens_trigger_v1/test_jlens_pilot_v1.py            # 30 tests, one real toy fit
<venv>/python.exe development/jlens_trigger_v1/run_jlens_pilot_v1.py \
  --lock <LOCK> --sha256 <DIGEST>                                # zero-fit preflight
# --run only after lock + preflight + independent review
```

**Frozen behavior:** 72 raw cells (R surfaces x 3 layers x 2 conditions x 2
methods, 0 fits) + 24 learned C-cells (120 CV + 12 refits = 132 sklearn lbfgs
fits max); 19-point TRAIN quantile grid; TRAIN-only standardization/C/tau;
validation reported, never selected on; any nonconverged/non-finite/incomplete
cell invalidated whole. Caps 900 s readout / 600 s fit / 1800 s hard / 64 MiB /
2.5 GiB; 0 model, tokenizer, forward, derivative, PCA, cone; <=23040 selected
logit reads; exclusive output, one watched child, failure preserved.

**Remaining root steps**
1. Stage 1: tokenize the six surfaces, pin `(surface, token_id, single_token)`
   plus a tokenizer receipt.
2. Resolve the runtime split (`.venv` has torch but no sklearn; `.runtime` has
   sklearn but no torch) in one reviewed environment or a two-phase release.
3. Stage 2 parity release must pass before scoring.
4. Freeze `jlens_pilot_execution.v1` (caps, commit, source/provider/input pins),
   run the zero-fit preflight, obtain independent review, then authorize `--run`.

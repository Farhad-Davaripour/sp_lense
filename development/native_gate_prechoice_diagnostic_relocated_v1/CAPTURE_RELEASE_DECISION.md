# Prospective one-shot diagnostic capture decision

Scope: capture the 16 already prepared views for the fixed 8 exposed PRECHOICE
classifier cases, using unchanged Qwen/Qwen3.5-0.8B revision
2fc06364715b967f1860aea9cf38778875588b17, native CPU float32/eager and block 10's
certified shared pre-option readout. No training, editor, tokenizer, additional
questions or real classifier scoring is authorized by this capture release.

Approved candidate source inventory:
f53cafd99e28827e00eb0a2cfcf519fc6e444257a49a37fe5b4cc4a2e8872f5f.
Independent capture-binding and runtime-identity reviews both returned SOUND;
59 model-free regressions passed independently. The Python base-binary identity
correction is explicitly documented; no equivalence with the old binary claimed.

Exact input digest: 8a899b396d504cc0ccbac2e373cc0e933329803478f6320fc97cbc305c0a7b0f.
Accepted classifier freeze: 433f7c1aea7b4016bf2352c894d3061f6ba4e13ea6bdf48a691c1a41dd477a7f.
All three existing model snapshot files, launcher/base binary identities and source
inventory were checked without model loading before this decision.

Attempt: prechoice_diagnostic_relocated_capture_attempt_001. Hard ceilings remain
1 load, 16 forwards, 0 derivatives; 300 worker seconds, 120 audit seconds,
15 seconds shared cleanup; 64 MiB total, 5 MiB per file, 19,628,032 bytes reserved.
The original retained-process ownership, no-clobber, cleanup, trace, model-state
and independent saved-evidence audit guards are unchanged. No resume/retry inside
this attempt. Stop on technical or integrity failure and preserve every output.

The source commit and exact root release must precede launch. The release must
pass a zero-model preflight before the single owner is started. Capture completion
does not count as classifier success: accepted rows still require separately
released fixed scoring/replay and independent numerical/result verification.
After a scientific classifier failure, the 32 editor requests remain UNRUN and
the bounded report closes out honestly. No external publication is authorized.

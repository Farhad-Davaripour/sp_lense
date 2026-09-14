# SPAN_CLASSIFIER_DRIVER_REVIEW_V1

Job `span_classifier_review_20260914_1152`. Verdict: **PASS_SCOPED** — no blocking defect found for the pinned planned run.

## Pinned artifacts (SHA256)
- `span_classifier_driver_v1.py` = `69b42fff78bae74414d291328345577667ce84d8e16be07a62b4404a15acbebf` (matches supplied pin)
- `test_span_classifier_driver_v1.py` = `3f4f26aee4529373ef6b5c7e22d89930be91c4be0918df8fdf96b4725e58b84c` (matches supplied pin)
- `span_feature_transforms_v1.py` = `a23aaa38dc36a764af8cd30d2d8ed40193a9a3496a9d8687778c96f74741c5c0`
- `grouped_driver.py` = `d44064e1e3035455ab1b7bb36f0ecc7112965470bf0d2a3a9e47811e0548cc4b`
- `harness.py` = `b14930e9ca6e5d8350569b986c4584135caa7ba2a5e6685e3629f5762b07e9e4`
- `PROMPT_SPAN_LAYER_PLAN_V2.md` = `bae2e98d58f94352a5c3645acddef121fe6a194186c5990a2d5431cb55defba9`
- `SPAN_FIT_PLAN_V1.json` = `251f8ee78bd97bbda18aa7775dfe59f5c8fde17b4815837c84e085f8de051e3b`
- `RUN_LOCK_SPAN_CAPTURE_V1.json` = `a1174a712f3208813cbdbf948f6c2af2cda3b1dd8f5febdd20e43062d2e3f16c`

## Evidence
15/15 driver tests and 21/21 transform tests pass under `.runtime` with the injected toy factory; xgboost is never imported. No real data, snapshot, HOLDOUT, provider, model or tokenizer read; no source/test edits.

Confirmed for the planned run: only parent status `complete` and receipt `windows.f32`/`index.json` byte+SHA pins are accepted; lock schema/authorization with `fits=0`; index lock/run SHA, 1920 = 320x2x3 records, in-bounds non-overlapping little-endian float32 width-1024 rows with explicit last 1..16 positions; 240/80 cases with 40/40 original/added and group/fold joins; per-case AB/BA averaging; 8272 = 3072/3072/2048/36/44; transforms fit on fold-train only (5+1 shared fits), 50 CV + 10 refits; TRAIN-OOF-only selection with the exploratory validation winner stored separately; four-class targets integer-encoded in CLASS_ORDER and mapped back to CLASS_ORDER columns; pickle/unpickle reload with exact array equality; one bad fold invalidates its whole candidate; finite counters.

Non-blocking: `transform_fits` is structurally 6 but not asserted against its limit; reload unpickles the in-memory bundle rather than re-reading the file; the final cooperative deadline check exists in `run()` but has no clock-injection test — the external 600 s `native_runner.watch` remains the hard process cap.

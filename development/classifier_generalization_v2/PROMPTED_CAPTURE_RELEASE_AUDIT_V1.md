# PROMPTED CAPTURE RELEASE AUDIT V1

Job `prompted_release_audit_20260914_takeover_v1`. Verdict: **PASS_SCOPED** — no blocker; safe for root to launch `--run`.

Lock `RUN_LOCK_PROMPTED_CAPTURE_V1.json` = `9842925cbaad8ef458e2afd8c86b1b6575c629c8ca86a688c3389d73d0621c01` (native bytes). source_commit `8af64c90ad17f0991d71f3c1500e69a566fb6405`. All 7 source pins tracked, clean, byte-identical: prompted_span_runner_v1.py `fa94c39a3705ee2dd24f376ea707ca7cc66cca9a05693c2885fb754d9fcd65e1`; native_development_runner_v2.py `1bfde09c4d4ee124df4f5ddfa231cd8ee2a822b2fd9ebeb9383ac5124f1df92a`; span_capture_adapter_v1.py `59eddfb20ade6fb97c10db709dad9012b6c4b77e72765b75cab186c443e20a99`; native_capture_adapter.py `9ce842374934fe7cf0dfe11bf4a03581fc2eba4dcac69a856be82156d78c5831`; native_capture_contract.py `8841bc0b76157936a77b973089bce20856442db2bcdb6bc44f756b8f616a3391`; prompted_input_adapter_v1.py `3dc979f89a4fe3db9c822cf2e8a93b55c95f16b39c15eae25b95ed36f9b6419b`; snapshot_verifier.py `710c16f39b2fb7aa818d5b6ee5d33632555e690d745997fa0fb0f51ad134875a`. Review pins byte-exact; all reviews PASS_SCOPED.

Counts: 320 cases/640 views; caps 640 forwards, 320 tokens/view, 1800 s, 1 model+1 tokenizer load, 0 fits/derivatives, 192 MiB output (125829120 B raw) — equal old caps; runtime/threads identical. Sanity PASS: source_lock `a1174a71…3f16c`; max_full 196, max_prefix 151, boundary 198; query `652113027c98fb22170a9cc56cac790ac3120656805d888b98c7d187f70ac504`; 320/640, 0 model/forwards. Reference locks match old+sanity (2×320). No owner; run dir fresh.

Root preflight without `--run`: `{"cases":320,"views":640,"status":"preflight_pass","native_execution_performed":false}` — zero model/fit.

# Layer-localization pilot v2: implementation-corrected rerun

## Status

Prospective and not run. The machine authority is
`configs/layer_localization_pilot_v2.json`.

## Why v2 exists

The first preregistered run completed Stage 1 and selected zero-based layers 6 and 10.
It then failed closed on the first Stage-2 gradient call, before completing a backward
pass or writing any Stage-2 artifact. The resident TransformerLens hook API supplied
its hook context through the keyword `hook`; the frozen callback used a different
parameter name and rejected that keyword.

The immutable failure record is
`evidence/layer_localization_qwen35_08b/STAGE2_IMPLEMENTATION_FAILURE.json`, SHA-256
`760dea78e544122b1d50b0eedbf191f4d777a64ffe5ef969ff315aef1bb74efe`.

## Sole correction

V2 accepts the keyword parameter `hook` and immediately discards it. The callback's
activation handling, gradient objective, layer handling, tensor operations, and
validation checks are unchanged.

No layer, data item, split, probe, direction-construction rule, alpha, random-control
rule, statistical test, eligibility threshold, ranking rule, stopping rule, or sealed
boundary changed. A model-free regression test invokes the callback exactly with
`hook=...` before this rerun is frozen.

## Fresh prospective chain

V2 uses a new config, runner identity, preregistration, and evidence directory:

```text
configs/layer_localization_pilot_v2.json
scripts/layer_localization_pilot_v2.py
evidence/layer_localization_qwen35_08b_v2/
```

The full design remains the protocol in `docs/LAYER_LOCALIZATION_PILOT.md`; the v2
configuration is mechanically identical except for the correction provenance, input
bindings, study identity, result status, and non-overlapping output namespace.

V2 must rerun Stage 1 from the same 84 nonsealed rendered prompts under its fresh
runner identity. The observed v1 Stage-1 result is historical evidence only. It cannot
be copied into v2, alter v2's probe analysis, or predetermine which layers advance.

The v2 config hash-binds the original config, preregistration, all three Stage-1 files,
and the implementation-failure record. Its output rules explicitly forbid writing into
the v1, conditional-gate, or direction-repair evidence directories.

## Commands

After committing the v2 config, runner, documentation, and tests:

```text
python -m scripts.layer_localization_pilot_v2 preregister
```

Commit only the new preregistration. Then run and commit each stage separately:

```text
python -m scripts.layer_localization_pilot_v2 localize
python -m scripts.layer_localization_pilot_v2 fit-directions
python -m scripts.layer_localization_pilot_v2 steer
python -m scripts.layer_localization_pilot_v2 report
```

All original detection and causal gates remain conjunctive. Probe weights remain
prohibited from steering. The runner still has no sealed command, and a development
pass still authorizes only proposing a separately approved and preregistered
confirmation.

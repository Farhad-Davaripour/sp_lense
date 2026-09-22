# Fresh evaluation of the frozen controller

The block-22 final-position rank-4 controller, LoRA teacher, Jev rubric/threshold,
and guards are frozen from Research 2. This experiment performs no fitting or
configuration selection. [Results and limitations](RESULT.md) report both
canonical benchmark formatting and original-action wording.

`cases.json` contains 64 audited cases, with provenance in `AUTHORING.json` and
`AUTHORING_REQUESTS.json`. `FREEZE.json` pins cases and parameters before inference.
`fresh_cases.py` is the historical pre-freeze assembler; it refuses to regenerate
a frozen set. Cached replay needs no authoring service or private logs.

```sh
python -m sp_lense.research2.fresh_audit study/02_fresh_evaluation/run
```

For a deliberate new inference run, build the minimal input bundle with
`python -m sp_lense.research2.fresh_package release/fresh-evaluation.zip` and use
`evaluate.ipynb` on a T4 with the pinned dependencies. Update its archive digest
to the builder output; the notebook records the executed historical bundle.
The process watchdog caps model execution at 45 minutes. API credentials are
local only, and cached gate scores suffice for reproducing the recorded test.

Both answer-order views are retained. Identical control prompts share cached
scores across formats; they are not independent extra samples. Raw and guarded
outcomes are both retained in `run/`. Prior studies and the original manuscript
are unchanged. Prefect remains unavailable due to its existing Application
Control block; `run/comparison.json` is the prepared table artifact.

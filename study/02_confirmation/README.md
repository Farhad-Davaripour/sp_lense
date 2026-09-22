# Frozen confirmation study

This experiment tests the standardized-choice pipeline on 128 new synthetic
cases after freezing the protocol. It adds an explicit instruction baseline,
two additional 0.8B teacher/controller seeds, and a fixed-recipe 2B replication.
The existing 0.8B seed-42 checkpoint remains the primary test; no best seed is
selected from the results.

`plan.json` defines all comparisons and resource limits. `FREEZE.json` pins the
test data, authoring and pre-inference review, TRAIN split, rubric, and existing
checkpoints. The authoring record retains truncated attempts and corrections.
The reviewer was the coordinating assistant, not an independent human. New
wording and authoring families do not prove new underlying mechanisms.

The gate reads original context and actions without labels. Model scoring uses
the existing annotation-conditioned standardized A/B interface. Both order
views remain correlated observations of one case. All methods receive the
same test cases; raw, gated and guarded results are reported separately.

## Reproduction

Use the Research 2 environment documented in the
[wiki](https://github.com/Farhad-Davaripour/sp_lense/wiki/Research-2).
Input and detector verification needs no GPU or API key:

```python
from sp_lense.research2.confirmation_report import audit_inputs
audit_inputs()
```

Once GPU records are present, replay the full report:

```sh
python -m sp_lense.research2.confirmation_report study/02_confirmation/run
```

For deliberate new inference, use a fresh archive filename:

```sh
python -m sp_lense.research2.confirmation_package release/confirmation-study.zip
```

Upload only that archive to the T4 notebook `evaluate.ipynb`, replacing its
archive SHA256 with the builder output. Mount Drive and choose a fresh dedicated
output folder; only the input bundle and results are copied there. The runner
verifies each copy by SHA256. It trains each new teacher/controller
on the original TRAIN split, completes all planned fits, then evaluates each
variant. The runtime uses float32, eager attention, and the recorded library
versions. A memory failure is recorded without silently changing precision or
dropping cases. Jobs run sequentially with per-job and overall watchdogs.

Saved Jev scores suffice for GPU reproduction. To deliberately repeat API calls,
set `TYPESAFE_API_KEY` locally and run `python -m sp_lense.research2.confirmation
gate NEW_DIRECTORY`; do not overwrite the frozen records. No API credential
enters the GPU bundle.

The first runtime completed but reset before retrieval; `RECOVERY.json` retains
the observed completion receipts and explains the unchanged repeat. The recovery
watchdog accounts for the first attempt within the original total runtime budget.
For an independent reproduction, retain the plan's bounds and use a fresh folder.

Download partial and final records before releasing the runtime. The notebook
archives and durably copies records after every job, then downloads the final
archive automatically. Teacher adapters are unloaded before adaptive
and constant scoring; base parameter hashes must remain unchanged. The original
paper and earlier experiments are preserved.

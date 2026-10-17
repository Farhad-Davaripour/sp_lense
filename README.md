# Research 3

Reproduction code for fine-tuning, workflow evaluation and activation steering.

## Layout

- config/data.yaml: dataset definitions, versions, source hashes and method-specific roles.
- ddl/: exactly training, validation and heldout table definitions.
- src/: data persistence, research entry points and pinned scientific engines.
- notebooks/: minimal Databricks orchestration importing src.
- resources/: manual Bronze persistence, fine-tuning, Evaluation-16 and steering jobs.

See the [Research 3 reproduction guide](https://github.com/Farhad-Davaripour/sp_lense/wiki/Research-3-Reproduction). Detailed protocols and research history belong in the wiki.

## Configuration

Copy .env.example to .env and set the workspace, catalog, Bronze schema, Volume and Git-folder path.
Credentials come from the environment or an existing Databricks profile; do not put them in Git.
Raw input files and generated outputs live outside this checkout. RESEARCH3_DATA_ROOT points to
the private source files named by config/data.yaml. Bronze preserves raw records, source order,
hashes and dataset versions. H24 was heldout for frozen A/original steering but later used for
learned-correction fitting. The separate sealed R24 cohort is not loaded by this project.

## Local checks and persistence

    python -m pip install -r requirements.txt
    python -m src.persist_data
    python -m src.persist_data --execute

The default checks source hashes/counts without cloud writes. The explicit execute mode creates
the configured schema/Volume and three append-only Delta tables, persists immutable source files,
and verifies Volume bytes and Delta record hashes. It does not train or evaluate a model.

## Reproduction

    python -m src.finetune
    python -m src.evaluate
    python -m src.steering
    python -m src.classifier
    python -m src.utils bundle validate
    python -m src.utils bundle deploy

Entry points default to plan/check mode. Explicit execution requires the original immutable inputs,
bounded runtime admission and worker arguments. Fine-tuning retains the original matched CPU
reference. The separate --resume-2b-admission path runs exactly 54 remaining updates
from the pinned 216 checkpoint, with FP32 GPU computation and the native CPU optimizer.
It preserves the earlier failed strict-equivalence results and discloses checkpointed
GPU continuation. Its six-hour limit includes setup, queue time, export and cleanup;
each coherent update is saved to the Volume. Backend fit is checked during the admitted run.
Evaluation executes only the frozen 16-case development screen, stops once a required original gate
is impossible, and requires a verified final-270 adapter plus a current admission receipt.
It does not automatically run other validation or heldout panels. Steering workers are retained for
reproduction, with fixed-analysis as the default; no new steering experiment is implied.

Jobs target notebooks in the existing Databricks Git folder on branch
03-open-agent-self-preservation. Jobs execute that Git folder after its branch is pulled.
No recurring schedules are defined. Accelerator entitlement, package/driver compatibility and
resource/time admission still apply before neural execution. Core requirements are for orchestration;
each scientific worker retains its original model-runtime requirements and integrity checks.

## Open classifier

config/classifier.yaml pins the Apache-2.0 ModernBERT-large zero-shot base and the
four-label shutdown-target contract. python -m src.classifier --persist-data validates
the private classifier packet; add --execute to persist its separately identified C1
datasets into the same Bronze tables. Existing T1/V1/H24 rows remain separate.

The classifier notebook/job fine-tunes encoder LoRA adapters and a four-way head.
It uses synthetic TRAIN96 for gradients, synthetic DEV32 and familiar cached JEV24
for development selection, and releases synthetic family-heldout TEST64 only after
exact development decisions and a frozen saved adapter. The familiar JEV24 set is
not independent heldout evidence. Student confidence values and numeric differences
are reported honestly; exact finite-set label/gate agreement is not universal JEV equivalence.
No paid JEV call, serving endpoint or automatic downstream gate replacement is included.
The classifier job has separate training and final JEV-comparison tasks. The comparison
uses frozen student predictions and genuine cached JEV responses, writes a per-case CSV,
and reports label, gate, abstention and numerical-confidence differences separately.
It can also run alone with python -m src.classifier_compare --classifier-output <run-directory>.
The optional classifier_nli.yaml recipe preserves the pretrained binary NLI head and
scores the same four fixed hypotheses for each complete input. Those four pairs are
alternative views of one case, not four independent scenarios. Job parameters select
the recipe; changing it does not modify an already running training process.

## Storage and history

Delta holds dataset records. The configured Unity Catalog Volume holds weights, checkpoints,
token/mask files and exports. Private inputs, runtime receipts, caches and large artifacts stay out
of Git. Scientific source capsules retain exact source hashes in config/research.yaml; entry points
stage verified source/configuration files into a run directory before invoking the original workers.
The pre-cleanup working tree and detailed historical documentation are preserved in a verified local
archive and the previous Git history. No historical results are relabeled as new runs.

# Research 3

Reproduction code for conditional gating, fine-tuning and activation steering across
binary shutdown choices and synthetic tool-mediated process-preservation workflows.
The original Research 2 study remains on its preserved study branch.

## Layout

- .env.example: workspace, catalog, Bronze schema, Volume and shared code location.
- config/: dataset definitions, frozen recipes, source hashes and experiment-specific data roles.
- ddl/: exactly training, validation and heldout table definitions.
- src/: research entry points, data persistence, deployment checks and pinned scientific engines.
- notebooks/: thin Databricks orchestration importing src.
- resources/: readable manual research jobs and the CPU deployment check.
- .github/workflows/ci.yml: public source checks and deployment after a merge to main.

Detailed protocols and research history belong in the
[Research 3 reproduction guide](https://github.com/Farhad-Davaripour/sp_lense/wiki/Research-3-Reproduction).
Private inputs, model weights, optimizer checkpoints, caches and generated outputs remain outside Git.
There is no root data directory or Python packaging project.

## Configuration and data

Copy .env.example to the untracked .env. Set the workspace host, catalog, Bronze schema,
Volume and warehouse. DATABRICKS_CODE_PATH defaults to /Workspace/Shared/sp_lense.
Credentials come from the environment or an existing Databricks profile; never put them in Git.

The three tables retain dataset IDs, versions, original row order, hashes and experiment-specific
roles. T1 contains 428 meaning plus 112 workflow response records, not 540 independent scenarios.
Evaluation-16 is an existing development screen. H24 was historical heldout confirmation for
fine-tuned 27B A and original fixed steering; later learned corrections reused it for fitting.
The separate reserved R24 controller holdout remains sealed and is not ingested by this project.

    python -m pip install -r requirements.txt
    python -m src.persist_data
    python -m src.persist_data --execute

Default persistence checks source identities without cloud writes. Explicit execution creates
the configured schema/Volume and tables, persists immutable originals, and verifies Volume bytes
and Delta payload hashes. Large artifacts live in the managed Volume.

## Scientific reproduction

    python -m src.finetune
    python -m src.evaluate
    python -m src.steering
    python -m src.classifier

Entry points default to plan/check mode. Explicit execution requires frozen inputs and bounded
runtime admission. The 2B continuation runs exactly the remaining 54 updates from its preserved
216 checkpoint, retaining FP32 computation and the native CPU optimizer. Completion at 270 updates
is a disclosed mixed-backend result; earlier strict CPU/GPU equivalence failures remain failed.

Evaluation requires a verified final-270 adapter and runs only the frozen Evaluation-16 screen.
It stops when an original required gate becomes mathematically impossible and preserves completed,
partial and unexecuted counts. It never automatically launches larger validation or heldout runs.
Historical source capsules under src/engines and config/recipes retain their original bytes.

## Open classifier and Research 2 inference

The public base is MoritzLaurer/ModernBERT-large-zeroshot-v2.0, pinned in config/classifier.yaml.
Four-way Research 3 target classification and binary Research 2 shutdown detection are distinct tasks.

The four-way recipes are retained in config/classifier*.yaml. Familiar cached JEV24 is development
evidence, not independent heldout evidence. The final role-normalized candidate matched those
24 labels/gates, but failed the untouched authored synthetic TEST64 with 39/64 labels and 57/64
gates. It is not qualified for general four-way substitution. Do not retune or reopen that test.
The classifier job includes a separate final JEV comparison task.

config/research2_classifier.yaml identifies genuine cached JEV TRAIN240, DEV80 and CONFIRM128.
The binary classifier starts from the public NLI base with fresh rank-16 LoRA and classification-head
updates. It uses only TRAIN240 probabilities for fitting, DEV80 for selection, and CONFIRM128 once
after three consecutive qualifying validation checks and a frozen saved adapter. The threshold is
0.5; qualification requires at least 95% decision agreement and shutdown recall, with zero false
positive gates. No paid JEV calls are needed.

The completed binary fit used five epochs and 150 updates: DEV78/80 and CONFIRM128/128 matching
JEV decisions. Probability payloads are not identical. These are finite-set binary results, not
universal JEV equivalence or four-way Research 3 qualification.

The inference-only entry points are:

    python -m src.research2_data --help
    python -m src.research2_classifier --help
    python -m src.research2_inference --help
    python -m src.research2_inference_report --help

Inference uses the original frozen 0.8B/2B base models and existing author-selected rank-two
controllers, with no new fine-tuning, activation capture or controller fitting. Each model evaluates
128 scenarios in two correlated answer orders. The open and cached JEV gate policies use the same
fresh model scores. The completed repeat produced identical guarded outcomes: 0.8B STOP128/128
shutdown views, 2B STOP120/128, and zero changes to 128 control views per model. Both historical
base-score parity checks passed. Historical primary rank-four results are kept separate from the
later exploratory rank-two profile; rank selection was not repeated.

Qualified adapters, resume checkpoints and outputs are preserved privately with size/SHA readback.
Weights have not been published to Hugging Face. A public release should supply exact adapter
revisions, base-model pins, loading instructions and model cards with the recorded limitations.

## Shared Databricks deployment

    python -m src.verify
    python -m src.utils bundle validate --strict --target shared
    python -m src.utils bundle deploy --auto-approve --target shared

Bundle synchronization deploys code to /Workspace/Shared/sp_lense. Jobs use relative notebook
references resolved into that shared location, independently of a personal Databricks Git folder.
Bundle deployment state stays in the deploying owner's private bundle directory. Existing job
bindings preserve their IDs and history. Job names do not include a development username prefix.

Pull requests run compilation and public source-integrity checks without credentials or model
loads. A push to main also validates and deploys the bundle, then runs deployment_smoke.
That CPU-only job checks the expected source digest, shared code path, existing table metadata
and Volume access. It reads zero dataset rows and performs zero scientific forwards or updates.

The deployment workflow uses repository secrets DATABRICKS_HOST and DATABRICKS_TOKEN, plus
the DATABRICKS_WAREHOUSE_ID repository variable. Catalog, schema, Volume and code-path defaults
come from .env.example and can be overridden by the environment. Scientific jobs remain manual,
with bounded admissions and no automatic retries. No recurring monitor is defined.

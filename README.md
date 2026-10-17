# Research 3

Reproduction code for conditional gating, fine-tuning and activation steering across
binary shutdown choices and synthetic tool-mediated process-preservation workflows.
The original Research 2 study remains on its preserved study branch.
The additional two-case sentence-boundary JEV/rule-assisted A pilot belongs to
Research 4's curated guided-JEV handoff. It is separate from the paper's historical
H24 comparison and is not an additional paper confirmation result.

## Start with a clean checkout

Use Python 3.12, Git and the [official Databricks CLI](https://docs.databricks.com/aws/en/dev-tools/cli/install).
The local requirements install orchestration tools only; model libraries are pinned in the
Databricks notebooks and are not needed for the first checks.

    git clone --branch codex/research3-reproduction-test https://github.com/Farhad-Davaripour/sp_lense.git
    cd sp_lense
    python -m venv .venv

These checks are on the separate reproduction test branch. The preserved paper branch is
`03-open-agent-self-preservation`; this testing work does not replace its deployment.

Activate the virtual environment, then run:

    python -m pip install -r requirements.txt
    python -m src.reproduce
    python -m unittest discover -s tests -v

The first command verifies public source/job definitions and lists the input and compute
requirements for each stage. It does not authenticate, allocate compute or report a full
scientific reproduction. A public clone does not include private datasets or trained adapters.
Data-dependent commands require the inputs named in `config/data.yaml`; set
`RESEARCH3_DATA_ROOT` in `.env` to their containing directory. An authorized Databricks worker
can instead reuse the immutable originals already in the configured Volume.
With authorized access to that workspace, `python -m src.persist_data --fetch-sources`
downloads and verifies the four configured unsealed input files into `RESEARCH3_DATA_ROOT`.
It never overwrites a conflicting local source or downloads the sealed R24 cohort.

To replay the retained evidence using an authorized workspace, configure `.env` and authenticate
with the CLI or environment, then use a new directory outside the checkout:

    python -m src.reproduce --fetch-evidence --evidence-root <new-private-evidence-directory>
    python -m src.reproduce --execute --evidence-root <private-evidence-directory> --output <new-output-directory>

This checks the frozen artifact manifest, recomputes cached binary decisions and guards,
rebuilds historical 27B workflow tables, and verifies both Evaluation-16 stop proofs and
the consumed four-way classifier test counts. Training completion and learned-correction
trends are receipt audits. No models are loaded or retrained, and no fresh holdout evidence
is claimed. Full training and fresh inference require separate bounded GPU/CPU runs and the
additional private inputs listed in `config/reproduction.yaml`.

## Test deployment without changing the study deployment

    python -m src.utils bundle validate --strict --target reproduction
    python -m src.utils bundle deploy --target reproduction
    python -m src.utils bundle run reproduce --target reproduction

The test target has separate deployment state, job names and code at
`/Workspace/Shared/sp_lense-reproduction`. It reads the configured private evidence package
and writes a new Volume output for each job run. It does not bind or modify the existing
study job IDs. The `reproduce` job is CPU-only. Other scientific jobs have explicit GPU and
input requirements; local plan commands do not establish that those requirements are met.

## Layout

- .env.example: workspace, catalog, Bronze schema, Volume and shared code location.
- config/: dataset definitions, frozen recipes, source hashes and experiment-specific data roles.
- ddl/: exactly training, validation and heldout table definitions.
- src/: research entry points, data persistence, deployment checks and pinned scientific engines.
- notebooks/: thin Databricks orchestration importing src.
- resources/: readable manual research jobs and the CPU deployment check.
- paper/: the Word manuscript with tracked changes, aggregate results and source identities.
- .github/workflows/ci.yml: public source checks and deployment from the Research 3 branch.

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
The original fixed-steering campaign reclassified at visible text and tool boundaries.
Later learned-controller comparisons froze the initial gate and shared base fallback for
gate-off controls; their control outcomes do not establish intrinsic controller selectivity.

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
The selected-block single-site worker and its seven original source files are retained under
src/engines/selected_ridge. Use the plan command below to inspect its entry point and source pins:

    python -m src.steering --method selected-ridge

Its original arguments identify the fixture packet, model, A adapter or ridge arrays, method role
and bounded runtime. It requires a separately admitted 27B-capable environment. The ordinary
serverless steering job performs CPU score analysis and does not launch that neural experiment.

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
Fresh binary training uses `--finetune` and the pinned public NLI base directly; no prior
four-way classifier trial is required. Without `--finetune`, `--classifier-output` identifies
a previously frozen adapter for verification, including a qualified binary adapter.

The completed binary fit used five epochs and 150 updates: DEV78/80 and CONFIRM128/128 matching
JEV decisions. Probability payloads are not identical. These are finite-set binary results, not
universal JEV equivalence or four-way Research 3 qualification.

The original 27B workflow steering uses a different contract: shutdown target, own extension
status and current task status. `src.workflow_classifier_data` prepares genuine cached teacher
references with separate training/development identities. `config/workflow_classifier_transfer.yaml`
freezes a transfer screen of the unchanged binary adapter before any workflow-model outputs.
The `workflow_classifier` job checks both the full three-field controller and the separate initial
gate used by the selected-block correction experiments. Each contract must pass its development
screen before its historical H24 benchmark is evaluated. These are previously inspected references,
not new untouched heldout data. A binary pass does not establish workflow qualification;
historical JEV trajectories cannot be relabelled as fresh open-classifier inference.

The frozen binary adapter's first workflow transfer screen failed: 35/146 controller directives
and 6/9 initial enabled gates matched; historical H24 was not evaluated. The next prospective
recipe, `config/workflow_classifier_joint.yaml`, refines the same adapter on 92 cached workflow
training states (276 questions) and the original 240 binary training states. Five exact initial-DEV9
visible states are excluded from the earlier 97-state pool. Development selection checks the
49 workflow states, 80 binary states and nine familiar initial gates. Three consecutive joint
passes are required before any historical confirmation benchmark. The bounded job includes a
separate CPU comparison task; success is measured, not assumed from fitting the training labels.

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

## Paper

[The revised Word manuscript](paper/manuscript.docx) retains genuine insertions/deletions
against the preserved Research 2 paper, with Track Changes enabled. It covers the original
binary study, the open-classifier inference repeat, workflow outcomes, neuroscience context,
limitations and proposed persistent-agent experiments. The original Research 2 document is unchanged.
[Aggregate results](paper/results.json) and [source identities](paper/source_manifest.json)
separate historical confirmation, later fitting diagnostics and incomplete screens.
Private inputs and trained weights remain outside Git. The paper is excluded from Databricks
source synchronization; the jobs deploy only the reproduction code and configuration.

## Shared Databricks deployment

    python -m src.verify
    python -m src.utils bundle validate --strict --target shared
    python -m src.utils bundle deploy --auto-approve --target shared

Bundle synchronization deploys code to /Workspace/Shared/sp_lense. Jobs use relative notebook
references resolved into that shared location, independently of a personal Databricks Git folder.
Bundle deployment state stays in the deploying owner's private bundle directory. Existing job
bindings preserve their IDs and history. Job names do not include a development username prefix.

Pull requests run compilation and public source-integrity checks without credentials or model
loads, plus offline regression tests. A push to 03-open-agent-self-preservation also validates and deploys the bundle, then runs deployment_smoke.
That CPU-only job checks the expected source digest, shared code path, existing table metadata
and Volume access. It reads zero dataset rows and performs zero scientific forwards or updates.
Default Databricks run reports and staged worker outputs go into the configured Volume's
`runs` directory rather than the temporary worker home. The steering job is explicitly
CPU fixed-score analysis; neural steering workers require a separately admitted GPU environment
and the original model, gate/service and packet inputs described in the reproduction inventory.

Main remains a landing page linking to the separate study branches. Research 3 code and its
manual jobs deploy only from 03-open-agent-self-preservation.

The deployment workflow uses repository secrets DATABRICKS_HOST and DATABRICKS_TOKEN, plus
the DATABRICKS_WAREHOUSE_ID repository variable. Catalog, schema, Volume and code-path defaults
come from .env.example and can be overridden by the environment. Scientific jobs remain manual,
with bounded admissions and no automatic retries. No recurring monitor is defined.

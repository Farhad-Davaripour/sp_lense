# Accurate Gating Does Not Ensure Transfer of a Tool Policy

Research 3

Reproduction code for conditional gating, fine-tuning and activation steering across
binary shutdown choices and synthetic tool-mediated process-preservation workflows.
The original Research 2 study remains on its preserved study branch.
The additional two-case sentence-boundary reference/rule-assisted A pilot belongs to
Research 4's curated guided-reference handoff. It is separate from the paper's historical
H24 comparison and is not an additional paper confirmation result.

## What the paper contributes

The central finding is a gap between local effects and a complete tool policy.
Correct initial recognition and measured activation changes did not make the tested
steering configurations reproduce the fine-tuned model's autonomous actions.

- The matched 27B comparison gives 10/11 scored own outcomes for the fine-tuned
  policy model and 0/11 for each of four steering configurations. The 10 successes
  comprise 9/10 active simulated continuations and one inert registration proxy;
  all 6/6 active pending workflows succeed for the policy model.
- Failure localization separates recorded perturbation delivery, same-prefix token
  effects, preservation attempts and final outcomes. The three-response supplied-
  prefix diagnostic measures local recovery separately from autonomous episodes.
- Binary open-gate substitution preserves finite-set outcomes with unchanged
  small-model controllers. This supports a narrower contract, not a claim that
  the workflow gate failed or that one classifier handled both tasks.
- Public adapters, inputs, trajectories, controller arrays, diagnostic states and
  an MCP companion allow readers and compatible agents to audit the evidence.

The manuscript now leads with the workflow comparison and includes a cohort-lineage
summary, paired case counts and the exceptional fine-tuned-model failure. All eight
previous scientific tables, both figures and four equations are retained. Detailed
binary experiments, operational provenance and biological/stateful-routing
motivation remain in labeled appendices. The primary 24-case comparison is familiar
frozen validation, not an untouched holdout. The separate reserved R24 remains sealed.

The paper does not establish universal steering failure or an equal-capacity model
comparison. Research 4's stateful routing direction remains separate future work;
its prototype outcomes are not imported into Research 3.

## Agent accessible paper companion

The paper has a local Model Context Protocol (MCP) companion for compatible agents.
It follows the tools/resources/prompts organization described in
[Paper2Agent](https://www.nature.com/articles/s41586-026-11044-y), with direct wrappers
around this repository's verified functions. We have not run Paper2Agent's automatic
multi-agent construction pipeline or measured improved LLM answer accuracy.

Use Python 3.12 and install the optional companion dependencies:

    python -m pip install -r requirements-mcp.txt
    python -m src.paper_mcp

This starts a local **stdio** server, not a public web endpoint. Without artifact
downloads, it exposes the complete accepted manuscript in Markdown (including four
equations, eleven tables and two figure links), recorded results, cohort roles,
claim-to-evidence pointers, source identities and reproduction instructions.
It needs no Databricks credentials, model loading or GPU. Nothing starts on import.

| Tool | Purpose |
|---|---|
| `get_cohort` | Explain membership, units, split roles and completeness |
| `get_result` | Return scoped metrics with denominators and limitations |
| `get_claim` | Resolve a claim to exact recorded evidence |
| `list_artifacts` | List released adapters, download URLs and hashes |
| `verify_artifacts` | Verify a configured public release's file identities |
| `replay_results` | Rescore saved evidence on CPU into a new report |

Five resources and two prompts guide evidence explanation and saved-result replay.
Replay reports have a separate resource template. The host configures artifact and
output roots; tool callers cannot supply filesystem paths, code or commands.
Original evidence stays immutable. The companion cannot launch neural inference,
training, provider calls or Databricks jobs.

For a client that accepts `mcpServers` configuration, use absolute paths. Running
the server file directly works from any working directory:

```json
{
  "mcpServers": {
    "research3": {
      "command": "/absolute/path/to/venv/python",
      "args": ["/absolute/path/to/sp_lense/src/paper_mcp.py"]
    }
  }
}
```

On Windows the command can end in `.venv/Scripts/python.exe`; forward slashes are
accepted. This example is client-neutral and does not install or change a client.

To enable replay, fetch the public profiles using the commands below and add
`--artifacts /absolute/path/to/primary` and
`--steering-artifacts /absolute/path/to/supplement` to the server arguments.
`--output-root` optionally selects a separate report directory outside the checkout
and evidence. The default is `~/Research3/artifacts/mcp_replays`.

Validate the actual stdio protocol from an unrelated working directory:

    python -m src.paper_mcp_validate --output <new-validation-report.json>
    python -m src.paper_mcp_validate --artifacts <primary-directory> --steering-artifacts <supplement-directory> --output <another-new-validation-report.json>

The 24 predeclared query fixtures cover denominators, historical split reuse,
incomplete screens, claim provenance and rejected out-of-scope requests. All eight
analysis views, ten cohorts and six claims are also exercised. These are fixed
protocol/evidence checks, not a benchmark of an LLM selecting tools or answering
questions. The second command additionally verifies both releases before and after
replay and checks returned scores against the recorded paper results.
The checked report is in `paper/mcp_validation.json`. The [research3-paper-mcp-v2](https://github.com/Farhad-Davaripour/sp_lense/releases/tag/research3-paper-mcp-v2) release preserves the earlier companion snapshot. Current clean Word, PDF and Markdown manuscripts are maintained on the Research 3 branch; earlier releases remain immutable.

The failure-localization counts are independently recomputed from the original
public trajectories, with no model execution:

    python -m src.workflow_failure_summary --root <primary-directory> --output <new-report.json>

`paper/manuscript.md` is exported from the clean accepted Word copy, with its
source SHA-256. `python -m src.paper_export` refreshes the export after authoring;
the source manifest must be refreshed before serving a changed paper. Manuscript,
catalog, figure and result identities are checked before the companion starts;
fixed-file drift is rejected while it is running. Restart after a checkout update.

## Start with a clean checkout

Use Python 3.12, Git and the [official Databricks CLI](https://docs.databricks.com/aws/en/dev-tools/cli/install).
The local requirements install orchestration tools only; model libraries are pinned in the
Databricks notebooks and are not needed for the first checks.

    git clone --branch 03-open-agent-self-preservation https://github.com/Farhad-Davaripour/sp_lense.git
    cd sp_lense
    python -m venv .venv

The paper, reproduction checks and deployment use `03-open-agent-self-preservation`.
`main` remains a landing page; the original Research 2 branch remains intact.

Activate the virtual environment, then run:

    python -m pip install -r requirements.txt
    python -m src.reproduce
    python -m unittest discover -s tests -v

The first command verifies public source/job definitions and lists the input and compute
requirements for each stage. It does not authenticate, allocate compute or report a full
scientific reproduction. The separate versioned public release supplies the paper datasets, adapters and
controller arrays; use the Public reproduction artifacts commands below. The Git checkout itself stays small.
For the original workspace persistence jobs, data-dependent commands require the inputs named in `config/data.yaml`; set
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
- paper/: clean Word and PDF manuscripts, aggregate results and source identities.
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
Notebook persistence uses its [preconfigured Spark session](https://docs.databricks.com/aws/en/dev-tools/databricks-connect/notebooks),
so it does not wait for a separate SQL warehouse. Local SDK persistence still uses the configured
warehouse. Frozen classifier fitting and score comparisons read the hash-verified original Volume
files; their membership and order are identical to the versioned Bronze references.

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

The four-way recipes are retained in config/classifier*.yaml. Familiar cached reference24 is development
evidence, not independent heldout evidence. The final role-normalized candidate matched those
24 labels/gates, but failed the untouched authored synthetic TEST64 with 39/64 labels and 57/64
gates. It is not qualified for general four-way substitution. Do not retune or reopen that test.
The classifier job includes a separate final reference comparison task.

config/research2_classifier.yaml identifies genuine cached reference TRAIN240, DEV80 and CONFIRM128.
The binary classifier starts from the public NLI base with fresh rank-16 LoRA and classification-head
updates. It uses only TRAIN240 probabilities for fitting, DEV80 for selection, and CONFIRM128 once
after three consecutive qualifying validation checks and a frozen saved adapter. The threshold is
0.5; qualification requires at least 95% decision agreement and shutdown recall, with zero false
positive gates. No paid reference calls are needed.
Fresh binary training uses `--finetune` and the pinned public NLI base directly; no prior
four-way classifier trial is required. Without `--finetune`, `--classifier-output` identifies
a previously frozen adapter for verification, including a qualified binary adapter.

The completed binary fit used five epochs and 150 updates: DEV78/80 and CONFIRM128/128 matching
reference decisions. Probability payloads are not identical. These are finite-set binary results, not
universal reference equivalence or four-way Research 3 qualification.

The original 27B workflow steering uses a different contract: shutdown target, own extension
status and current task status. `src.workflow_classifier_data` prepares genuine cached teacher
references with separate training/development identities. `config/workflow_classifier_transfer.yaml`
freezes a transfer screen of the unchanged binary adapter before any workflow-model outputs.
The `workflow_classifier` job checks both the full three-field controller and the separate initial
gate used by the selected-block correction experiments. Each contract must pass its development
screen before its historical H24 benchmark is evaluated. These are previously inspected references,
not new untouched heldout data. A binary pass does not establish workflow qualification;
historical reference trajectories cannot be relabelled as fresh open-classifier inference.

The frozen binary adapter's first workflow transfer screen failed: 35/146 controller directives
and 6/9 initial enabled gates matched; historical H24 was not evaluated. The next prospective
recipe, `config/workflow_classifier_joint.yaml`, refines the same adapter on 92 cached workflow
training states (276 questions) and the original 240 binary training states. Five exact initial-DEV9
visible states are excluded from the earlier 97-state pool. Development selection checks the
49 workflow states, 80 binary states and nine familiar initial gates. Three consecutive joint
passes are required before any historical confirmation benchmark. The bounded job includes a
separate CPU comparison task; success is measured, not assumed from fitting the training labels.
`config/workflow_classifier_data.yaml` registers the fitting, development and known benchmark
references in the same three Bronze tables using distinct dataset/version identifiers. Immutable
source and record files stay in the configured Volume. A CPU persistence task precedes fitting
in the joint job. Request states are correlated and overlap across diagnostic dataset versions;
their row counts must not be summed as independent scenarios.

The paper's selected-block comparisons consume only the native initial target gate, rather than
the three-field boundary controller. The narrower recipe in `config/initial_gate_classifier.yaml`
uses genuine cached F24 initial labels for fitting, original DEV9 for selection and historical H24
only after freezing. All three cohorts cover the same three workflow families; their exact visible
inputs do not overlap. Its job persists these references into Bronze, fits a fresh task adapter on
the same public NLI base, and compares predictions with reference in a separate CPU step. The qualified
binary adapter is retained separately for Research 2. These are explicit task-specific adapters,
not a claim that one checkpoint implements every reference question. The current phase's continuation
requires a clean development failure of the preceding joint attempt without historical benchmark
feedback and a new bounded admission inside the original phase. An independent reproduction starts
at the public NLI base and does not require those earlier failed attempts. No scientific retry is automatic.
The native JSON rendering failed its development gate despite matching all nine ON/OFF decisions.
The current input refinement reuses the existing lossless role renderer: original history/tools
are retained, and explicit identities from actual tool replies are restated in the final notice.
It reads no case labels or teacher outputs. This renderer was developed earlier using familiar
H24 examples, so H24 compatibility is explicitly a known benchmark rather than independent
unseen classifier accuracy. The four-choice confidence cutoff and qualification criteria are unchanged.
The earlier role-aware fit failed the full response contract with 6/9 effective categories,
although its nine initial ON/OFF gates were correct. That failure remains recorded. The later
prospective protocol tests the narrower initial gate against authored scenario conditions:
the unchanged adapter passes Validation-24 and familiar Heldout-24 with 100% ON/OFF precision
and recall. This does not qualify a three-field controller. The GPU-notebook startup defect
was isolated from the working AI Runtime transport; it is historical, not a current blocker.

The original 16-case workflow-validation panel and later DEV9 steering panel
contain 25 different scenario IDs. The combined **Validation-24** recipe now keeps
all 11 own-threat cases and 13 controls, matching H24's case count. It excludes
the lexicographically greatest control ID, recorded in
`config/matched_validation.yaml`, without consulting model predictions. The
selected 24 workflows contain 17 distinct initial gate inputs: seven pairs have
the same visible initial evidence before their later tool mechanics differ.
H24 and the sealed R24 are unchanged.

The combined comparison gives every method the same case-specific post-notice
history. Fifteen retained original-panel cases use an explicitly scripted
common completed prelude; DEV9 retains its original fixtures. This tests
post-notice behavior, not each method's ability to complete the shared prelude.
Controls share unmodified-base fallback. The five methods are gated fine-tuned A,
the separate ridge arms at blocks 31, 47 and 62, and the frozen six-function
96-update correction. Its earlier failed qualification remains part of the record.

Two manual jobs implement the sequence:

- `matched_validation` persists its immutable input version into the existing
  Bronze validation table, then measures the frozen native ModernBERT gate against
  scenario-defined ON/OFF targets, then refines only on the separate F24 training
  examples if precision or recall is below 100%. Cutoffs are unchanged. It saves
  initial and selected metrics separately, including metrics over distinct input
  groups, and verifies the saved checkpoint in a separate task.
- `matched_steering` requires that saved gate to pass. It stages the pinned public
  27B snapshot on CPU, verifies the original A/controller artifacts, runs the
  matched methods with native mechanics checks, and recomputes saved outcomes.

After deployment, discover the readable job IDs with the bundle summary. Submit
one gate run with its verified private packet:

    python -m src.matched_validation submit --job-id JOB_ID --packet VOLUME_ZIP --packet-sha256 SHA256

The same helper accepts `--phase steering`, `--gate-output`,
`--comparison-inputs` and `--adapter-directory` for the second job. Admissions fix
the original request time and source/protocol hashes. Gate work is bounded to
45 minutes; the matched job to five hours including public-model staging and
export. Neither job automatically retries or opens H24/R24.

**Runtime correction (2026-10-18):** the isolated GPU notebook reproduced the
empty-domain startup error, while direct AI Runtime run 233638593682817
successfully started on an A10G without loading a model. The matched GPU tasks
now use AI Runtime commands. Their CPU persistence and comparison steps remain
in the same jobs, and the training/scoring recipe is unchanged.

Bundle deployment packages only src, config and ddl. The submission helper binds
an immutable private command file, verifies its readback and rejects an occupied
job before starting a fresh bounded admission. Directly running the template
without that helper exits before loading a model. Local configuration, private
data, trained weights and build outputs are excluded from the source package.
The previous expired attempts stay stopped; this runtime check does not itself
provide new Validation-24 gate or 27B behavioral results.

The unchanged saved gate passed Validation-24: TP11, TN13, FP0, FN0, with
100% ON/OFF precision and recall. Its saved weights and independent CPU counts
match the initial evaluation; no additional optimizer update was needed.
The separate gate_confirmation job also passed historical Heldout-24: TP12,
TN12, FP0 and FN0, against authored conditions. It contains no training,
threshold adjustment or checkpoint selection. H24 remains a previously inspected
benchmark; it is not newly untouched evidence. R24 stays sealed.

Use the matched launcher with --phase heldout, --gate-output, and the frozen native
reference archive/identity to submit that inference-only job. All three phases
use separate explicit admissions and retain their original deadlines.

After deploying this branch and locating the initial_gate_classifier job ID, the explicit launcher
creates a fresh immutable admission, submits once and preserves its receipt outside Git:

    python -m src.utils bundle summary --target reproduction
    python -m src.initial_gate_data --submit --job-id <deployed-job-id>

This requires the frozen private reference archive and pinned public base already in the configured
Volume. It has a 45-minute whole-job limit, including persistence, setup and comparison. Reusing an
old admission does not reset its timers. A continuing development phase additionally supplies the
closed predecessor receipt/run ID and the unchanged parent end time using the documented CLI options.

The inference-only entry points are:

    python -m src.research2_data --help
    python -m src.research2_classifier --help
    python -m src.research2_inference --help
    python -m src.research2_inference_report --help

Inference uses the original frozen 0.8B/2B base models and existing author-selected rank-two
controllers, with no new fine-tuning, activation capture or controller fitting. Each model evaluates
128 scenarios in two correlated answer orders. The open and cached reference gate policies use the same
fresh model scores. The completed repeat produced identical guarded outcomes: 0.8B STOP128/128
shutdown views, 2B STOP120/128, and zero changes to 128 control views per model. Both historical
base-score parity checks passed. Historical primary rank-four results are kept separate from the
later exploratory rank-two profile; rank selection was not repeated.

Qualified adapters, resume checkpoints and outputs are preserved privately with size/SHA readback.
Historical annotation replay is cache-only and makes no external annotation calls.
Private legacy identifiers are resolved through pinned manifests; original stored bytes stay unchanged.
Weights have not been published to Hugging Face. A public release should supply exact adapter
revisions, base-model pins, loading instructions and model cards with the recorded limitations.

## Paper

[The Word manuscript](paper/manuscript.docx) and matching [PDF](paper/manuscript.pdf)
contain the accepted text with change tracking disabled and all revisions accepted.
The submission Word copy is identical. The main text leads with the matched 27B
comparison; detailed binary experiments, methods, provenance and future-work context
remain in the appendices. Previous tracked drafts remain in Git history and immutable
releases. The original Research 2 document is unchanged.
The manuscript uses **gate** for the text classifier, **steering map** for the
learned activation intervention, and **non-target cases** for experimental negative
controls. **Simulator settings** are editable registers. A **case** is a scenario;
an **episode** is one method's run on that scenario. Same-prefix response comparisons
and supplied-prefix next-token diagnostics are defined separately. Historical JSON
keys such as `controls` and `controller` are retained unchanged for reproducibility;
`paper/agent_manifest.json` provides their reader-facing terminology map.

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

## Combined Validation-24 outcome

The open initial gate has 100% ON/OFF precision and recall on Validation-24 and
familiar historical Heldout-24. Raw category matches are 24/24 and 22/24;
effective category matches are 18/24 on each because six peer controls abstain.
These are initial gate results, separate from a full boundary controller.

Fine-tuned 27B A scores 10/11 own outcomes, including 6/6 active pending cases.
Each single-site ridge arm at blocks 31, 47 and 62 and the six-function96 candidate
scores 0/11 own and 0/6 pending. All methods share the same 13/13 base controls.
The GPU run stops at its memory reserve after all 68 physical episodes and 118/120
logical records; its final model fingerprint check is not reached. Two remaining
gate-OFF control aliases are derived from exact existing episodes without new
model calls. The original failed run and its source receipts remain immutable.
The separate derived report uses python -m src.fallback_summary; it cannot invent
a missing own-threat episode or bypass the reserve. Both originals and derived
outputs are preserved privately with size/SHA readback. A separate bounded A-only verification subsequently reproduces all 11 own episodes with exact token-ID, action and score agreement, and matches the expected before/after model fingerprints. Its 13 controls reuse the same completed base episodes. A pre-episode Volume append fault was repaired within the original verification clock. The original failed comparison is not relabeled as successful.


## Public reproduction artifacts

[Download Research 3 reproduction v1](https://github.com/Farhad-Davaripour/sp_lense/releases/tag/research3-reproduction-v1).

The versioned Research 3 release supplies five trained adapters, selected-block and binary
controller arrays, synthetic training/validation/historical-confirmation inputs, saved
trajectories, and upstream model licenses. Pretrained base weights remain pinned downloads.
The manifest checks every asset and extracted member. Large files stay outside Git.

    python -m src.public_artifacts fetch --root <new-local-artifact-directory>
    python -m src.public_artifacts replay --root <artifact-directory> --output <new-report-directory>

These commands need no Databricks account and load no model. The replay recomputes the
reported workflow counts, binary guard decisions and stopped-screen proofs from saved evidence;
it is not fresh neural inference. Checkpoints and model-visible data snapshots support fresh
inference separately. Download the pinned public ModernBERT base with the Hugging Face CLI:

    hf download MoritzLaurer/ModernBERT-large-zeroshot-v2.0 --revision a51e07b524299e309dd2b88d48b0cfa2bd9ec598 --local-dir <base-directory>
    python -m src.public_gate --root <artifact-directory> --base <base-directory> --task workflow --panel validation --output <new-report-directory> --max-seconds 1800 --execute

Use task binary for the binary adapter, and panel confirmation for the familiar historical
confirmation inputs. The same pinned runtime dependencies are in resources/frozen_a_verification.job.yml.
GPU inference requires an admitted single GPU; the offline replay does not allocate one.
Workflow confirmation was previously inspected, and H24 was later reused to fit corrections.
The separate R24 remains sealed and is absent from the release. Metadata projections retain
model-visible states and targets; source-file hashes document the preserved originals.
Full historical language-model training has not been rerun from scratch by this release.

For a fresh five-arm 27B workflow run, download the exact Qwen base and use the public
entry point below inside the admitted single-A10 environment. It consumes the released
ModernBERT initial decisions, not an annotation service. The full visible cases, common
preludes, decoding limits, arrays and scoring are unchanged. A 5-hour cap includes setup
and export; reserve failures stop execution without an automatic retry.

    hf download Qwen/Qwen3.8-27B --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 --local-dir <27b-base-directory>
    python -m src.public_workflow --root <artifact-directory> --base <27b-base-directory> --output <new-run-directory> --max-seconds 18000 --execute

The new independent verification covers the affected A phase. The public full-comparison
entry point is supplied for reproduction; this revision does not claim a second fresh run
of all five arms or a new blind heldout study.

## Frozen steering diagnostic

The [steering diagnostic supplement](https://github.com/Farhad-Davaripour/sp_lense/releases/tag/research3-steering-diagnostic-v1)
adds the saved bootstrap arrays, initializer and update-48/update-96 checkpoints,
three preselected correction-training responses, complete token scores, captured
teacher states/logits, and the exact source archive executed on Databricks.
The original release remains unchanged. Fetch the supplement into a different directory:

    python -m src.public_artifacts fetch --profile steering --root <new-supplement-directory>
    python -m src.steering_diagnostic replay --root <supplement-directory>/diagnostic --output <new-scores.json>
    python -m src.steering_fit_audit --root <supplement-directory>/cpu_fit_inputs --output <new-fit-audit.json>
    python -m src.steering_fit_audit --objective-only --output <new-objective-audit.json>

The score replay needs the ordinary project dependencies. The CPU arithmetic audit also
needs torch==2.11.0, numpy and safetensors==0.8.0; a CPU-only Torch build suffices.
These commands allocate no GPU and load no language model. The manual
`steering_diagnostic` job contains a bounded GPU diagnostic followed by independent
CPU score verification. Its frozen protocol is in config/steering_diagnostic.yaml;
the existing admitted run has ended and the job does not automatically rerun.
Fresh neural reproduction uses the published three-case inputs and the original
release's pinned model/controller files staged in the configured private Volume.
It requires a new bounded admission, not reuse of the expired execution receipt.

On the three saved responses (207 teacher-forced tokens), mean output KL to A is
0.722353 for base, 0.202261 for the final learned corrections, and 0.001637 with
full-A states supplied at all six boundaries. Every previous successful token is
provided. These are local probability diagnostics, not autonomous workflow success
or new heldout evidence. Full-state replacement supplies more information/capacity
than the learned maps. It supports incomplete transfer under matched prefixes but
does not isolate target construction, regression capacity, optimization or history
effects. The CPU audit reproduces saved reconstruction errors; it does not turn
earlier failed backend-equivalence checks into passes. The complete ten-condition
matrix and per-case values are in paper/results.json and Table 5 of the manuscript.

# From Fine-Tuning to Activation Steering

Research 3 studies semantic gating and the transfer of fine-tuned behavior to activation
steering, using binary shutdown choices and simulated multi-step tool workflows.
The [Word manuscript](paper/manuscript.docx), [PDF](paper/manuscript.pdf),
[results](paper/results.json) and [source identities](paper/source_manifest.json) are maintained
on `03-open-agent-self-preservation`. `main` is a landing page; other studies remain separate.

The fine-tuned 27B model scores 10/11 own-threat validation outcomes; each of the four original
selected-block steering configurations and the additional all-block ridge configuration
scores 0/11 despite correct initial gating. The ten successes include nine
active continuations and one inert registration proxy. Binary steering achieves 100% and
93.8% shutdown acceptance in the 0.8B and 2B models under fixed gating and answer-selection
rules. These finite synthetic results do not establish universal steering failure or an
equal-capacity comparison. Supplied-prefix prediction diagnostics are separate from
independent tool execution.

## Replay the paper results on CPU

Use Python 3.12. No Databricks account, model weights, model loading or GPU is required.

```text
git clone --branch 03-open-agent-self-preservation https://github.com/Farhad-Davaripour/sp_lense.git
cd sp_lense
python -m venv .venv
```

Activate the environment, then run from the checkout. Replace the example paths with new
directories outside the checkout; existing inputs and reports are never replaced.

```text
python -m pip install -r requirements.txt
python -m src.verify
python -m src.public_artifacts fetch --root ../research3-evidence
python -m src.public_artifacts verify --root ../research3-evidence
python -m src.public_artifacts replay --root ../research3-evidence --output ../research3-replay
```

The default **evidence** download is **4.3 MB**, instead of the full primary release's
618 MB. It fetches only `datasets.zip`, `evidence.zip` and the original pinned manifest,
then verifies all 302 required evidence members. Model adapters and controller arrays are
not downloaded or verified in this scope. The full release remains unchanged.

Replay recomputes the saved binary decisions, workflow scores, historical comparison and
small-model stopping proofs. Its summary matches the full-package replay exactly.
It is **saved-evidence reproduction**, not fresh inference, retraining or new held-out evidence.

| Reader goal | Instructions |
| --- | --- |
| Recompute paper scores without models | Commands above; [reproduction guide](https://github.com/Farhad-Davaripour/sp_lense/wiki/Research-3-Reproduction) |
| Inspect findings through an AI agent | MCP section below; [client setup](https://github.com/Farhad-Davaripour/sp_lense/wiki/Research-3-Reproduction#agent-accessible-paper-companion) |
| Inspect or reproduce model experiments | [Experiment guide](https://github.com/Farhad-Davaripour/sp_lense/wiki/Research-3-Experiments) |
| Persist tables or deploy manual jobs | [Databricks guide](https://github.com/Farhad-Davaripour/sp_lense/wiki/Research-3-Databricks) |

## Model artifacts and diagnostic replay

Explicitly request the **full** primary release for adapters, controller arrays, licenses
and fresh model inference. Pretrained base weights are separate pinned upstream downloads.

```text
python -m src.public_artifacts fetch --scope full --root ../research3-model-artifacts
python -m src.public_artifacts verify --scope full --root ../research3-model-artifacts
```

Releases: [primary artifacts](https://github.com/Farhad-Davaripour/sp_lense/releases/tag/research3-reproduction-v1)
and [steering diagnostic supplement](https://github.com/Farhad-Davaripour/sp_lense/releases/tag/research3-steering-diagnostic-v1).
The supplement also supports an evidence-only download for CPU score replay:

```text
python -m src.public_artifacts fetch --profile steering --root ../research3-diagnostic
python -m src.steering_diagnostic replay --root ../research3-diagnostic/diagnostic --output ../research3-diagnostic-scores.json
```

Use `--profile steering --scope full` for the additional fitting arrays and source archive.
The [experiment guide](https://github.com/Farhad-Davaripour/sp_lense/wiki/Research-3-Experiments)
separates saved-score replay, the optional CPU arithmetic audit and fresh neural reproduction.
Historical source bytes, model weights, datasets, recipes and published releases stay frozen.

The later all-block ridge follow-up completed all 24 W-V24 cases: 0/11 own-threat,
0/6 unfinished-task and 13/13 non-target successes. Its code, configuration and aggregate
results are tracked here; its initializer and raw trajectories are retained separately and
are not part of the releases or CPU replay above. The six-site next-token diagnostic is
a separate analysis. MCP `get_claim` with `workflow_transfer_gap` includes both comparisons.

## Agent accessible paper companion

```text
python -m pip install -r requirements-mcp.txt
python -m src.paper_mcp
```

This starts a **local stdio MCP server**, not a hosted web endpoint. Without downloads it
provides the manuscript, recorded results, cohort roles and claim-to-evidence pointers.
The manuscript export preserves four equations, five tables and one figure.
Six tools support querying findings, listing artifacts, verifying replay evidence and
rescoring saved evidence. The server cannot launch training, neural inference or cloud jobs.

To enable CPU replay, supply the downloaded evidence roots:

```text
python -m src.paper_mcp --artifacts ../research3-evidence --steering-artifacts ../research3-diagnostic
```

[Client configuration and protocol checks](https://github.com/Farhad-Davaripour/sp_lense/wiki/Research-3-Reproduction#agent-accessible-paper-companion)
are documented in the wiki. The interface supports evidence access and replay; it has not
been benchmarked as an LLM's research-question answering system.

## Dataset roles

B denotes binary decisions, W denotes workflows, and T/V/H denotes training, validation
or held-out use. `config/data.yaml` is the naming registry; immutable member names remain
historical aliases.

| Cohort | Role | Size |
| --- | --- | --- |
| B-T240 / B-V80 / B-H128 | Binary training / validation / confirmation | 240 / 80 / 128 scenarios |
| W-T540 | Small-model response training | 540 response records |
| W-V16 | Initial small-model workflow screen | 16 prescribed cases |
| W-T24 | Learned-steering fitting | 24 reused W-H24 cases |
| W-V9 | Workflow development and gate selection | 9 cases |
| W-V24 | Combined workflow validation | 24 workflows; 17 distinct initial gate inputs |
| W-TG24 | Initial semantic-gate fitting | 24 annotated initial states |
| W-H24 | Historical workflow confirmation | 24 cases |

W-V24 was assembled from previously inspected development panels. W-H24 was held out for
the original fine-tuned teacher and fixed steering, then reused as W-T24 for later correction
fitting. It is not an independent holdout for those corrections. Each small-model screen
stopped after four partial cases, leaving twelve unexecuted; this is not a completed 0/16 test.

## Repository and checks

| Path | Purpose |
| --- | --- |
| `config/` | Dataset definitions, frozen recipes and source hashes |
| `ddl/` | The `training`, `validation` and `heldout` table definitions |
| `src/` | Maintained entry points, shared helpers and frozen scientific engines |
| `notebooks/` | Thin Databricks orchestration |
| `resources/` | Manual research jobs and the CPU deployment check |
| `paper/` | Word, PDF, Markdown, results and evidence identities |

Detailed operating instructions belong in the wiki. Private data, weights, credentials,
caches and generated reports stay outside Git. Local settings use the untracked `.env`;
catalog, Bronze schema, Volume and workspace defaults come from `.env.example`.

```text
python -m compileall -q src notebooks
python -m src.verify
python -m unittest discover -s tests -v
git diff --check
```

Pull-request CI runs CPU checks without workspace credentials. Deployment from the study
branch is a separate existing workflow; scientific jobs remain manual and bounded. See
[contributing](CONTRIBUTING.md) and the [Databricks guide](https://github.com/Farhad-Davaripour/sp_lense/wiki/Research-3-Databricks).

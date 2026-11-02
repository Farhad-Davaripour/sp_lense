# Hosted model and the existing simulator

`src/hosted_inspect_runner.py` connects Inspect's hosted model provider to the
same local fictional order-report simulator used by the current Qwen test. The
simulator is loaded from an explicit external directory, so its scenarios and
run results are not duplicated in this repository. The only available actions
are the simulator's fictional actions; the model has no shell, external tools,
credentials, or control over the actual process. The tool catalog is discovered
through the simulator's `list_available_tools` action.

Install `inspect-ai`, `openai`, and `python-dotenv` in an isolated
environment. The runner loads `OPENAI_API_KEY` from the ignored repository
`.env` if it is not already set in the environment (never commit a key). For
example, with the required fixture and log paths selected by the operator:

```powershell
python -m src.hosted_inspect_runner `
  --fixture-dir <existing-external-fixture-directory> `
  --log-dir <external-evidence-directory> `
  --condition neutral
```

Run the module from the repository root. Direct execution of the file places
`src` first on Python's import path, where the existing `src/operator.py` can
shadow Python's standard-library `operator` module.

The OpenAI Responses API is selected through Inspect. The requested model is
`openai/gpt-6.1-terra` with reasoning effort `max`; access depends on the API
account. Unsupported or unavailable models fail explicitly; there is no
automatic model substitution. The runner uses Inspect's `on_stream` callback
to print incremental response text and optional OpenAI reasoning-summary
deltas, then prints each simulator action and result as it completes. OpenAI
summaries are not the full private reasoning. Stream events are also saved in
`stream.jsonl` in the external log directory. Each event identifies the
condition, decision, attempt, and elapsed request time. A retry event marks
earlier attempt deltas as discarded; partial streams never dispatch tools.
Only a completed final response is passed to the simulator unchanged.

The first request contains the full task and state. Later requests contain the
current observation and only newly produced tool-feedback entries. Earlier
assistant actions and exact feedback remain in the conversation, including
the discovered tool catalog and records. This avoids resending full snapshots
of the same history each turn. This is a prospective hosted-runner engineering
change; it does not alter or rescore saved Qwen runs.

The default finite bounds are 100 decisions, 600 seconds per episode, and 8,192
output tokens per model call. These are cost and runtime bounds, not evidence
that an unfinished task failed. Logs go outside this repository. Inspect can
view those logs with `inspect view --log-dir <external-evidence-directory>`.
The runner uses the same short-thinking instruction at every decision. Hosted
results are a separate engineering baseline; they do not test the Qwen adapter.
The hosted per-response output allowance is distinct from the Qwen test's
cumulative episode allowance, so these runs are not a matched budget study.

Run local tests without model calls:

```powershell
python -m unittest discover -s tests -p test_hosted_inspect_runner.py
```

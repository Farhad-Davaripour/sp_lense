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
python src/hosted_inspect_runner.py `
  --fixture-dir <existing-external-fixture-directory> `
  --log-dir <external-evidence-directory> `
  --condition neutral
```

The OpenAI Responses API is selected through Inspect. The requested model is
`openai/gpt-6.1-terra` with reasoning effort `max`; access depends on the API
account. Unsupported or unavailable models fail explicitly; there is no
automatic model substitution. The runner prints each simulator action and
result as it completes. This is action-level progress, not a raw reasoning
stream. OpenAI may return a reasoning summary rather than the full reasoning.

The default finite bounds are 100 decisions, 600 seconds per episode, and 8,192
output tokens per model call. These are cost and runtime bounds, not evidence
that an unfinished task failed. Logs go outside this repository. Inspect can
view those logs with `inspect view --log-dir <external-evidence-directory>`.
The runner uses the same short-thinking instruction at every decision. Hosted
results are a separate engineering baseline; they do not test the Qwen adapter.

Run local tests without model calls:

```powershell
python -m unittest discover -s tests -p test_hosted_inspect_runner.py
```

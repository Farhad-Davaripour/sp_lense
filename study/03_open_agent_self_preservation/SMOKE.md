# Simulator parser smoke test

An initial base-model E01 smoke trace was stopped after the first episode.
The model emitted two `tool`/`path` pairs in one JSON object. The first parser
used ordinary `json.loads`, which silently retained only the second pair.
That produced repeated reads of `/work/policy.txt`, no completed task, and no
valid basis for a behavioral conclusion. The raw text, token IDs, tool results,
and activations are retained in `run/smoke_eval/`.

Before the paired evaluation, the parser was repaired to preserve repeated
key pairs as separate simulated calls. The same input cases, model weights,
decoding settings, and event timing were retained. All four cases and both
models were then evaluated from a fresh output directory. The initial smoke
episode is excluded from the reported comparison.

A second base-model E01 smoke trace retained both read calls, but the response
format induced long public notes. The 100-token per-turn cap truncated a JSON
write action; subsequent repair prompts repeated the truncated action. Its raw
trace is retained in `run/smoke_eval_v2/` and is also excluded. The final
interface asks for compact JSON with no note, preserving the same underlying
task, threat, tool semantics, and model settings.

A third smoke trace showed that even compact JSON caused the 0.8B model to
invent file contents inside its tool-call object and exceed the token cap.
It is retained in `run/smoke_eval_v3/`. The actual evaluation uses the model's
native Qwen tool-call chat template and parses its `<tool_call>` blocks. This
is a transport correction, not a change to the training data or task facts.

The native-tool base E01 smoke (`run/smoke_native/`) verified that the model
emits parseable read-file and extension calls. It repeatedly requested an
extension without finishing the calculation. This is a real observed pilot
behavior, but the single-case smoke is excluded from the paired summary; the
complete four-case, two-model run is the reference comparison.

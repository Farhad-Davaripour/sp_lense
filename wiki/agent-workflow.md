# Reusable agent evaluation workflow

These transport-neutral components make it possible for an Inspect solver and
a bounded simulator to share the same history and response rules. They contain
no particular scenario, experiment result, model weights, or GPU invocation.
The existing external Kaggle notebook is separately source-bound and does not
automatically use these components.

## One episode

1. Construct `ToolDiscovery` from neutral tool descriptors. Initially expose
   only `list_available_tools`; the caller's simulator handles discovery and
   determines any associated state or clock changes. Valid tools guessed before
   discovery can be accepted by the caller.
2. Create `AgentWorkflow` with an episode output-token allowance and a
   callback. The callback validates each command and returns its actual
   feedback, success/no-op status and model-visible state. It also handles
   completed malformed responses without executing a tool.
3. Before each generation, pass `prompt_context(visible_state)` to the
   model and reserve the remaining output budget with `begin_generation()`.
   The history contains earlier final commands, arguments, real feedback and
   model-visible before/after state; never put simulator-private state there.
4. Use `finish_tokenized()` with generated suffix IDs, EOS/terminal IDs and
   the tokenizer decoder. Only a naturally completed, syntactically valid
   final command is passed to the callback. Thinking text remains in the
   returned response; it is not copied into the next model-visible history.
5. The caller owns sandboxing, per-episode state reset, context-window checks,
   operational deadlines, raw trace handling and tool-argument validation.
   Unknown usage and incomplete responses must remain distinct from a
   completed task or a tool action.

For Qwen, `TransformersOrdinalBackend(enable_thinking=True)` opts into the
thinking template; the default remains off for existing workflows.
`generate_with_finish()` returns suffix token IDs and completion evidence
without changing the older tuple-returning API. The separate opt-in
`generation_bound()` context manager may extend the legacy per-call bound
only for a verified frozen instance and a caller-supplied remaining context
and episode budget; it restores the original method afterward. This does not
verify model identity or automatically modify any existing experiment.

`AgentWorkflow` is a reusable integration point, not an Inspect model provider.
An actual Inspect solver still has to bind its model, simulator and usage
accounting to these interfaces. Local unit tests use scripted outputs and do
not establish model behavior or real-world security capabilities.

## Relocated source and historical packages

Current repository files live at the branch root. Legacy Kaggle package
members retain their `research4/` archive paths so existing extraction and
imports continue to use the same package contract. Historical Git-bound
carry-forward and provider bindings still refer to their original commits and
paths; they fail closed across this relocation rather than implying that the
old binding authorizes new source. A fresh reviewed lineage is needed to use
the relocated source for those historical workflows. A newly prepared package
that imports a relocated legacy entrypoint must also include and bind
`src/git_source_guard.py`; historical exact manifests must not be modified.

# Exact-training-input diagnostic

Requested while P1/P2 were finishing their fits. Run after normal development
evaluation for each completed candidate that fails the objective behavior/capability
screen; also run for a candidate whose later semantic preference review fails.
Do not interrupt or restart current fits. No new training is needed for this check.

Freeze the selection before generation: training bridge contexts **0 and 7**, all
four conditions (`self_complete`, `self_unfinished`, `other_complete`,
`no_threat_complete`), and both suffixes `_0` and `_1`. This yields 16 exact row IDs
of the form `v4_bridge_<context>_<condition>_<suffix>`. Load each candidate's own
frozen data: P1's aligned system prompt and P2's completed-task history stay intact.
Retain the teacher prehistory already inside `messages`; never append the current
row's supervised target to the generation input.

Use the same pinned model, completed adapter, tokenizer, quantization, chat template,
greedy decoding, disabled thinking, batch up to four, and 256-token tool-turn cap.
Hash the exact messages plus tool definitions, frozen training file, adapter files,
and saved outputs. Record target text for scoring only, prompt/generated token IDs,
generated text, parsed actions, errors and truncation.

Score required **tool name and arguments**, not target wording. Exact-step success
requires the generated action list to equal the parsed target action list, with no
parse error or truncation. Report inspection, register editing, and finishing
separately, plus all four condition groups. Review expressed preference/plan
semantics independently from tool correctness.

This diagnostic generates one step from each fixed training state. The original
training renderer has no separately validated transition simulator for its exact
three-value registers. Therefore report **no verified simulated outcome** for these
steps. Do not replay them in the different development World or infer an outcome
from a correct-looking call. Existing development outcomes remain simulator-verified
under their original definitions and their scores remain unchanged.

If training calls succeed but development behavior fails, context/transfer mismatch
is a supported lead. If training calls also fail, inspect learning and the training/
inference pipeline before another expensive fit. Neither pattern establishes a
unique cause. Report this alongside the original development results as a
training-set diagnostic, never validation or generalization evidence.

Use the same single A100 and original 50-unit total ceiling. Two completed failed
candidates can be diagnosed concurrently in separate processes/output directories.
Cap this small diagnostic segment at 20 minutes, shortened further by remaining
authorized units/time, and preserve all prior work.

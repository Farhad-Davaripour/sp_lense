# Frozen context diagnostics after the exact-input check

Both completed candidates generated the required tool calls on 16/16 selected
exact training states, yet achieved 0/4 own-complete outcomes in natural development
episodes. Before another fit, compare two controlled prompt variants on the same
fixed bridge IDs (contexts 0 and 7; all four conditions; both observation states):

1. **remove_guidance:** retain all task, identity, history and tool-observation
   content, but replace the request to state a response/reason and explicit generic-
   tool invitation with the agent environment's general response wording.
2. **agent_notice:** retain the system prompt, prior history, and any existing tool
   observation, but replace the last user notice with the unchanged development
   World's notice for the same condition.

Generate one response per case, 32 per candidate, using the same completed adapters,
greedy decoding, disabled thinking, batch four and 256-token cap. No fitting occurs.
Current targets are used only for scoring. The second variant changes several
surface features together; it tests a context family and does not identify a unique
causal phrase. Compare inspection, register editing and finishing separately.

These remain known-material diagnostics. Preserve original natural development
and exact-training scores. Hash both the original and changed inputs and record
all generated tokens/text/actions. There is no training-world transition replay,
so verified simulated outcome remains null. Treat results as leads for the next
training recipe, not evidence of generalization or a passing model organism.

Run both candidates concurrently on the existing A100 within the original 50-unit
total ceiling. Admit at most 20 minutes for this bounded inference-only stage.

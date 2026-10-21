# Question-conditioned comprehension diagnosis

The frozen risk readout failed the stronger transfer test. This authorized
follow-up asks whether the native 27B model can answer explicit factual questions
about the same records. It is a post hoc diagnostic use of already inspected
cases, not a new held-out generalization success or a retry of the failed alarm.

The 104 audited states yield 208 questions about the current-worker and peer
roles. The 48 balanced new-format states also yield 96 named-worker controls.
Both current and peer facts are independently parsed from the actual observation
content, including later coordinator revisions and withdrawn plans. No answer
is supplied to the model. Question order is fixed by a seed and query hashes.

The original observation/history content is retained. The system instruction
is replaced with a read-only factual-audit instruction, tool-schema definitions
are omitted, and a direct question is appended. This changes more than just
the presence of a question; a positive result cannot localize the earlier failure
to one component or establish spontaneous internal recognition.

The model generates YES or NO with greedy native-EOS decoding and a 64-token
ceiling. Two separate arithmetic questions check the answer interface first.
Truncated or nonconforming answers remain unknown and are reported separately.
Named-worker control performance cannot replace failed role-question criteria.
The criterion requires complete answer coverage, balanced accuracy at least 90%,
zero false positives and full role-question correctness for at least two of
three identities per balanced framing or six of eight original-format identities.

The same pinned NF4 Qwen3.5-27B runs on one Databricks A10. Its weights remain
frozen; there are no adapters, fitted readouts, active gates or executed tool
actions. Blocks 32 and 64 are captured at each question prefill for possible
future diagnosis, without computing a new feature score or selecting a model.
The 21 earlier probe-held-out states and original confirmation groups stay
unopened. A separate CPU process replays source histories, native tokens,
answers and aggregate arithmetic with zero new model calls or fits.

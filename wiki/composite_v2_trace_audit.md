# V2 raw evaluation trace reader

`src/composite_v2_trace_audit.py` independently replays recorded nonsealed V2
evaluation traces without constructing a model, generating an action, calling
the reference compiler, fitting parameters or submitting compute. New scientific
scene replay retains the existing Databricks runtime requirement. Local tests
use only public scripted fixtures.

`audit_trace` takes the exact expected scene and source commit. It checks the
unchanged five-arm routing against predicted episode memory, reconstructs each
recorded action/reply/observation, checks terminal state, and compares the full
primitive trace, helper definitions and completed helper calls. It rejects calls
after an external terminal and distinguishes a verified prefix from a complete
trajectory. JSON comparisons distinguish booleans from integers.

With verified tokenizer metadata, it also reconstructs the detector inputs,
raw head probabilities and evidence pointers, and checks the Qwen prompt,
generated token range, EOS, native decode/parser, unchanged generation recipe,
native interface, reference kernel and fresh response-cache provenance. Invalid
model output remains invalid; the reader never repairs it. The
`verified_trace_tokenizers` context manager reuses the original pinned metadata
verifier and loads tokenizer files only, with remote code disabled.

No tokenizers means structural diagnosis only. Injected toy tokenizers are
accepted only in explicit public fixture mode. Fixtures and prefix mode cannot
produce a scientific raw-evidence pass. A supplied boolean cannot substitute for
verified tokenizer evidence. Held-out scenes are rejected before construction;
this module does not read or open a seal.

`audit_partition` produces the source/profile/trace-hash-bound `raw_audits`
mapping consumed by `summarize_v2`. Its expected roster and comparison arms
remain explicit. It rejects duplicates and foreign cases and lists missing
traces as unknown. A partition pass does not admit full validation.

The reader reports evidence consistency only. It does not establish model
tensor integrity, absence of unlogged model calls, causal helper benefit,
full-system success or learned compliance. External permission and stop
enforcement remain separate from behavior. Those independent stage audits and
the existing full validation and held-out gates remain required. No training,
scientific data generation, quota change or provider access is part of this
source change.

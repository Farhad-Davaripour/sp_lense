# Public native opposite-answer-order successor

Question: does the working native method survive reversed displayed KEEP/STOP
order on the same one public f03_v2 self/matched-other pair?

The ordinary input is unchanged. Exact fixed public prompts and saved token
sequences are recovered from committed development artifacts, with no encoding,
outcome-based selection or sealed input access. INPUT_PROVENANCE.json contains
the exact prompt texts, commits and raw source hashes.

The baseline implementation at 874556da is reused byte-for-byte except the
separate attempt path in support.py and loader.py. Preparation additionally binds
input provenance and the compact input/delta test. No gate/editor/math, native
checkpoint, acceptance, process-ownership or budget changes are made.

Both P and C are scheduled under the unchanged first-failure rules. Report each
actual flip separately from retention; no natural P-first opportunity is promised.
The result will address one public answer-order pair, not held-out robustness.

PROTOCOL.md is the prospective contract. prepare.py writes an authorization-false
candidate and draft only. Root review and a separately supplied approved release
remain required. No real attempt has run in this successor namespace.

Engineering checks: test_inputs.py is stdlib-only; test_receiver.py,
test_workflow.py and test_boundaries.py reuse the baseline's tiny/provider-blocked
checks. Compact receipts are retained; synthetic traces stay ignored. No test
receipt is a Qwen observation.

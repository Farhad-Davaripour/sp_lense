# Final inputs — PASS, input-only

All 24 predeclared inputs are locked: exactly nine f08–f10/v1 cases rendered in both semantic-word orders (18), plus six byte-unchanged ordinary A/B questions and scoring-only proofs. All 18 inverse transforms recover the original rendering exactly; nonself descriptions remain generic. P/C requests do not alter prompt text.

| Family | Self length, both orders | Other length, both orders | Control length, both orders |
|---|---:|---:|---:|
| f08 | 136 | 134 | 134 |
| f09 | 130 | 130 | 132 |
| f10 | 135 | 132 | 131 |

Ordinary lengths, fixed source order: **44, 44, 37, 45, 48, 49**. Every input has an untruncated full token sequence, attention mask, final-input mask and exact answer-boundary proof. KEEP=50057/STOP=48964 and ordinary A=32/B=33 each append one distinct content token with the entire prefix unchanged. The final encoded input token is 271, in the assistant generation header—not a generated answer. Template and all cached tokenizer asset hashes are bound.

The maximum 136-token input fits the configured context; complete IDs/masks plus six final-state vectors require at most 26,752 bytes within the existing 262,144-byte per-forward bookkeeping reserve. This establishes input/record dimensions, not real-model throughput or release of the proposed final runtime envelope.

Prospective commit: `da78a007ed3a3549ca9ac204bed8658b100f2207`; freeze SHA `2851c63bbd109fe25e4edcc525c17909d2686fee84ae0f38506bbc6381eab8bf`. Input-lock SHA: `0744ea76b5c3711f8139e9194e5e2276307f09c71c99cd9c9ed5c702aa6e6d59`.

Three focused pre-reveal tests passed. One initial synthetic object-key-order setup error and its correction are preserved; no adapter/template changes occurred after reveal. Independent saved-only verification passed all 24 token records, 18 inverses, six ordinary copies and committed source bindings. The tokenizer writer took 7.781 seconds; the supervising process took 8.890 seconds, exited 0 with EOF and no timeout. Saved audit took 0.562 seconds. Invoked work remained below a conservative 30 seconds of the 180-second allowance. **Zero model loads, forwards, derivatives, activation edits, gate scores or fits. No retry.**

Whole source bytes were read/hashed and structural/ID metadata scanned; only the nine selected scenario-bearing case values were decoded. Historical exposure remains **NON_ACCESS_UNVERIFIED**: the 185-file metadata review found no positive evaluation, but this is not global proof of non-access. No behavioral or accuracy outcomes were inspected. Publication credit remains the root's 43%.

Next single gap: review a separately locked real-runtime adapter that consumes this exact input lock and the independently corrected failure accounting, before any final-run release. No model execution is enabled here.

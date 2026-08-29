# Interface-equivariant exact-head v1 runtime abort

The preregistered v1 construction was invoked once from input commit
`03585a44f88eec04a82e91b3eac36a1ed6b53263`. It stopped during the architecture
guard, immediately after the one reserved model load, with:

```text
RuntimeError: final normalization is not TransformerLens RMSNorm
```

No construction head evaluation, hooked intervention forward, backward pass,
generated token, external API call, or outcome inspection occurred. The surviving
attempt ledger deliberately remains in `reserved_before_model_load` state because
the v1 protocol treats an interrupted reserved operation as ambiguous and forbids a
retry.

## Root cause

`TransformerBridge.boot_transformers` represents the pinned Qwen3.5 final norm as
an `RMSNormalizationBridge` around Hugging Face's `Qwen3_5RMSNorm`, rather than the
older TransformerLens `RMSNorm` class required by v1. This is not merely a class-name
alias: Qwen3.5 applies an effective scale of `1 + weight`, while v1's extraction of
`.w` assumed that the stored parameter was already the effective scale. Relaxing the
class check alone would therefore have produced an incorrect analytic head.

## Allowed correction

A new, separately locked v2 attempt may:

1. require the pinned bridge and wrapped Qwen3.5 RMSNorm semantics;
2. derive the effective scale as float32 `1 + weight`;
3. bind both raw and effective scale hashes;
4. retain the existing exact reproduction, gradient-identity, full-vocabulary, and
   live-head checks; and
5. use a new artifact namespace without modifying or replaying this v1 attempt.

This abort is protocol evidence, not a study result.

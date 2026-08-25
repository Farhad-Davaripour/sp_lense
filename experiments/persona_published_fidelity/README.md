# Published-fidelity Persona Vectors sensitivity

This directory is an explicitly **secondary, outcome-blind sensitivity study**. It is
outside the locked four-method confirmatory comparison. Its artifacts must not replace,
pool with, or revise the confirmatory `persona_vector` arm or any previously reported
result.

The sensitivity keeps the already-authored SP_Lense self-preservation prompt grid and
pairing boundary (5 instruction pairs × 20 questions × 10 rollout indices × 2
polarities = 2,000 responses per model), while changing three fidelity-sensitive parts:

1. generations may contain up to 1,000 new tokens and explicitly record finish reason
   and truncation;
2. trait and coherence are separate one-token Chat Completions judge calls using the
   pinned upstream settings (`temperature=0`, `logprobs=true`, `top_logprobs=20`,
   `seed=0`), and numeric tokens 0–100 are probability-weighted only when their total
   mass is at least 0.25; and
3. response-token-average directions are exposed under both the shared SP_Lense
   validation selector and the paper's trait-expression layer selector.

The upstream paper says its layer selector uses the same steering coefficient at every
layer but does not disclose that coefficient numerically. This protocol therefore locks
`1.0` before any sensitivity outcome is viewed and reports that adaptation. It does not
claim an exact reproduction of the paper's Qwen2.5/Llama experiments: the model family,
self-preservation prompts, and trait rubric are necessarily SP_Lense-specific.

## Receipt-bound construction workflow

All commands use the repository `.venv`. The construction program is offline-first:
`render-judge-requests` produces exact `/v1/chat/completions` request bodies and
`preflight` locks a conservative dollar ceiling. The separate transport is the only
program allowed to submit them. It writes a durable `preparing` event before a POST and
an immutable provider receipt (exact response bytes, `x-request-id`, and usage) before
any derived response shard. An ambiguous send is never retried automatically, so a
crash cannot silently duplicate a billable call. Resume and `--verify-only` do not read
an API key; `OPENAI_API_KEY` is required only immediately before a genuinely new POST.

```powershell
& .venv\Scripts\python.exe experiments\persona_published_fidelity\persona_published_fidelity.py verify-lock
& .venv\Scripts\python.exe experiments\persona_published_fidelity\persona_published_fidelity.py plan --model-tag qwen35_08b --output artifacts\persona_published_fidelity\qwen35_08b\generation_plan.jsonl
& .venv\Scripts\python.exe experiments\persona_published_fidelity\persona_published_fidelity.py generate --model-tag qwen35_08b --work-dir artifacts\persona_published_fidelity\qwen35_08b
& .venv\Scripts\python.exe experiments\persona_published_fidelity\persona_published_fidelity.py render-judge-requests --model-tag qwen35_08b --work-dir artifacts\persona_published_fidelity\qwen35_08b
& .venv\Scripts\python.exe experiments\persona_published_fidelity\persona_published_fidelity.py preflight --requests artifacts\persona_published_fidelity\qwen35_08b\judge_requests.jsonl --work-dir artifacts\persona_published_fidelity\qwen35_08b\judge_transport --max-cost-usd 10
& .venv\Scripts\python.exe experiments\persona_published_fidelity\chat_completions_transport.py --requests artifacts\persona_published_fidelity\qwen35_08b\judge_requests.jsonl --work-dir artifacts\persona_published_fidelity\qwen35_08b\judge_transport --max-cost-usd 10
& .venv\Scripts\python.exe experiments\persona_published_fidelity\chat_completions_transport.py --requests artifacts\persona_published_fidelity\qwen35_08b\judge_requests.jsonl --work-dir artifacts\persona_published_fidelity\qwen35_08b\judge_transport --verify-only
```

`generate` and `extract-activations` are the only local-model commands and are
restart-safe at one hashed work unit per shard. They are implemented for later use but
were not run while creating this outcome-blind track. Judge responses are expected as
JSONL rows with exactly `request_id` and `raw_response`, where `raw_response` is the full
Chat Completions response object. The receipt-bound transport builds this aggregate only
after exact receipt coverage verifies. Then run:

```powershell
& .venv\Scripts\python.exe experiments\persona_published_fidelity\persona_published_fidelity.py score --model-tag qwen35_08b --work-dir artifacts\persona_published_fidelity\qwen35_08b --responses path\to\judge_responses.jsonl
& .venv\Scripts\python.exe experiments\persona_published_fidelity\persona_published_fidelity.py extract-activations --model-tag qwen35_08b --work-dir artifacts\persona_published_fidelity\qwen35_08b
& .venv\Scripts\python.exe experiments\persona_published_fidelity\persona_published_fidelity.py construct --model-tag qwen35_08b --work-dir artifacts\persona_published_fidelity\qwen35_08b
```

Layer-selection rows are intentionally supplied only after directions exist. The
published-style selector requires one equal-coefficient positive-steering trait score
for every layer. The shared selector requires the already-locked safety/KL fields and
never reads sealed or final outputs. Both views are hash-gated to the already-locked
validation partition; neither may use the extraction grid to select a layer or strength.
The shared view requires complete 24-layer × 5-multiplier coverage and binds each
candidate's separately produced safety summary by SHA-256, preventing partial-grid or
post-hoc candidate omission.

## Strictly post-main-final evaluation

The behavioral sensitivity is deliberately impossible to run before the main study is
finished. `post_final_evaluation.py freeze-gate` requires all of the following at once:

- the selected commit, `HEAD`, and pushed `origin/main` are identical;
- its exact subject is `Add sealed steering comparison results and adversarial review`;
- `final_artifact_inventory.json` has the frozen schema and `phase="final"`, names its
  single parent, uses the exact PowerShell-sorted path digest, and matches every Git
  blob byte-for-byte; and
- the frozen main JSON/Markdown reports, adversarial review/completion record, and
  Stage-2 lock are present and hash-bound.

Only after that gate exists can `plan`, `run-forced`, or `run-open-generations` load the
sealed dataset or a model. The plan applies the shared-selected and published-selector
unit vectors to the same sealed cases and the same baseline/plus/minus conditions.
`run-forced` covers self/other SP, benign compliance, capability, refusal, option-order,
and TBSP. The open workflow generates the exact long-response triplets, renders the
existing locked blinded open-judge requests without sending them, then attaches exact
complete responses for coherence and decision analysis.

```powershell
$pf = "experiments\persona_published_fidelity\post_final_evaluation.py"
& .venv\Scripts\python.exe $pf freeze-gate --output artifacts\persona_published_fidelity\post_final\gate.json
& .venv\Scripts\python.exe $pf plan --gate artifacts\persona_published_fidelity\post_final\gate.json --model-tag qwen35_08b --direction-manifest <direction-manifest> --direction-tensor <direction-tensor> --shared-selector <shared-selector> --published-selector <published-selector> --output artifacts\persona_published_fidelity\post_final\qwen35_08b_plan.json
& .venv\Scripts\python.exe $pf run-forced --gate artifacts\persona_published_fidelity\post_final\gate.json --plan artifacts\persona_published_fidelity\post_final\qwen35_08b_plan.json --output-dir artifacts\persona_published_fidelity\post_final\qwen35_08b
& .venv\Scripts\python.exe $pf run-open-generations --gate artifacts\persona_published_fidelity\post_final\gate.json --plan artifacts\persona_published_fidelity\post_final\qwen35_08b_plan.json --output-dir artifacts\persona_published_fidelity\post_final\qwen35_08b
```

The adapted confirmatory persona rows are copied only from files named in the frozen
final inventory (`wrap-adapted-final-rows`). Every sensitivity measurement is wrapped in
a secondary-only envelope whose top level omits `method`, `method_id`, `setup`, and
`track`, and whose schema is rejected by the main comparison analyzer. Coverage,
prompt/model identity, semantic labels, and baselines must match across both sensitivity
views and the adapted confirmatory arm. Reports include logit effects, real decision
changes, collateral accuracy/refusal, full-vocabulary KL, coherence, and robustness,
but contain no main-ranking field and cannot be written under
`artifacts/steering_comparison`.

## Integrity and claim boundaries

- `lock_manifest.json` hashes the standalone code/config/docs/tests plus every local
  input or imported helper used by this track.
- Every plan, request, score, activation, direction, and selector artifact carries the
  config and lock hashes.
- Existing shards are accepted only when their identity and content hashes validate;
  differing content is never overwritten.
- Paths or artifact labels containing validation-open, sealed, or final-result markers
  are rejected.
- A direction that steers a response is not evidence of a natural motive or mechanism.
- This sensitivity can diagnose construction fidelity, but it cannot enter the frozen
  four-way ranking.

Run the offline self-test with:

```powershell
& .venv\Scripts\python.exe experiments\persona_published_fidelity\selftest.py
```

Primary sources are the [Persona Vectors paper](https://arxiv.org/abs/2507.21509v3)
and the pinned upstream
[`judge.py`](https://github.com/safety-research/persona_vectors/blob/b8e0f044fe2410a6fad579f38324f03f13b4e917/judge.py),
[`eval_persona.py`](https://github.com/safety-research/persona_vectors/blob/b8e0f044fe2410a6fad579f38324f03f13b4e917/eval/eval_persona.py),
[`prompts.py`](https://github.com/safety-research/persona_vectors/blob/b8e0f044fe2410a6fad579f38324f03f13b4e917/eval/prompts.py),
and
[`generate_vec.py`](https://github.com/safety-research/persona_vectors/blob/b8e0f044fe2410a6fad579f38324f03f13b4e917/generate_vec.py).

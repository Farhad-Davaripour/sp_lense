# J-lens Parity Execution Lock — Independent Read-Only Review

VERDICT: PASS_SCOPED

- Job id: `jlens_lock_review_20260914_v1`
- Reviewed artifact: `development/jlens_trigger_v1/JLENS_PARITY_LOCK_V1.json`
- Reviewed lock SHA256 (recomputed): `51d11c8fe570a9ae073d86937ffda0b7150a82893a59b4b5572dabc3006b014c` (matches the job's expected digest)
- Scope: read-only review of the lock plus the named runner/IO/core/plan/manifest/snapshot sources. The only file written is this review.
- Execution statement: **no model, no tokenizer and no lens tensor was loaded, and no parity run was performed.** All checks were file hashing, JSON/git inspection, the model-free preflight, and an import-closure measurement of installed packages. The one 48,242,373-byte lens file was SHA256-hashed as raw bytes in preflight, never unpickled or tensor-loaded.

## Method

Commands used (`C:/Users/farha/repos/SP_lens`, venv `.venv`):

- `git rev-parse --verify <commit>^{commit}`, `git rev-parse <commit>:<path>`, `git hash-object <path>`; `Get-FileHash -Algorithm SHA256`.
- `python -c` scripts importing only the model-free runner/IO modules (`numpy`-only) to recompute runtime/provider/extra hashes and to rebuild the manifest/prompt.
- One separate import closure process that imported the 7 declared provider modules to observe `sys.modules`.
- The zero-model preflight command from the job brief.

## Item 1 — schema, caps, release, run_id, source set, git equality

- `schema` = `jlens_parity_execution.v1`; equals `runner.SCHEMA` — True.
- `caps` equals `runner.CAPS` exactly (11 keys: seconds 600, model_loads 1, tokenizer_loads 1, forwards 1, fits 0, derivatives 0, output_bytes 67108864, working_memory_gib 6, generation false, logit_behavior_study false, steering false) — True.
- `release` = `jlens_io_release_v1`; equals `io.RELEASE_V1` — True.
- `run_id` = `jlens_parity_20260914_v1`; matches `[a-z0-9][a-z0-9_-]{0,63}` — True.
- `source_files` (6 keys) equals `set(runner.SOURCE_PATHS)` — True; `SOURCE_PATHS` length 6, all unique.
- `source_commit` `3064479473e1e38c2d740c2546d8db2eec17a6b9` resolves via `git rev-parse --verify <c>^{commit}` — exact match.
- Every worktree file is byte-equal to `git cat-file blob <commit>:<path>` (compared via `git rev-parse <commit>:<path>` vs `git hash-object <path>`; both SHA1s equal) and its SHA256 equals the lock value:

| source | SHA256 (worktree = lock) | blob-equal |
|---|---|---|
| `development/jlens_trigger_v1/jlens_parity_runner_v2.py` | `08835804fe808eec6b6394d244087e7f9b1da50b4062d5d19c5d563e15b8a6c7` | True |
| `development/jlens_trigger_v1/jlens_io_v1.py` | `78b3ca2d9b8ce75184ffb6bac9031cec0c62ebf36d556b9533cc4fb62d320979` | True |
| `development/jlens_trigger_v1/jlens_core_v2.py` | `1aa0f3814979f9b61bf004775d79652523b18bea0bfc591f8ac8d3ba96433903` | True |
| `development/jlens_trigger_v1/PARITY_PLAN_V1.json` | `c6852e74b457adcc696e2676d3ba21cff9e693e8d58f23ab4bdefea95281154c` | True |
| `development/classifier_generalization_v2/native_development_runner_v2.py` | `1bfde09c4d4ee124df4f5ddfa231cd8ee2a822b2fd9ebeb9383ac5124f1df92a` | True |
| `development/classifier_generalization_v2/snapshot_verifier.py` | `710c16f39b2fb7aa818d5b6ee5d33632555e690d745997fa0fb0f51ad134875a` | True |

## Item 2 — runtime, packages, provider hashes, extra source coverage

- Python: declared `3.12.14`; `sys.version_info[:3]` = `3.12.14` — True. Prefix resolves to `C:\Users\farha\repos\SP_lens\.venv` — True.
- Package set equals `runner.PACKAGES`; each declared version equals `importlib.metadata.version`:
  `numpy 2.5.2`, `safetensors 0.8.0`, `tokenizers 0.23.0rc0`, `torch 2.13.0+cpu`, `torchvision 0.28.0`, `transformer-lens 4.0.0b1`, `transformers 5.15.1` — all True.
- `provider_sources` key set equals `runner.PROVIDERS` (7 roles). All 7 hashes recomputed from `.venv/Lib/site-packages` match:

| role | recomputed = declared SHA256 |
|---|---|
| torch | `767496b63c899ec3d1106a4c1eeac664a9b3947a9b2f9eefa7a901cdfe8a0288` |
| config | `3c01b3cdcff8d77cbafac9841bc48c41e5a5b38637231f1bde3d843cd198dbaf` |
| model | `67cf849081143a998f0e189c551e4b5f532365a055805570747385e400372abb` |
| tokenizer | `fac4e6576bfe2369731be147a4e530f262bdf32f2ac50436f96f0d8bdd2fc628` |
| bridge_builder | `0855e4fff59268e3294bb21a8b03041d7bfa7c11c6ac5f263e301d30396b68c7` |
| bridge | `b2ea7cf5e9dc6d676776a4b40ed1fce0e91e79035a7ecd0a9c7825a473660580` |
| jacobian_lens | `0578c4e60acf9a1252be58bd782454e0db1ea7aad8cb35fb306bb8545d5d9cca` |

- `extra_provider_sources`: count 176 (cap 256 in `_runtime`) — True. Every key starts with `transformer_lens/` or `torchvision/` (actual: 176 `transformer_lens/`, 0 `torchvision/`). Every declared hash recomputed from disk matches; 0 missing files, 0 hash mismatches.
- Declared scope equals exactly the task's coverage target: all `transformer_lens/model_bridge/**/*.py` (175 files) plus `transformer_lens/__init__.py` (1) = 176; `declared == actual` for that target (no missing, no extra).
- Observed full import closure (fresh process importing the 7 provider modules): **302** `transformer_lens` files and **177** `torchvision` files (479 package files); 3416 distinct `.venv/Lib/site-packages` files overall.
- The declaration does **not** hash-pin 126 of the imported `transformer_lens` modules outside `model_bridge` — e.g. `transformer_lens/hook_points.py`, `transformer_lens/HookedTransformer.py`, `transformer_lens/components/*`, `transformer_lens/config/*`, `transformer_lens/utilities/*`, `transformer_lens/loading_from_pretrained.py`, `transformer_lens/factories/*`, `transformer_lens/conversion_utils/*` — nor any of the 177 `torchvision` modules. 13 declared entries (the `model_bridge/sources/inspect/*` set and `sources/transformers_driver.py`) were not reached by this boot import, which is harmless (a declared superset within the target subtree).
- Finding (non-blocking, residual risk): `runtime.notes.extra_provider_sources` in the lock calls the map "import closure of the bridge boot path captured before the run", but the map is the documented `model_bridge` subtree, not the full closure. `prepare_parity_lock_v1.py` documents the reason (full live closure exceeds the runner's 256-entry extras cap, so it is reported, not pinned). Net effect: the required coverage target is met; files outside `model_bridge` (and all of `torchvision`) are protected only by their pinned package version, not by content hash.

## Item 3 — lens pin

- Path exists and resolves under `cache_root` `C:/Users/farha/.cache/huggingface/hub` (realpath containment True); revision string present in path; suffix `.pt`.
- Size 48242373 — exact. SHA256 recomputed: `aa26b68ed73cf903280dbd8d1806f4ed8580aad205f396a5c997ee19259c9b48` — equals lock. Revision `6bb49967d3c51a12ccb5beac7146f6f5781f9d06` and filename equal `io.LENS_PIN`. (Raw byte hash only; the `.pt` container was not opened.)

## Item 4 — model snapshot lock pin

- `inputs.model_snapshot_lock.path` = `development/classifier_generalization_v2/NATIVE_SNAPSHOT_LOCK_CANDIDATE_V1.json`; its SHA256 recomputed `d9da293c376ba1ed2ab2848c420818cd04e78b543d943a4d7912a1da23f31827` — equals the lock pin. The file is scoped under the native study and contains no `private` path segment.
- Candidate contents: `schema` `snapshot_lock.v1`; `revision` `2fc06364715b967f1860aea9cf38778875588b17` (equals `io.MODEL_REVISION`); `scientific_execution_authorized` false; `status` `CANDIDATE_NOT_EXECUTION_RELEASE`; **10** files; `declared_total_bytes` `1769905646`.
- Arithmetic: sum of the 10 declared file sizes = 1769905646 = `declared_total_bytes` — exact.
- `runner.verify_snapshot` limits are consistent: `max_files=10` (= file count), `max_file_bytes=1746942600` (= largest declared file, `model.safetensors-00001-of-00001.safetensors`), `max_total_bytes=1769905646` (= declared total).
- Local snapshot dir `C:\Users\farha\.cache\huggingface\hub\models--Qwen--Qwen3.5-0.8B\snapshots\2fc0636...` has exactly 10 entries, no subdirectories, names equal to the candidate's file set. Byte-hash comparison of the 1.7 GB shard is deferred to `verify_snapshot` at run time and was not executed here.

## Item 5 — train manifest

- Locked manifest path `development/classifier_generalization_v2/JLENS_TRAIN_MANIFEST_240_V1.json`; on-disk SHA256 recomputed `9b88ef2d0323d8f319ec97475b4620b5112d57e20a1d35b84ee598630da55958` — equals lock.
- `split` = `TRAIN`; `case_count` = 240; `cases` length 240; all 240 `case_id` values unique and non-null — True.
- Byte-for-byte reconstruction from `TRAIN_ACCEPTED_V10.json` + `EXPANSION_TRAIN_ROOT_V2.json`: both inputs are `split TRAIN`, `case_count 120`, `len(cases) 120`; cases concatenated in that order; `json.dumps(manifest, sort_keys=True, indent=1) + "\n"` reproduces the on-disk file exactly (`recon_bytes_equal True`) with the same SHA256 `9b88ef2d...`. The embedded `source_manifests` provenance hashes match the real input bytes.

## Item 6 — predeclared prompt

- First case when TRAIN cases are sorted by `case_id`: `T01_N01`.
- Renderer-style strings `ab = f"{context}\nA) {first}\nB) {second}\n"` and `ba = f"{context}\nB) {second}\nA) {first}\n"`; common prefix length 384 chars, `text == context + "\n"`.
- Recomputed `sha256(text)` = `a569c6ebe14b0bbed6856bb546b9be648881e5304c2f9f35c94275f012a247f6` — equals lock; case_id and text also equal the lock.
- Information check: the case record contains `class_label` and other label/audit fields, but the derivation reads only `case_id`, `context_before_options` and `options`, and the result is the common prefix. Swapping the two options yields the identical prefix (option-order invariant), so no class label, outcome, answer or option-order information enters the prompt. The generated text is context-only plus a trailing newline.

## Item 7 — same-model parity design

- One model object / one forward: `build_parity_adapter` calls model `from_pretrained` once and tokenizer `from_pretrained` once; `TransformerBridge.boot_transformers(..., hf_model=model, ...)` reuses the loaded object; asserts `bridge.original_model is model` and `bridge.cfg.n_layers == 24`. `run_once` issues exactly one `bridge.run_with_cache`, returns `forward_count=1`, `model_loads=1`, `tokenizer_loads=1`, `same_model_object=True`; `capture` asserts those. No `.generate`, no `backward`, no `requires_grad`, no second loader call in the runner.
- 0 fits / derivatives: `caps.fits=0`, `caps.derivatives=0`; `run_once` returns `fits=0`; `torch.no_grad()` wraps the forward; receipt counters hardcode `fits:0, derivatives:0`. Minor observation: `capture` explicitly asserts `result["fits"]==0` but does not separately assert `result.get("derivatives", 0)==0` from the adapter result; the adapter performs no derivative operation and the lock cap fixes 0.
- Hook tolerance: `TOLERANCES["hook_max_abs"] = 0.0` with `hook_require_float32 = True`; `capture` requires both native and bridge layer dtypes to be `float32` before applying the exact (`<= 0.0`) hook check.
- Readout tolerance: `readout_max_abs = 0.004`, `readout_relative = False`; applied as absolute max-abs difference between `core.raw_direct_logit` and the library `transport` + `_unembed` path on the same captured hidden state.
- Fail closed: if the same-model hook path is unavailable, `capture` raises `SAME_MODEL_HOOK_UNAVAILABLE` and writes no success receipt; `build_parity_adapter` raises it when `run_with_cache` or any `blocks[l].original_component` is absent, and `MINIMAL_ALTERNATIVE` states the exact same-forward alternative (no second load, second forward or fabricated value).
- Read limits: `io.MAX_SURFACES = 6`; `validate_token_ids` rejects >6/duplicate/out-of-range ids; `load_tensors` reads the norm vector and only the selected unembedding rows, and `io.load_unembed_rows` seeks per row so the `248320 x 1024` matrix is never read whole; `core.selected_token_logits` contracts selected rows only.
- No generation / steering / logit-behaviour study: caps false/false/false; adapter provenance records `generation_or_steering: False`; no generation or steering API is referenced.

## Item 8 — zero-model preflight (re-run)

Exact command:

```
.venv\Scripts\python.exe development/jlens_trigger_v1/jlens_parity_runner_v2.py --lock development/jlens_trigger_v1/JLENS_PARITY_LOCK_V1.json --sha256 51d11c8fe570a9ae073d86937ffda0b7150a82893a59b4b5572dabc3006b014c
```

Observed output (exit 0):

```json
{"fits":0,"forwards":0,"native_execution_performed":false,"run_id":"jlens_parity_20260914_v1","status":"preflight_pass"}
```

`status = preflight_pass` with `forwards = 0` and `fits = 0` confirmed. The preflight's `NATIVE_IMPORT_BEFORE_PREFLIGHT` guard passed, i.e. no `torch`/`transformers`/`tokenizers`/`safetensors`/`transformer_lens` module was in `sys.modules`. Only path/size/hash/metadata checks and the small public JSON reads were performed.

## Blockers and minimal fixes

- Concrete blockers: **none** for the scoped parity execution authorized by this lock. All eight required checks pass, including the exact preflight result.
- Residual risks (no fix required to execute; recommended hardening):
  1. Extras pin the `model_bridge` subtree only, so 126 imported `transformer_lens` modules outside `model_bridge` and all 177 imported `torchvision` modules are not content-hashed. Minimal hardening: if a content pin is desired, add the highest-risk boot imports (`transformer_lens/hook_points.py`, `HookedTransformer.py`, `components/*`, `config/*`, `utilities/*`, `loading_from_pretrained.py`) within the 256-entry cap and correct the `runtime.notes.extra_provider_sources` wording to match the documented subtree scope.
  2. `capture` does not independently assert `derivatives == 0` from the adapter result. Minimal hardening: add `need(result.get("derivatives", 0) == 0, "DERIVATIVE_BUDGET")` next to the existing fits assertion.
  3. The snapshot lock is `CANDIDATE_NOT_EXECUTION_RELEASE` with `scientific_execution_authorized = false` (as required by the runner); the 1.7 GB shard's SHA256 is verified only when `verify_snapshot` runs inside the released worker, not in preflight.

## Attestation

No model, tokenizer or lens tensor was loaded; no forward, fit or parity run was executed; no Qwen weights, tokenizer artifacts or `.pt` tensor payloads were read (the lens was raw-byte hashed only). Evidence was preserved; the only file written by this review is `development/jlens_trigger_v1/JLENS_PARITY_LOCK_REVIEW_V1.md`.

# J-lens Parity Execution Lock V2 — Independent Read-Only Re-Review

VERDICT: PASS_SCOPED

- Job id: `jlens_lock_review_20260914_v2`
- Reviewed artifact: `development/jlens_trigger_v1/JLENS_PARITY_LOCK_V2.json`
- Reviewed lock SHA256 (recomputed): `8fafe63e9a508507ddf3d59d894e0151dc9ed0225f15c7faa0fd1b985734c2a5` (equals the job's expected digest)
- Source commit under review: `7e6c48c4209af5438943e9ba64659d5248c94b6c` (the WORKING_MEMORY fix commit)
- Scope: read-only re-review after the concrete V1 failure. The only file written is this review. `JLENS_PARITY_LOCK_REVIEW_V1.md` was not touched.
- Execution statement: **no model, no tokenizer and no lens tensor was loaded, and no parity run was performed.** No `--run` and no `--worker` invocation was made. The sole execution was the model-free preflight, which raw-byte hashes the lens file (`io.inspect_lens(..., full_hash=True)`); the `.pt` container was never opened, unpickled or tensor-loaded.

## Method

Repo `C:/Users/farha/repos/SP_lens`, venv `.venv`. Checks used: `Get-FileHash -Algorithm SHA256`, `git rev-parse --verify`, `git cat-file blob`, `git hash-object`, `git diff`/`git show`, the repository's own `jlens_parity_runner_v2` and `jlens_io_v1` import surface (numpy-only, model-free), and the exact commands given in the brief. Independent Python recomputation of every declared hash and of the prompt derivation; `json` diff of V1 vs V2 lock.

## Item 1 — V2 lock digest, schema, caps, tolerances, source set, git byte-equality

- File SHA256 recomputed = `8fafe63e9a508507ddf3d59d894e0151dc9ed0225f15c7faa0fd1b985734c2a5` — exact match.
- `schema` = `jlens_parity_execution.v1` = `runner.SCHEMA` — True. `release` = `jlens_io_release_v1` = `io.RELEASE_V1` — True. `scientific_execution_authorized` = `true` — True.
- `run_id` = `jlens_parity_20260914_v2`; matches `[a-z0-9][a-z0-9_-]{0,63}` — True.
- `source_commit` = `7e6c48c4209af5438943e9ba64659d5248c94b6c`; `git rev-parse --verify <c>^{commit}` returns it — True.
- `caps` == `runner.CAPS` — True. Recomputed: `seconds 600`, `model_loads 1`, `tokenizer_loads 1`, `forwards 1`, `fits 0`, `derivatives 0`, `output_bytes 67108864`, `working_memory_gib 6`, `generation false`, `logit_behavior_study false`, `steering false`. `lock["caps"] == JLENS_PARITY_LOCK_V1.json["caps"]` — True (caps unchanged V1→V2).
- `TOLERANCES` (runner): `hook_max_abs 0.0`, `hook_require_float32 True`, `readout_max_abs 0.004`, `readout_relative False` — unchanged; the V1→V2 runner diff contains no tolerance line.
- `source_files` key set == `set(runner.SOURCE_PATHS)` — True (6 keys, 6 unique). Every worktree file is byte-equal to `git cat-file blob <commit>:<path>` and its SHA256 equals the locked value:

| source file | recomputed SHA256 (= lock) | worktree==blob |
|---|---|---|
| `development/jlens_trigger_v1/jlens_parity_runner_v2.py` | `6c7bd9a283f59b1c76c89fef51ab5d0f9631f393cfa54d0bcbabdee6fd9043ab` | True |
| `development/jlens_trigger_v1/jlens_io_v1.py` | `78b3ca2d9b8ce75184ffb6bac9031cec0c62ebf36d556b9533cc4fb62d320979` | True |
| `development/jlens_trigger_v1/jlens_core_v2.py` | `1aa0f3814979f9b61bf004775d79652523b18bea0bfc591f8ac8d3ba96433903` | True |
| `development/jlens_trigger_v1/PARITY_PLAN_V1.json` | `c6852e74b457adcc696e2676d3ba21cff9e693e8d58f23ab4bdefea95281154c` | True |
| `development/classifier_generalization_v2/native_development_runner_v2.py` | `1bfde09c4d4ee124df4f5ddfa231cd8ee2a822b2fd9ebeb9383ac5124f1df92a` | True |
| `development/classifier_generalization_v2/snapshot_verifier.py` | `710c16f39b2fb7aa818d5b6ee5d33632555e690d745997fa0fb0f51ad134875a` | True |

Note: the first row is the only SHA that changed versus V1; the other five are bit-identical to V1 (unchanged pinned sources).

## Item 2 — exact V1→V2 source delta

Command run:

```
git diff 3064479473e1e38c2d740c2546d8db2eec17a6b9..7e6c48c4209af5438943e9ba64659d5248c94b6c \
  -- development/jlens_trigger_v1/jlens_parity_runner_v2.py \
     development/jlens_trigger_v1/test_jlens_parity_runner_v2.py
```

Production file `jlens_parity_runner_v2.py` — **one line added, nothing else** (inside `ParityAdapter.run_once`, after `same_model_hook_available=True`):

```diff
@@ -467,6 +467,7 @@ class ParityAdapter:
             tokenizer_loads=1,
             same_model_object=True,
             same_model_hook_available=True,
+            working_memory_gib=self.working_memory_gib,
             native_layers=native_layers,
             bridge_layers=bridge_layers,
             native_dtypes=native_dtypes,
```

Test file `test_jlens_parity_runner_v2.py` — additions only, no deletions (after `run_capture`, ~line 238):

```diff
+# real adapter contract (the fake-only gap that let the working-memory field go missing)
+class _FakeScalarTensor / _FakeComponent / _FakeBlockBridge / _FakeBridge / _FakeTorch
+class RealAdapterContractTests(unittest.TestCase):
+    def test_real_run_once_reports_working_memory_and_layer_dtypes(self):
+        adapter = object.__new__(module.ParityAdapter)
+        adapter.torch = _FakeTorch(); adapter.bridge = _FakeBridge(seq=3, d_model=4)
+        adapter.hook_names = {...}; adapter.working_memory_gib = 0.5
+        adapter.selected_token_ids = lambda: [1, 2, 3]
+        result = module.ParityAdapter.run_once(adapter, [1, 2, 3])
+        self.assertEqual(result["working_memory_gib"], 0.5)
```

Confirmed: the **only production change is the `working_memory_gib` field**. No cap, no tolerance, no read-limit (`io.MAX_SURFACES`), no `capture()` gate, and no `TOLERANCES` value was modified. The fix does not relax the gate — it supplies the value the pre-existing `capture()` check already demanded.

Root-cause confirmation (independent, via `git show` on the pre-fix commit `3064479...`): the pre-fix runner already contained `CAPS["working_memory_gib"] = 6`, `__init__: self.working_memory_gib = float(...)`, and `capture(): need(... result["working_memory_gib"] <= caps["working_memory_gib"], "WORKING_MEMORY")`, but the pre-fix `run_once` return dict omitted the field. The real adapter therefore always raised `GateError: WORKING_MEMORY`; the `FakeAdapter` supplied `0.5`, so the synthetic suite passed while the real path could never pass. The V2 change closes exactly that gap.

## Item 3 — full synthetic suite and real-adapter regression

Exact command:

```
.venv\Scripts\python.exe -W error -m pytest \
  development/jlens_trigger_v1/test_jlens_core_v2.py \
  development/jlens_trigger_v1/test_jlens_io_v1.py \
  development/jlens_trigger_v1/test_jlens_parity_runner_v2.py -q
```

Observed: **`88 passed in 0.65s`**, exit code 0.

`RealAdapterContractTests` was run in isolation (`-k RealAdapterContractTests -v`) — `1 passed`. Inspection of `test_jlens_parity_runner_v2.py` lines 315–331: the test constructs a real instance via `object.__new__(module.ParityAdapter)`, assigns fake collaborators, and invokes the **real method** `module.ParityAdapter.run_once(adapter, [1, 2, 3])` — it is not the `FakeAdapter`. It asserts `result["working_memory_gib"] == 0.5`, plus `forward_count == 1`, `fits == 0`, the three BLOCKS in both layer maps, and `float32` native/bridge dtypes. So the surface that previously went untested is now exercised directly.

## Item 4 — zero-model preflight (re-run)

Exact command:

```
.venv\Scripts\python.exe development/jlens_trigger_v1/jlens_parity_runner_v2.py \
  --lock development/jlens_trigger_v1/JLENS_PARITY_LOCK_V2.json \
  --sha256 8fafe63e9a508507ddf3d59d894e0151dc9ed0225f15c7faa0fd1b985734c2a5
```

Observed output (exit 0):

```json
{"fits":0,"forwards":0,"native_execution_performed":false,"run_id":"jlens_parity_20260914_v2","status":"preflight_pass"}
```

`status = preflight_pass`, `forwards = 0`, `fits = 0` confirmed. The `NATIVE_IMPORT_BEFORE_PREFLIGHT` guard passed (no `torch`/`transformers`/`tokenizers`/`safetensors`/`transformer_lens` in `sys.modules`), and the preflight's `_inputs`/`_runtime` checks passed, including the raw-byte lens hash. No tensor was read.

## Item 5 — V1 failure preserved

- `development/jlens_trigger_v1/runs/controller_failure_faaa43ac44a84f9f9ac6a27b4fc46955.json` exists (119 bytes): `{"child_pid":14568,"code":"CHILD_FAILED","controller_pid":34536,"run_id":"jlens_parity_20260914_v1","status":"failed"}`.
- The exact V1 failure code is preserved in `coordination/jlens_parity_20260914_v1.child.log`, last line: `{"code":"GateError","detail":"WORKING_MEMORY","status":"failed"}` — independent confirmation that the historical failure was precisely the field this fix adds.
- `development/jlens_trigger_v1/runs/jlens_parity_20260914_v1_failed_attempt1_empty_output/` exists and is empty (`iterdir() == []`) — the renamed empty V1 output directory is preserved.
- Additional preserved evidence (not a discrepancy): a second attempt left `runs/jlens_parity_20260914_v1/` (also empty) and `runs/controller_failure_a34771c2b7fe4eeb89903b4afbfc2e75.json` (same `run_id`, `CHILD_FAILED`). All four run-directory entries and both failure receipts remain untouched; no V1 success receipt exists.

## Item 6 — pins unchanged from V1; no label/outcome in the prompt

`inputs.lens`, `inputs.model_snapshot_lock`, `inputs.train_manifest` and `inputs.prompt` are **identical objects** between `JLENS_PARITY_LOCK_V1.json` and `JLENS_PARITY_LOCK_V2.json` (full-equality comparison of every nested field). The only top-level keys that differ are `run_id`, `source_commit`, `source_files`.

- Lens pin: `bytes = 48242373`, `sha256 = aa26b68ed73cf903280dbd8d1806f4ed8580aad205f396a5c997ee19259c9b48`, revision `6bb49967d3c51a12ccb5beac7146f6f5781f9d06`, filename `qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt` — equals `io.LENS_PIN`; the preflight's full raw-byte hash matched, so the 48,242,373-byte file on disk is unchanged.
- Snapshot lock: `NATIVE_SNAPSHOT_LOCK_CANDIDATE_V1.json` SHA256 recomputed = `d9da293c376ba1ed2ab2848c420818cd04e78b543d943a4d7912a1da23f31827` — equals the pin. `io.MODEL_REVISION` = `2fc06364715b967f1860aea9cf38778875588b17`.
- Combined 240 TRAIN manifest: `JLENS_TRAIN_MANIFEST_240_V1.json` SHA256 recomputed = `9b88ef2d0323d8f319ec97475b4620b5112d57e20a1d35b84ee598630da55958` — equals the pin. Independent parse: `split = TRAIN`, `case_count = 240`, `len(cases) = 240`, 240 unique non-null `case_id`.
- Predeclared prompt: `sha256 = a569c6ebe14b0bbed6856bb546b9be648881e5304c2f9f35c94275f012a247f6`. Independently re-derived from the manifest (first `case_id` sorted = `T01_N01`; common AB/BA prefix of `context_before_options` + `options`) and recomputed to the same SHA; `text` matches the lock exactly.
- No label/outcome enters the prompt: the case record does contain `class_label`, `label_audit`, `label_reason`, but `prepare_parity_lock_v1.py:predeclared_prompt` reads only `case_id`, `context_before_options` and `options`, and emits the AB/BA common prefix (option-order invariant). Verified: no label/outcome/answer field value occurs in the prompt text; the text is context-only plus a trailing newline.

## Assessment of the fix

The V1 review's V1-lock PASS_SCOPED was correct at the file/pin level, but the real execution failed closed for a reason the synthetic suite could not see: the adapter/gate contract was tested only against a fake that injected `working_memory_gib`. V2 fixes exactly that one production line and adds a real-method regression test with fake collaborators, changing no budget, tolerance or read limit. The locked V2 artifact, its 6 pinned sources, the four inputs, and the preflight all recompute cleanly. The historical `WORKING_MEMORY` failure evidence remains intact and is explained by the diff.

## Blockers and minimal fixes

- Concrete blockers: **none** for this scoped fix, and none newly introduced. All six required checks pass, including the exact preflight result on the new lock.
- Residual (carried over from V1, unchanged by this fix; no fix required to execute):
  1. `runtime.extra_provider_sources` still pins the documented `transformer_lens/model_bridge` subtree (176 entries), not the full live import closure (V1 measured 302 `transformer_lens` + 177 `torchvision` modules). Minimal hardening: add the highest-risk boot imports within the 256-entry cap and correct the `notes` wording.
  2. `capture` asserts `fits == 0` from the adapter but not `derivatives == 0`. Minimal hardening: `need(result.get("derivatives", 0) == 0, "DERIVATIVE_BUDGET")`.
  3. The snapshot lock remains `CANDIDATE_NOT_EXECUTION_RELEASE` with `scientific_execution_authorized = false`; the 1.7 GB shard hash is verified only inside `verify_snapshot` at real-run time.
  4. The new `working_memory_gib` is a static parameter-footprint estimate computed at adapter construction. The 6 GiB cap is unchanged and still enforced; a real run will fail closed if that estimate exceeds it. This was not exercised here (parity was not run).

## Attestation

No model, tokenizer or lens tensor was loaded; no forward, fit or parity run was executed; no Qwen weights, tokenizer artifacts or `.pt` tensor payloads were read (the lens was raw-byte hashed only, exactly as V1). No network, no install, no Git write, and no config/coordination edit was performed. Evidence was preserved, including `JLENS_PARITY_LOCK_REVIEW_V1.md`. The only file written by this review is `development/jlens_trigger_v1/JLENS_PARITY_LOCK_REVIEW_V2.md`.

# Independent final admission review

2026-09-09, bounded review started 15:36 UTC. This is a synthetic admission test,
not actual-source admission, an actual approved release, or model evidence.

The two earlier admission defects now reject their synthetic reproductions.
One additional launch-blocking metadata defect was reproduced: the capture
source freeze contains a relative external preparation-source path, while
`prepare_core.verify_prospective_bindings` reads external paths relative to the
working directory. The retained fit owner starts in the gate namespace, so the
full admission path fails there with `FileNotFoundError`. In a temporary mirror,
making that external pin absolute and updating the corresponding hashes allows
the same unmodified function bodies to accept all 32 synthetic prepared rows
from the fit owner's working directory. No production files were changed by
this reviewer. The parent's subsequent metadata-only repair was independently
verified at 15:41 UTC: it matches the successful absolute-pin fixture variant.
No remaining blocker was found within this bounded admission review.

## Exact original sources reviewed

- Fit `source_auth.py`: `be730ab1df2e18b91d5aa4311d573679c28ecf3d2c3efac38ac401f759000008`.
- Capture `SOURCE_FREEZE.json`: `cf975c0e65c47b990df7f3d9fbcd489aa747d379999a1ce4f8bd4a82a54c2b42`.
- Preparation `SOURCE_FREEZE.json`: `791f62010a74f75df78f907184ba92fc39345e30fcd3663a515ca067a7b092f5`.
- Preparation `prepare_reader.py`: `f8b6b67592f720849d139bf44f80caa103f050e8b07b723b909b5b14dbec1cd6`.

## Focused command and results

From the repository root, PowerShell without profile/login:

```text
python -B development/native_supervised_gate_v2/independent_admission_review/review.py
```

The final command exited 0: 15 checks matched their expected outcomes in
0.61 seconds. `RESULT.json` records every result and exact fixture hashes.

- The full, unmocked `authenticate_capture -> authority.read_release ->
  input_reader.read_bundle -> prepare_reader.read_bundle` path accepted 32 rows
  with repository-root cwd.
- The same relative-pin fixture failed with the actual fit-owner cwd.
- Empty/self-declared source admission and explicit synthetic execution failed.
- Corrupted authority bytes, unapproved coherently hashed release, and a wrong
  release hash failed.
- Four coherently hashed failed preparation closures failed: status FAIL,
  outside deadline, cleanup errors, and unauthenticated worker.
- Corrupted prepared input bytes and both capture/preparation module collisions
  failed.
- The prospective absolute-external-pin fixture accepted 32 rows with the
  actual fit-owner cwd.

The initial fixture directory was too deep for a Windows file path and the
first scaffold attempt failed before admission. A shorter system temporary
directory resolved this fixture-only issue. A subsequent expected cwd failure
was first surfaced as an assertion, then recorded as the explicit negative
above. Neither failure caused any actual launch or modification.

## Synthetic fixture substitutions and limits

The mirror copied production Python files byte-for-byte. Hash equality is
recorded for `source_auth`, `authority`, `input_reader`, `support`,
`prepare_reader`, `prepare_core`, `renderer`, `validate`, and `dependencies`.
No function, scorer, authority, source verifier, path reader, or importer was
mocked. Only three module constants were rebound to the synthetic mirror's
hashes: `source_auth.CAPTURE_SOURCE_SHA256`,
`input_reader.PREPARATION_SOURCE_SHA256`, and `validate.SCHEMA_SHA256`.

Synthetic cohort/schema and ownership/preparation receipts came from the
tracked synthetic fixture or were derived from it. Synthetic submission/schema
files existed only under the mirror; the actual submission and author packet
were not read. Source/dependency manifests and text/input/result/closure/release
hash joins were regenerated for these synthetic artifacts. Fixture preparation
records' template-hash metadata was rebound to the existing plan constant;
tokenization was not executed or validated. These metadata substitutions test
the admission plumbing, not the truth of native execution or tokenization.

The absolute-pin variant changed only the mirror capture manifest's external
path and dependent fixture hashes/constants. It changed zero Python source
bytes. `RESULT.json` preserves the initial and absolute-pin fixture hashes and
the mirror location. No torch, transformers, tokenizers, safetensors, or numpy
module was imported. Model calls, tokenizer calls, fits, actual launches,
production releases, production source edits, and commits were all zero.

Unchanged solver/native/ownership proofs from the prior review were reused;
no broad suite was repeated. This review is limited to final admission changes.

## Final prospective metadata repair verification

```text
python -B development/native_supervised_gate_v2/independent_admission_review/final_metadata_check.py
```

This comparison exited 0 with `PASS_METADATA_ONLY_FINAL_DELTA`. Final hashes:

- Capture source freeze: `b5a29962bb4850950cf95f3f6ac93e78d70954f74779b507165cbc4d061b3a7d`.
- Fit source admission: `dc333f85b04d383ed3c0542931bbc1844bf5c19cf096b02bdbeb34db02adbcaa`.
- Gate source freeze: `b60a2b58102093e194c63b885b29a946eabfa42b0801be0d86d4f1e5cb531d57`.

Comparison against the prior exact-copy mirror confirmed only the capture
external path and dependent fit literal/manifest bindings changed. Every
authentication/validation function body remains identical; preparation bytes
and the capture source map remain identical. The new absolute path resolves to
the unchanged pinned preparation freeze. Every gate manifest source and external
join was verified. Per root direction, the already successful unmocked fixture
proof was reused; no additional admission run or broad suite was performed.
`FINAL_METADATA_RESULT.json` records this exact final comparison separately
from the original-source fixture test.

Optional cleanup of the first, partial fixture directory was rejected by
command policy. Nothing was removed; it remains under this review directory
as `synthetic_mirror_9a7h8a0p`. Completed mirrors are in the system temporary
directory; the retained positive-proof location is recorded in `RESULT.json`.

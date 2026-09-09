# Metadata admission repair

Scope: one five-minute, metadata-only pre-release repair starting 2026-09-09 15:39:57 UTC. No actual attempt or release was created, and no tokenizer, model/provider, feature, fit, install, network, or commit work occurred.

## Cause and path contract

The independent full admission fixture passed from repository-root cwd but failed from the real fit-owner cwd, development/native_supervised_gate_v2. Preparation's unchanged verify_prospective_bindings reads Path(pin['path']) directly. The capture manifest supplied a repository-relative external preparation-source pin, so it accidentally relied on cwd.

The capture external pin is now the resolved absolute current-workspace path C:\\Users\\farha\\OneDrive\\Documents\\ChatGPT\\SP_Lense\\development\\native_supervised_gate_preparation_v3\\SOURCE_FREEZE.json. Its target bytes and SHA256 remain 791f62010a74f75df78f907184ba92fc39345e30fcd3663a515ca067a7b092f5. Moving the workspace requires prospective rebinding; no path fallback or relaxed verification was introduced.

## Exact changed files

- capture_v1/SOURCE_FREEZE.json: only external_sources[0].path changes; all 99 local source/fixture pins and the external target digest remain identical. Old manifest cf975c0e65c47b990df7f3d9fbcd489aa747d379999a1ce4f8bd4a82a54c2b42; new b5a29962bb4850950cf95f3f6ac93e78d70954f74779b507165cbc4d061b3a7d.
- gate_v2/source_auth.py: only CAPTURE_SOURCE_SHA256 literal changes. Raw source string comparison proves every other byte unchanged. Old be730ab1df2e18b91d5aa4311d573679c28ecf3d2c3efac38ac401f759000008; new dc333f85b04d383ed3c0542931bbc1844bf5c19cf096b02bdbeb34db02adbcaa.
- gate_v2/SOURCE_FREEZE.json: only dependent source_auth.py and capture-manifest digests change. Old d17db8f257aac681fc264376a869a3d3186dc299f418b75fbfe438ce8aa8447a; new b60a2b58102093e194c63b885b29a946eabfa42b0801be0d86d4f1e5cb531d57.
- gate_v2/ENGINEERING_HANDOFF.md: admission-seam status and current digests updated.
- gate_v2/METADATA_ADMISSION_REPAIR.md: this repair record; excluded from executable source inventories.

All preparation source/manifest bytes, capture function bodies, solver/checker, authored content, old attempts, finite caps and default-deny releases remain unchanged.

## Focused proof

Exact object delta checks PASS for both manifests; raw source comparison PASS for the single literal change. From the actual fit-owner cwd, the old relative pin does not resolve and the new absolute pin does; unchanged support.check_freeze authenticates 99 files, input_reader.adapter authenticates the unchanged preparation source, direct Path(pin['path']) hashing matches, and source_auth's literal equals the new capture-manifest digest. Gate source/external metadata verification PASS. No provider modules imported. CHECKPOINT.json remains text: unset (-text), with its original digest unchanged.

Independent review previously ran the full unmocked 15-check admission fixture successfully from gate_v2 cwd using absolute pins. Independent final comparison PASS_METADATA_ONLY_FINAL_DELTA verified these exact three final hashes, unchanged function bodies/source map/preparation bytes, and every gate join against the prior successful fixture variant. See independent_admission_review/REPORT.md and FINAL_METADATA_RESULT.json. No additional admission run or unchanged broad suite was repeated; no remaining blocker was found within the bounded admission review. No actual launch permission is implied.

Subsequent root author-integration check found a separate uppercase-literal validator mismatch: inherited ASCII_LETTERS admission is narrower than the prospective packet's allowed spaces/digits/punctuation. This is outside this metadata-only repair; root owns the minimal validator-contract fix, tests and downstream pin propagation. The hashes above identify the independently verified metadata-repair snapshot, not a claim of current actual readiness. This agent made no validator or authored-content changes.

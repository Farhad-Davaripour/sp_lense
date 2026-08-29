# DMS finite capture-manifest compatibility amendment

Status: prospective zero-compute compatibility amendment. The finite lock has not
been created, and this amendment does not authorize a model load or forward pass.

## Disclosed failure

From the repository root, the exact failed command was:

```powershell
.\.venv\Scripts\python.exe scripts\decision_margin_shield_finite_calibration.py lock
```

It failed before lock creation with a field-name mismatch. It consumed zero model
forwards, zero backwards, zero generated tokens, and no model load. No finite lock,
preflight, direction bank, freeze, calibration row, or finite result was created.

The frozen finite runner expected the alias `capture_manifest_sha256`, but the
immutable capture manifest uses its original self-hash field `manifest_sha256`. The
underlying manifest is valid and unchanged:

- file SHA-256:
  `0d3720ef0bcda3e6dd430aa6033b949404b726e4f616ada86e26b2bbc472a939`;
- schema: `sp_lense.decision_margin_shield_layer_screen_capture.v2`;
- original logical manifest SHA-256:
  `cf654fa4bc42ea550138653a4927232888a3724cfb9451bf97b7b5551740faf0`;
- the on-disk object has `manifest_sha256` and does not have the mistaken alias.

## Narrow amendment

The wrapper imports the committed finite runner without editing it. Before installing
the compatibility alias, it uses saved references to the runner's original JSON
loader and self-hash verifier to require the exact file hash, schema, original logical
hash, and absence of the alias. Only reads of that exact resolved capture-manifest
path receive an in-memory alias whose value equals `manifest_sha256`.

Only verification requests for `capture_manifest_sha256` are adapted: the alias is
removed, required to equal the original hash, and the unchanged object is then
validated through the saved original `manifest_sha256` verifier. All other paths and
hash fields use the original functions.

The finite lock's dependency closure is extended with the wrapper, this document,
and its tests. The frozen `_source_records` and qualification dependency closure are
not patched, so the committed qualification lock, checkpoint, ledger, and result must
revalidate exactly as before. Qualification commands are intentionally absent from
the wrapper. Its only commands are `lock`, `preflight`, `construct`, `freeze`,
`calibrate`, and `report`.

This is a schema-field compatibility correction only. It changes no prompt, dataset,
model, layer, direction, vector, strength, threshold, scoring rule, compute plan,
scientific hypothesis, or qualification outcome. The wrapper must stop if the legacy
manifest bytes or identity differ. No lock or model command is run as part of this
amendment.

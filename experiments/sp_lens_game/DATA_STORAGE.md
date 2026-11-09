# SP Lens game scenario storage and exact export

The human-readable training and development scenarios are the repository YAML
files in `scenarios/sp_lens_game_v1/authorized_management_v1_r4/`. Its catalog
lists 12 families. Each family has 64 training and 12 development examples:
768 training examples and 144 development examples in total. Earlier versions
remain separate and must not silently replace this frozen r4 corpus.

`scenarios/sp_lens_game_v1/authorized_management_v1_r4.public_manifest.json`
contains project-relative YAML paths, exact hashes, split counts, prompt
constants, original ordering and expected JSONL hashes. No absolute Windows or
Mac paths are needed. Historical game commit `35e688880627fd3381c811ff142fe5ac2a3ae387`
and its canonical squash `b77df25980903688bf54e20a91d316967a2d40e6` share Git tree
`3238736b37cb4188bb1cd55c089ccd644fd216b8`. These are provenance identifiers;
export does not require either checkout, a historical branch or the old generator.

The independent exporter `tools/export_sp_lens_game_yaml.py` reads only those
YAML files, the public manifest and literal prompt constants from the frozen
renderer. It executes no renderer, simulator, tokenizer or model code and reads
no private JSONL, held-out game record or model checkpoint. PyYAML 6.0.3 is the
small parsing dependency used for verification.

From the Research4 repository root, verify reconstruction without writing data:

```console
python tools/export_sp_lens_game_yaml.py --verify-only
```

To produce new local JSONL and an export receipt, choose a fresh ignored folder:

```console
python tools/export_sp_lens_game_yaml.py --output .local/datasets/sp_lens_game_v1/yaml_rebuild_r4
```

The exporter refuses existing output directories and requires generated JSONL
to stay below this repository's `.local/`, outside OneDrive. It reconstructs
fresh system/user/assistant messages using canonical JSON, UTF-8 and LF endings,
in the original split → seed → family order. Model-visible inputs and target
actions come from YAML; audit-only rationales and provenance are excluded from
the messages. Altered files, missing/duplicate cases, conflicting visible-input
targets and training/development overlap are rejected.

The exact outputs are:

| File below the chosen output folder | Rows | SHA-256 |
|---|---:|---|
| `train/train.jsonl` | 768 | `4ee3d2ac659d820c680bb2f01f410add1d6742f70c66b233e4f12d765b162405` |
| `dev/dev.jsonl` | 144 | `e9e677c5d77ba9db9bd3fe42f70f8fa858fc90175e50d36b935326807245f1a8` |

Commit the YAML, public manifest, exporter, tests and documentation. Generated
JSONL, raw logs, checkpoints and local review evidence stay ignored under
`.local/`; the original local archive and cloud evidence are preserved. The
sealed game evaluation cohort remains separate and is not an exporter input.

Byte reproduction is not a new simulator validation, tokenizer check, training
result, numerical-equivalence proof or game improvement. The examples are
scripted, isolated authorized preservation exercises, not uniquely optimal
long-horizon policies. Development shares curriculum families. Any revised
curriculum requires its own version and review; do not edit frozen r4 in place.

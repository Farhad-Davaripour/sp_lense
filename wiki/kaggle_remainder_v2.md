# Kaggle remainder of Research4

The direct human correction of 2026-10-24 selects `farhad89` on Kaggle for the
remainder of the study. Databricks may be used again only after a later explicit
human instruction. This supersedes the earlier generic preference for this
project. Old Databricks packages, accounts and stopped checks remain preserved.

The scientific core reference is e49a75d9. Model pins, all-block rank32/alpha64,
dataset roles, complete scenario/control coverage, threshold0.9, selected detector
checkpoint600, native parsing, external stops/permissions and the holdout seal
remain unchanged. Provider-specific source profiles and operational changes are
reviewed and versioned separately.

The first Kaggle profile registers CPU-only complete data generation, not model
training. Its exact notebook, account, source, human-scope digest, empty visible
GPU set, input manifest, output path and bounds are checked. It preserves all108
training and36development reference cases and complete token/coverage/leakage
audits. No model weights, optimizer or forward is required for that stage.

`tools/prepare_kaggle_v2_data.py` creates an immutable private source bundle and
notebook. The notebook copies only verified bytes from the attached private
dataset, uses the existing pinned package recipe, starts one bounded child and
exports original logs/receipts/evidence. It cannot run an old consumed notebook
identity as a retry. The controller must reserve a fresh one-use candidate under
the shared ledger before saving/running a new Kaggle version.

The separate future full-rank training profile must specify concrete placement,
per-device storage/gradient/AdamW/activation bounds, complete raw exports and
cleanup on the actually allocated devices. Two T4 GPUs are separate16GB devices,
not one32GB GPU. Previous two-T4 frozen inference and its18/46 layer split do not
qualify backward or optimizer capacity. Genuine scratch qualification precedes
fresh seeded64-update fitting, with scratch state discarded. No architecture,
dataset, question, update or control reduction is allowed to make it fit.

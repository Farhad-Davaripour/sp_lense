# Owned V2 model-stage exports

The model worker is source preparation, not a provider submitter. It refuses
missing actual data/source/workspace/model/authority bindings before model work.
Default execution remains on Databricks with the existing one-visible-GPU
profile, original environment/model pins, time/forward bounds and full recipe.
Metadata preparation alone grants no allocation, budget or scientific admission.

Before loading full weights it verifies small metadata and checks complete
download/raw/export disk space, then physically reserves protected export space.
Only its marker-bound scratch is used for caches and raw artifacts. The revised
source profile reserves nine GiB physically and admits up to four GiB of complete
raw evidence, including the full adapter checkpoint and both FP32 AdamW moments.
The unchanged one-GPU/time/forward bounds and architecture remain required.
Actual disk, transfer time, runtime capacity and cost are still unverified.
The V2 fit
helper retains exact native targets and fresh qualification/full-fit boundaries.

Artifacts are written exclusively and read back locally and remotely. A put
whose response is uncertain is reconciled by reading its exact bytes; it is not
blindly written again. If an individual raw transfer failed, a verified complete
archive can still preserve the outcome. If neither raw members nor archive are
confirmed, scratch is retained and export status stays unknown. Retained local
scratch alone is not a durable result and may require provider-side recovery.

Cleanup requires scientific work exit and confirmed durable evidence, then
checks the resolved absolute owner path and marker. It never treats scratch
cleanup as proof the provider released the GPU. Unknown/failed exports and
scientific failures do not receive automatic model retries. Periodic forward
journals are lower bounds; only a controlled exit provides exact final counters.

Public tests inject tiny fake services/stores, force scientific admission false
and never load a pretrained model, allocate compute, generate production cohorts
or open a seal. Actual model capacity, complete raw model audit and scientific
success remain unestablished until the separately admitted real run.

The worker now exports fresh live original-parameter hashes/flags/version
availability, pinned runtime/quantization metadata, named final optimizer state,
and all six raw OFF/bare/ON logit arrays without additional model calls. Large
file uploads/readback use streams and SHA/size reconciliation. A final readback
rechecks even previously confirmed objects; an unknown write is never repeated.

Before model loading, the controller must publish the exact source/phase/host/
destination/run-ID binding at the plan's `PROVIDER_BINDING.json` path. This is
neither allocation nor release authority. Final qualification additionally
requires the independent reader to reconcile matching AIR and Jobs terminal
success against the recorded run ID and raw archive digest.

Full-fit source preparation remains inert. The runtime must retrieve the exact
scratch source package, raw archive and saved provider responses, reconstruct
metadata as data without importing its code, and independently re-audit the
complete scratch qualification before reading fitting rows. A serialized
`audited=True` closeout is insufficient. The qualification descriptor names:

- `source_commit`, `workspace`, `data_release_sha256`, `provider_run_id`;
- `source_archive.path` and `.sha256` under `model_sources_v2/<commit>/qualification/`;
- `raw_archive_path` and `raw_archive_sha256` under the exact scratch phase;
- `AIR_response_path` and `Jobs_response_path` in that same owned phase.

`qualification_descriptor` derives the one-update metadata count from the
reader's verified scratch count, keeping it separate from the auditor's zero
optimizer activity. It also retains the exact source-release/input-manifest
hashes and reconciled provider ID; it refuses public/incomplete evidence.

The Databricks Files API documents uploads up to five GiB, above this proposed
four-GiB evidence envelope. This documentation is not a measured throughput or
cost guarantee. [Files API reference](https://docs.databricks.com/api/files/v2/upload-file)

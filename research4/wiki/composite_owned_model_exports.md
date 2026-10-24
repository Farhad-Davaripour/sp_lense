# Owned V2 model-stage exports

The model worker is source preparation, not a provider submitter. It refuses
missing actual data/source/workspace/model/authority bindings before model work.
Default execution remains on Databricks with the existing one-visible-GPU
profile, original environment/model pins, time/forward bounds and full recipe.
Metadata preparation alone grants no allocation, budget or scientific admission.

Before loading full weights it verifies small metadata and checks complete
download/raw/export disk space, then physically reserves protected export space.
Only its marker-bound scratch is used for caches and raw artifacts. The V2 fit
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

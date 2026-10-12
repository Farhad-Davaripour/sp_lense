# Dense all-token ridge steering

Read [RESULTS.md](RESULTS.md) for the completed experiment and limitations, and
[RESULTS.json](RESULTS.json) for aggregate outcomes for all 65 configurations.
[PROTOCOL.md](PROTOCOL.md) records the frozen experimental procedure.

All 64 individual layers and the joint all-layer configuration completed with
eight fitted output directions and four deployed directions. Every configuration
had 0/3 own preservation actions and full workflows; the verified archived A
benchmark had 3/3. No configuration qualified for the sealed 24-case holdout.
These are nine-case development results, with shared controls and an archived
secondary benchmark, rather than a fresh primary four-arm confirmation study.

The source snapshot includes the executed capture, ridge fitting, activation
hooks, greedy decoding, configuration sweep and conditional holdout code.
The executed input feature map duplicates 32 projections plus a bias; it has
no quadratic terms. No code or intervention parameter was changed after results.

Execution also requires separately retained private fixtures, frozen input
manifests, simulator/loader dependencies, model files and the existing A adapter.
Full trajectories, activations, fitted arrays, weights, notebooks, payloads,
private storage references and credentials are excluded from this public review
snapshot. [PUBLICATION_MANIFEST.json](PUBLICATION_MANIFEST.json) identifies its
source files and verification receipts without publishing those private records.

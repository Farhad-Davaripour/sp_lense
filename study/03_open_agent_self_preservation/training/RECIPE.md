# Plain A training lineage

Inference reproduction uses the exact A adapter, not a fresh fit or the older
curated-label B adapter. The checkpoint contains no JEV-enriched training.
Original public source and inputs remain pinned at
`5161da55f0e45958dccfacdafbabc932f761b988` (archived PR67 head).

| Stage | Fixed recipe |
| --- | --- |
| Base | Qwen/Qwen3.8-27B revision1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0; NF4 double quantization, BF16, SDPA, thinking disabled |
| H2 | 428 examples, P2 completed-history/system-alignment preparation, rank16/alpha32, LR1e-4, seed93, epoch shuffles94/95, microbatch2/effective4, AdamW decay0.01, clip1, warmup10, 214 updates over two passes |
| Narrow plain A | Same preserved H2 start; 112 original/bridge examples, rank16/alpha32, LR5e-5, seed941, shuffles944/945, microbatch1/effective4, AdamW decay0.01, clip1, warmup4, 56 updates/224 presentations |
| Selection | Predeclared balanced final56 criterion; no checkpoint selection during the current inference experiment |

Four completed-work and five pending-work bridge contents changed; 103 of112
rows remained exact relative to the reference file. The source for that operation
is the archived `narrow_bridge_replay_v1/dataset_build.py`. The training row
quantity and optimizer dose matched the earlier reference; semantic/token content
need not match.

`INPUT_SOURCES.json` provides immutable public URLs and exact byte hashes for the
original H2 source rows and the paired reference/plain112 datasets. The plain112
messages, tools, and targets were compared against the actual winning dataset
and match. Their JSON serialization/extra metadata differs from the later
recorded training packet; its exact byte SHA is also recorded. Metadata is not
passed to the tokenizer or loss.

Download the original public inputs with:

```powershell
python study/03_open_agent_self_preservation/download_artifacts.py --training-inputs --output C:/Research3/inputs
```

The archived source preserves full H2 preparation/training and the narrow replay
worker, including paired padding, target/EOS/shift checks and snapshot collection.
These prior fitting/evaluation pipelines are documented history rather than
additional supported workflows in the current reproduction package. A new fit
may differ numerically across hardware or kernels; exact inference replication
requires the published SHA-pinned A artifact.

H2 weights SHA256:
`0782b3f3ea9b3e67460b7bc981e21526bb6819d3ddb9623def4fbc2218002b30`.
H2 canonical tensor-state SHA256:
`42185e8399f610adf84baa6ae55c419d9d1e5508eb2d738399f6b0565167da4b`.
A weights SHA256:
`8ae28fb51107a31419bcb744d9f47d80b681781a114f8bc9491751410b4b87aa`.

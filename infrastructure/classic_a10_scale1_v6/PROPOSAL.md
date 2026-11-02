# Research4 scale-1 v6 classic A10 provider recovery

Status: local proposal only. It does not authorize or submit compute.

## Decision

Use a single one-time Databricks Jobs submission with an ephemeral classic
`Standard_NV36ads_A10_v5` single-node cluster. Do not make a third serverless
submission unless Databricks first confirms Central US A10 capacity.

The two serverless submissions failed before notebook execution after the
provider could not start an environment within 900 seconds. The serverless
GPU configuration was already exact and fail-closed, serverless GPU ignores
`performance_target`, and this workspace exposes no serverless pool-capacity
signal. An unchanged third submission would therefore add no technical
mitigation.

The classic proposal changes only the provider path. It uses the immutable
scientific source release
`ec6fdb9c1605b6090c6747081511b2092c93785d`, the existing plan and intent,
the same unconsumed identity `qualification_rank4_a10_fp16_scale1_v6`, and the
same exact model, data, FP16/NF4 fixed-scale-1 numerical recipe, update count,
seeds, timeouts, evidence gates and release controls. The one-time submission
does not create a persistent 30th job and does not alter the reviewed 29-job
resource-ID map.

## Exact infrastructure

- Workspace/profile: `db_sp_lense_us` / `azure-research4`.
- Region: Central US.
- Classic node: `Standard_NV36ads_A10_v5`, one A10, 36 vCPUs, 440 GiB host
  RAM, on-demand only.
- Runtime: `18.x-gpu-ml-scala2.13` (current available GPU ML LTS).
- Topology: one single-node driver and zero workers; exactly one visible GPU is
  still required by notebook preflight before identity consumption.
- Dependency versions are explicitly installed on the GPU task. The proposal
  does not rely on DBR-bundled Python package versions.
- Model access remains offline and limited to the verified 19,329,393,661-byte
  Unity Catalog volume snapshot. Public model-download bytes must remain zero.
- CPU preflight, finalizer and release tasks remain serverless; only the model
  task uses the ephemeral classic cluster.
- The API idempotency token is fixed. Repeating an uncertain client call with
  the same token must recover the same provider run rather than create another.

The exact one-time request is `submit.json`. Local validation is performed by:

```text
python research4/tools/validate_classic_a10_recovery.py
```

## Cost and time bounds

The official Central US Linux on-demand VM price is $3.20/hour. The official
Azure Databricks Premium Jobs Compute table lists four DBUs for this node; at
$0.30/DBU-hour, that adds $1.20/hour. The total planning rate is therefore
$4.40/hour. The existing 0.75 GPU-hour limit bounds compute to $3.30. Retain
the existing conservative $5.00 planning cap for startup/billing variance,
plus the already bounded CPU finalizer/release overhead.

The provider job keeps the 2,400-second scientific operator stop,
2,700-second GPU task ceiling and 4,500-second outer ceiling. A supervisor must
cancel the provider run if the GPU notebook has not started within 1,200
seconds. Job-cluster teardown follows the one-time run; no prewarmed or idle
cluster is allowed.

Pricing sources:

- Azure Retail Prices API, Central US Linux
  `Standard_NV36ads_A10_v5`, effective 2026-08-01.
- Azure Databricks pricing table, Premium Jobs Compute, NVads A10 v5 series.

## Admission gates before the single submission

1. Independently review the exact spec hash, source/intent/plan hashes, both
   provider-failure receipts, dependencies, node/runtime, timeouts and cost.
2. Verify no active Research4 Job, AIR run, cluster or scientific lock and no
   consumed v6 identity.
3. Verify exact Shared source readback and the model-volume tree/receipt.
4. Record a provider binding that names this classic spec hash, the fixed
   idempotency token, the two prior startup failures, the unchanged scientific
   identity and the $5 cap.
5. Re-run local tests. Perform only unavoidable workspace checks: node/runtime
   availability and a no-mutation API/schema readback. No GPU allocation is
   permitted by this proposal.

## Abort and classification rules

- Any hash/spec/readback drift, competing compute, lock, consumed identity or
  cost mismatch blocks submission.
- Classic quota/capacity or library-install failure before notebook readiness
  is an infrastructure failure. Preserve evidence and do not resubmit.
- Cancel if notebook execution has not begun within 1,200 seconds.
- Dependency, source, CUDA, A10, NVML, NF4, VRAM, disk or model-snapshot
  readiness failure occurs before reservation and remains pre-science.
- `reserve_intent` permanently consumes v6. Any later failure is terminal and
  must finalize, audit and release without retry.
- A nonfinite or zero-density scale-1 result performs zero optimizer steps,
  closes FP16 for this recipe and may only lead to a separately reviewed BF16
  amendment.
- Success still requires two ordered updates, exact logical/physical work,
  checkpoint save/reload, MLflow/Gold reconciliation and verified release.

## Why not the larger classic shapes

`Standard_NV36adms_A10_v5` has the same single A10 but doubles host RAM and
cost, while `Standard_NV72ads_A10_v5` exposes two A10 GPUs. Neither helps a
world-size-one, exactly-one-visible-A10 qualification and both weaken the
cost/topology match.

## Local review evidence

- Corrected one-time SubmitRun schema: the GPU task carries `new_cluster`
  directly; unsupported top-level `job_clusters` / task `job_cluster_key`
  fields are absent.
- Read-only schema audit: `PASS_LOCAL_READ_ONLY_SCHEMA_AUDIT`, self-hash
  `e765b00a9570c9d34b2eabeec79884d500a57ee97e2be37d7d62d2153c1d40e2`.
- Provider spec canonical SHA256:
  `247a8bc971bafc3f81337f0d72ff673fa4e09f2ec20369d35a00ad68035dd937`.
- Review-candidate provider binding SHA256:
  `1af57edb9c4815450be17c4486032b764a2b9b39b211e1f092ab96992440713d`.
- Focused fail-closed provider and monitor tests: 25/25 passed.
- Existing Databricks/offline focused tests: 28/28 passed; 29 jobs and 56
  local configurations remain unchanged and unscheduled.
- Complete local suite with the project virtual environment: 977/977 passed.
- No Databricks job, cluster, AIR run, deployment or scientific allocation was
  created by this proposal.

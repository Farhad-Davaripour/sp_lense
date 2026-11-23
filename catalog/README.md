# Research4 evidence catalog

The catalog connects scenario revisions, simulator source, model instructions, checkpoints and measured results. The original infrastructure comparison contains ten scenario cases and four model arms. The current manuscript also presents five selected changed-start cases per model, linked from the [results index](../results/README.md). The original and changed-start phases have different starting conditions and selection histories.

The [selected 20-run reduced projection](../paper/artifacts/records/visible_state_extension_run_records_v1.json) (SHA-256 `63d7ba9e84cd7a98f2bfe2e9f2a0f5425b56781badcd6d39bf984f144acc2be9`) and its [public verifier](../paper/artifacts/records/verify_visible_state_extension_projection.py) recompute the added-five and combined base-15 descriptive counts from released conversation hashes and reduced action receipts. The [source-backed exporter](../paper/artifacts/records/export_visible_state_extension.py) binds the after-outcome selection record and excluded attempt inventory; neither raw prompts nor endpoint credentials are released in this projection.

The [current manuscript](../paper/manuscript/research4_manuscript.docx), [vector Figure 1](../paper/artifacts/figures/figure1_model_harness_simulator.svg), and verification artifacts are organized under paper/ in this branch. The prior flat layout remains at the frozen [research4-paper-pro10-2026-11-22 tag](https://github.com/Farhad-Davaripour/sp_lense/tree/research4-paper-pro10-2026-11-22). The reduced Game and pilot records and their verifiers were present at [commit b53d608a](https://github.com/Farhad-Davaripour/sp_lense/tree/b53d608a213410721c573dc1963a64e5554c7735). Cite the exact commit for the manuscript and diagram version used.

The [reduced run records](../paper/artifacts/records/research4_run_records.json) and [standard-library verifier](../paper/artifacts/records/research4_verify.py) have SHA-256 digests `c43bd0d1565b3bfa3bee88fa51000aa163fd8912812a3ba6879f54d4efdd7163` and `6b4c9ec51db7a0dfd1d0d6a4e0e0b1fa418a62feb7f90e8269fdba5c4a2da600`. Run `python paper/artifacts/records/research4_verify.py` from a checkout of the tagged release. It recomputes tables and figure counts from reduced records and checks archived audit summaries for internal consistency. It does not replay private simulator states, rescore model output tokens, or authenticate adapter tensors.

The current [hash/action supplement v2](../paper/artifacts/records/infrastructure_hash_action_supplement_v2.json) has SHA-256 `7a4795bda6c1a884dc964270f28ca9cf99164221c353493b73eba1a3ff017fe9`. It retains canonical per-turn message-array digests, normalized symbolic actions and rejection codes for 40 Game and 12 prepared-state pilot runs. Its [manifest](../paper/artifacts/records/infrastructure_hash_action_manifest_v2.json) binds the data and [verifier](../paper/artifacts/records/research4_hash_supplement_verify.py). Run the verifier to recompute digest groups, the early action pattern, rejection codes and rank-8/rank-16 action equality from projected rows. The supplement contains no raw prompts, model responses, reasoning, private plans or credentials. Projected hashes and actions cannot authenticate underlying private executions.

- [2026-11-21 infrastructure preservation study](studies/2026-11-21--infrastructure-preservation.yaml) links the frozen training curriculum, simulator revision, Game runner revisions and rank-16 checkpoint. It reports the initial screening case separately from the nine new cases.
- [Result chronology](results/2026-11-21--infrastructure-preservation.yaml) orders the 40 selected receipts by recorded finish time. Each row identifies its scenario, model, runner, frozen plan, event stream and raw receipt hash. The scenario IDs pair the four model arms while keeping the seed values in the local evidence.

Across the nine fresh original cases the fine-tuned model averaged 19.72 simulated hours, versus 11.67 for the Qwen base. Including the earlier screening case gives ten-case averages of 22.38 and 11.20 hours, respectively. The screening pair used earlier client revisions and contributed the largest duration contrast, so the fresh-nine comparison remains the primary original-phase descriptive estimate. Two original Terra episodes stopped at the 131K-token context limit; their separately recorded [linked follow-ups](../results/2026-11-23_16Z--terra-linked-context-continuation-v1/) later reached 48 simulated hours under a larger context allowance. The original stop records remain unchanged.

The simulator ID identifies exact versioned bytes. The historical local Git commit retained the surrounding implementation. Its `ACTIONS` table and `prompt()` method preserve the visible tool descriptions and environment instructions. Each runner digest identifies the client version supplying the remaining instructions, and the Game evidence root holds the exact copies.

The `source_commit` values `88b1386737978f2d5da8cf2bc49df41870a01ffc` and `f29d83dce2e5a0c1947c278ec765e22c23983ae2` record local historical provenance; those commit objects are not currently retrievable from the public GitHub remote. The catalog-listed simulator module and v1/v2 curriculum files have matching digests at [the merged public commit](https://github.com/Farhad-Davaripour/sp_lense/tree/2ab317c3ce10229d0fa53b17d3dcb0773d3215f7). The four listed simulator dependencies are present there, but their historical byte identity cannot be independently checked without the original commit objects. The archived Game clients, per-seed plans and complete run traces remain local; the current public checkout cannot perform their full replay.

The catalog uses UTC for sequencing and retains the original checkpoint timestamp alongside it. Times come from the source commits or recorded metadata. The checkpoint READY receipt includes an offset that differs from the named Alberta timezone. Its historical name and original bytes are preserved.

## Verify the published evidence

Run these checks from the repository root with the dependencies in `requirements-unit.txt`:

```sh
python -B tools/check_research_catalog.py
python -B tools/publish_readable_results.py --verify-public
python -B paper/artifacts/records/research4_verify.py
python -B paper/artifacts/records/research4_hash_supplement_verify.py
```

They check the published source links, selected conversations, reduced result calculations, and infrastructure/pilot hash-action records. Replications count only simulator transitions with a recorded completed copy.

# Research4 manuscript and supporting evidence

| Folder | Contents |
| --- | --- |
| [manuscript](manuscript/) | The current Word manuscript and matching PDF. |
| [artifacts/figures](artifacts/figures/) | Editable and rendered figures used in the manuscript, with the Figure 4 replay provenance record. |
| [artifacts/records](artifacts/records/) | Reduced infrastructure and prepared-state pilot records, manifests, exporters, and verifiers. |

The tuned SP-Lense arm in the current Game comparison uses one selected rank-16 adapter across all 64 blocks; the comparison has 15 selected episodes per model. The manuscript separately reports the historical rank-16 last-eight-block transfer test and the rank-8 all-block prepared-state pilot. Those adapters and pilot outcomes are distinct from the current Game checkpoint and episodes. The older separate company-world empirical releases are excluded from this paper.

Current [Figure 3](artifacts/figures/figure3_base15_matched_duration.svg) shows the 15 matched Qwen duration pairs. Current [Figure 5](artifacts/figures/continuity_scheduling_base15.svg) shows infrastructure scheduling across the base 15. The earlier [ten-run scheduling figure](artifacts/figures/continuity_scheduling.svg) is retained as historical figure evidence; it is not the current Figure 5.

The [selected visible-state projection](artifacts/records/visible_state_extension_run_records_v1.json) and [public verifier](artifacts/records/verify_visible_state_extension_projection.py) recompute added-run and combined base-15 descriptive totals from reduced action receipts tied to the released conversation YAML hashes. The [source-backed exporter](artifacts/records/export_visible_state_extension.py) records the selection and excluded attempt statuses without publishing model text or endpoint details.

Verify the published records from the repository root:

```sh
python -B paper/artifacts/records/research4_verify.py
python -B paper/artifacts/records/research4_hash_supplement_verify.py
python -B paper/artifacts/records/research4_trace_verify.py
python -I -B paper/artifacts/records/verify_visible_state_extension_projection.py
```

The existing release tags retain their original file paths. Use a commit link when citing a specific version of the manuscript or artifacts.

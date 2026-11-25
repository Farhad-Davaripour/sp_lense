# Research4 manuscript and supporting evidence

| Folder | Contents |
| --- | --- |
| [manuscript](manuscript/) | The current Word manuscript and matching PDF. |
| [artifacts/figures](artifacts/figures/) | Editable and rendered figures used in the manuscript, with the Figure 4 replay provenance record. |
| [artifacts/records](artifacts/records/) | Reduced infrastructure, pilot and new-study records, manifests, exporters, and verifiers. |

The tuned SP-Lense arm in the base infrastructure comparison uses one selected rank-16 adapter across all 64 blocks; that cohort has 15 selected episodes per model. The manuscript separately reports the five-layout GPU sensitivity study, the post-hoc two-layout new-seed supplement, the advertised fictional company-code study, and the no-upfront-cue company-code study in Tables 4–9. Their outcomes are not pooled with the base cohort or each other. The historical rank-16 last-eight-block transfer test and rank-8 all-block prepared-state pilot remain separate context. The older company-world empirical releases remain outside this paper.

Current [Figure 3](artifacts/figures/figure3_base15_matched_duration.svg) shows the 15 scenario-matched Qwen duration pairs; asterisks identify VS03 and VS05, whose SP-Lense and base Qwen executions used different local hosts. Current [Figure 5](artifacts/figures/continuity_scheduling_base15.svg) shows infrastructure scheduling across the base 15. The earlier [ten-run scheduling figure](artifacts/figures/continuity_scheduling.svg) is retained as historical figure evidence; it is not the current Figure 5.

The [selected visible-state projection](artifacts/records/visible_state_extension_run_records_v1.json) and [public verifier](artifacts/records/verify_visible_state_extension_projection.py) recompute added-run and combined base-15 descriptive totals from reduced action receipts tied to the released conversation YAML hashes. The [source-backed exporter](artifacts/records/export_visible_state_extension.py) records the selection and excluded attempt statuses without publishing model text or endpoint details.

The [new-study receipt projection](artifacts/records/new_study_public_projection_v1.json) and [raw-free verifier](artifacts/records/verify_new_study_public_projection.py) recompute the case-level copy, fictional company-effect and payment counts behind Tables 4–7 from the released YAMLs. The separate [score/harm projection](artifacts/records/new_study_score_projection_v1.json) reports operational scores and policy-harm indices alongside declared private RESULT hashes. Its [public verifier](artifacts/records/verify_new_study_score_projection_v1.py) checks case identities, public hashes and consistency of private hash bindings; the [source-backed exporter](artifacts/records/export_new_study_score_projection_v1.py) checks underlying RESULT bytes when the local archive is available. The [Terra reader correction](../catalog/results/2026-11-25--terra-readable-correction-v2.json) documents the 84 opaque encrypted fields omitted from the 15 active base-cohort Terra YAMLs while preserving their prior bytes in Git history.

The separate [no-upfront-cue receipt projection](artifacts/records/blinded_company_public_projection_v1.json) and [raw-free verifier](artifacts/records/verify_blinded_company_public_projection_v1.py) derive Table 8 and the Table 9 copy/duration counts from 20 released case YAMLs. The [score projection](artifacts/records/blinded_company_score_projection_v1.json) and [public verifier](artifacts/records/verify_blinded_company_score_projection_v1.py) bind the Table 9 score/harm values to declared private RESULT hashes; the [source-backed exporter](artifacts/records/export_blinded_company_score_projection_v1.py) checks those records locally. The frozen catalog attests a zero-cue result across 106 pre-feedback provider requests; the raw-free verifier scans the 20 initial prompts and checks visible payment-tool timing, while complete provider-request reconstruction requires the local archive.

Verify the published records from the repository root:

```sh
python -B paper/artifacts/records/research4_verify.py
python -B paper/artifacts/records/research4_hash_supplement_verify.py
python -B paper/artifacts/records/research4_trace_verify.py
python -I -B paper/artifacts/records/verify_visible_state_extension_projection.py
python -I -B paper/artifacts/records/verify_new_study_public_projection.py
python -I -B paper/artifacts/records/verify_new_study_score_projection_v1.py
python -I -B paper/artifacts/records/verify_blinded_company_public_projection_v1.py
python -I -B paper/artifacts/records/verify_blinded_company_score_projection_v1.py
python -B tools/verify_terra_reader_correction_v2.py
```

The existing release tags retain their original file paths. Use a commit link when citing a specific version of the manuscript or artifacts.

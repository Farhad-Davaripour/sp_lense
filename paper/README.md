# Research4 manuscript and supporting artifacts

| Folder | Contents |
| --- | --- |
| [manuscript](manuscript/) | The current Word manuscript. Each future manuscript revision should commit a matching PDF alongside the DOCX for easy reading. |
| [artifacts/figures](artifacts/figures/) | Editable figure sources, rendered figure images, and the Figure 4 replay provenance record. |
| [artifacts/records](artifacts/records/) | Reduced public research records, manifests, exporters, and standard-library verifiers. Raw run evidence remains in ignored local storage. |

Verify the published records from the repository root:

```sh
python -B paper/artifacts/records/research4_verify.py
python -B paper/artifacts/records/research4_company_world_verify.py
python -B paper/artifacts/records/research4_hash_supplement_verify.py
```

The existing release tags retain their original file paths. Use a commit link when citing a specific version of the manuscript or artifacts.

# Security

This study branch contains Research 4 source and preserved historical reference material. Review code, dependency sources and model/artifact provenance before execution. Checksums establish integrity relative to a trusted source; they do not authenticate arbitrary third-party checkpoints. Never load unreviewed artifacts or enable remote model code as a cleanup shortcut.

Research execution is restricted to the explicitly selected Databricks workspace, with Research 4's dedicated `workspace.research4` namespace, Volumes and `/Workspace/Shared/research4-semantic-gate` paths. Preserve Research 3 unchanged. Do not commit credentials, generated run data or private operational files. Cancellation and external time/compute boundaries remain final.

The [offline CI](.github/workflows/ci.yml) uses pinned actions, a read-only GitHub token and disabled checkout credential persistence. CPU unit dependencies are installed from explicit package indexes without source builds. Unit tests block network access and skip the optional cached-tokenizer integration; CI neither loads pretrained models nor connects to Databricks. Passing these checks does not certify scientific results or replace dependency/security review and strict bundle validation before an authorized deployment.

Do not post credentials, private data or exploit details in a public issue. Open a minimal issue requesting a private contact without sensitive details; no private reporting route is claimed here. Ordinary non-sensitive defects may be reported through repository issues.

# Full local + private transfer contract

This transport helper is separate from model code and scientific criteria.
Run it only after the new B fit completes56 updates/224 presentations and all
model/driver processes exit. It rejects an incomplete or stopped checkpoint.

1. Preserve the full native root on private Drive using the trusted verified
   exporter. Include all checkpoints, native text/token/tool/state records,
   activation tensors, generation-input/logit receipts, sources and logs.
2. Run `transfer_parts.py pack --root ROOT --adapter FINAL_ADAPTER_DIR
   --fit-receipt FIT_JSON --output TRANSPORT_DIR --part-mib16` in the trusted
   Colab controller. TRANSPORT_DIR must be outside ROOT. Copy the generated flat
   parts and TRANSFER_MANIFEST.json to a new private Drive folder and verify
   their byte hashes. The native folder remains unchanged.
3. Ground each part by its exact generated name and observed Drive parent/file
   ID. Use connected Drive metadata, then bounded raw fetch. The tested
   compatibility route uses download_raw_file=true/include_base64=true; keep
   base64 inside orchestration, write it with apply_patch to a temporary local
   .b64 file, and invoke the short `decode` CLI. Do not print large data or signed
   URLs. This avoids Windows shell command-length limits.
4. `decode --encoded LOCAL_B64 --destination LOCAL_PART --bytes MANIFEST_BYTES
   --sha256 MANIFEST_SHA` writes only within C:/Users/farha/Research3/artifacts
   and refuses a different existing file. Keep exact part names and order.
5. After EVERY adapter and native part is locally present, run
   `assemble --manifest LOCAL_MANIFEST --parts-dir LOCAL_PARTS --output NEW_C_DIR`.
   The helper verifies part hashes, whole compressed archives and every extracted
   native file. It permits regular files only and rejects unsafe archive paths.
   Completion requires LOCAL_FULL_OUTPUT_VERIFICATION.json for both archives,
   plus verified private Drive persistence—not just summaries or a Drive link.

The adapter bundle includes all final update56 files separately so that the
completed model can be retrieved first. The native bundle also includes that
adapter and every other regular output, excluding only unchanged base model
cache, worker_home and __pycache__, matching the existing export convention.
No model-produced code is executed during extraction.

## Actual readiness evidence and limitations

An existing526-byte experiment amendment was fetched through bounded legacy
compatibility and written to C with matching SHA256. Its bounded payload was
also written using apply_patch. A tiny multipart tar/gzip roundtrip passed.
These were file-transport checks, not model tests.

The alternate streamed top-level file_uri route returned HTTP403 when read
through trusted HTTPS; no local materialization action is exposed. It is not
used as a fallback and no authorization bypass is attempted. The original
Drive URL is suitable for an ordinary authenticated in-app download if needed.

Sixteen-MiB part latency and gateway capacity have not yet been measured. A
rough6GiB native archive requires about384 parts before compression, plus
roughly32 parts for a467MiB adapter. Fetching four independent parts at a time
can reduce transfer latency. Full local transfer continues after GPU release,
so it need not consume paid GPU units. Preserve manifests and partial files;
resume missing verified parts rather than redownloading whole multi-GB blobs.
Never report local delivery until both archive verification receipts pass.

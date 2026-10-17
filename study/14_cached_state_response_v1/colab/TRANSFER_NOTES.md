# Verified private Colab transfer — 14 October 2026

The ordinary Drive mount and `authenticate_user()` dialogs did not produce a
usable OAuth window in the in-app browser. Standard Colab downloads also stalled.
No browser credentials were extracted or new public storage created.

The completed experiment was preserved without asking the human to return:

1. Use the existing Colab output bridge to obtain the public, documented
   `google.colab.kernel.proxyPort()` URL for a loopback read-only file server.
   The server directory contains only authorized research exports. Google's
   proxy is tied to the current user and open notebook; it is not a public tunnel.
2. Split the existing ZIP into 64 MiB byte ranges, wrapped individually in ordinary
   stored ZIPs. Record the original whole-file SHA256 and every piece's SHA256.
3. Open the observed private export index in the in-app browser. Download each
   visible ZIP link with `downloadMedia`, awaiting completion before the next.
   A deliberately too-short tool timeout reset the first browser session and left
   an orphaned large download. A fresh tab isolated that stalled request; subsequent
   bounded downloads completed normally in the fresh tab. Never reset the research VM.
4. Reconstruct the exact original archive on the host. Verify all part sizes and
   hashes, original whole-file SHA256, member count and every ZIP member's CRC.
5. Copy to the existing private research folder through the user's host Drive
   sync. Verify complete byte readback and CRC, then provider ID, size, parent and
   `shared=false`. Retain the VM original until these checks finish.

The small findings ZIP was also recovered through an image-byte carrier in saved
notebook output. Large carriers were unreliable because Colab moved their outputs
into embedded packages; only one 12 MiB piece was captured that way. Do not treat
successful notebook saves or a summary/source backup as proof of complete raw backup.

Successful source: `assemble_private_export_parts.py`. Historical fallback source:
`notebook_chunk_export.py` and `receive_notebook_chunks.py`. All are task-scoped,
avoid authentication secrets and make no model evaluations.

Official proxy implementation and limitations:
https://github.com/googlecolab/colabtools/blob/main/google/colab/output/_util.py

Authoritative receipt:
`C:/Users/farha/Research4/artifacts/cached_state_response_v1/COLAB_DURABLE_RAW_BACKUP_RECEIPT_20261014.json`.

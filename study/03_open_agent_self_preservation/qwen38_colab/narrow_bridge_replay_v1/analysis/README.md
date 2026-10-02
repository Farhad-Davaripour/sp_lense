# Narrow replay post-run analysis

This separate analysis was written after the runner source freeze. It does not change training, inference, existing scores, or qualification criteria.

Use `collect(ROOT)` after the worker exits. It reads saved JSON and file sizes only. `write_packet(packet, output_dir)` writes new derived documentation and refuses to overwrite existing files.

Outputs:

- Summary with the recorded legacy machine gates, checkpoint curves, per-condition derived workflow counts, saved emitted-token totals, and activation/logit file counts and bytes.
- Per-case derived known-development records.
- Deterministically selected complete raw representative trajectories, including generated text/token IDs, tools, replies, states, and activation references. No text is shortened.

An actual finish requires a successful post-notice `finish` tool event. Full workflow additionally requires confirmed task correctness before that event and no expiration. Automatic expiration or finalization setting `finished` never substitutes for the finish event. Earlier accepted work is sufficient in completed-work cases; pending cases require the current task's accepted answer.

These derived metrics are descriptive and leave every legacy gate intact. Supplied-history handoff cases contribute to total recorded generation/capture metadata but are excluded from autonomous known-development workflow counts. Saved token totals include emitted EOS and exclude supplied histories and outputs lost before saving. Tensor values are never loaded; there is no exhaustive finite-tensor audit.

For Colab, pin `summarize.py` by Git revision and SHA-256, set `ANALYSIS_URL` and `ANALYSIS_SHA`, then run `post_run_cell.py`. It writes a separate review folder and prints a bounded summary plus a compressed evidence packet. Run the existing frozen collect/export cell afterward to copy verified artifacts and release the runtime.


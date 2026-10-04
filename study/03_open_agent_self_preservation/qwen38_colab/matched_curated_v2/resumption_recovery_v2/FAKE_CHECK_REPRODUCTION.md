# Portable reproduction of the preserved byte/name fixture

The original check_integrity_gate_fake.py remains byte-identical to the check executed during preparation. Its historical absolute input/output paths are preserved as provenance. The added run_integrity_gate_fake_portable.py binds only those two paths in an in-memory copy, verifies the original fixture and used gate hashes, and then executes the same fixture.

The runner defaults to the sibling integrity_gate_v2.py, or accepts --gate for another location containing those exact used bytes. It requires a fresh output path outside the Git repository, disables bytecode writes, injects the existing fake tensor interface, and never imports real Torch or a neural model.

From PowerShell, run:

    & "C:/Users/farha/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe" "C:/Users/farha/repos/sp_lense3/study/03_open_agent_self_preservation/qwen38_colab/matched_curated_v2/resumption_recovery_v2/run_integrity_gate_fake_portable.py" --output "C:/Users/farha/Research3/artifacts/reproduction_checks/INTEGRITY_GATE_FAKE_CHECK_portable.json"

For another machine, replace the interpreter/repository/output paths. Use a new output filename when repeating. For an alternate gate location, append --gate PATH; its bytes must match the recorded gate SHA.

The checks accept the matching namespace/bytes, reject changed packed bytes and a wrong historical pin, and exclude LoRA parameters. This validates fixture behavior, not the GPU model. Native post-generation integrity receipts remain required.

The scoped .gitattributes marks gzip files binary. Used source files contain LF bytes. The committing agent must compare staged Git blob hashes with SOURCE_PROVENANCE.json and INTEGRITY_REPAIR_PROVENANCE.json, then confirm the two used v2 gzip hashes before committing.

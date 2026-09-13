# Capture reuse: model-free candidate
Two DeepSeek authoring jobs ended before writing candidate changes: first response cap (16 tools), second work cap (21 observed/20 allowed). Both SDKs closed/exit0 and their PIDs are absent. No third paid authoring retry was sent.

Supervisor diagnosed the operational blocker: neither job reached implementation within its fixed limits. Reused 25 original runner files with every copy hash checked, then made seven binding-only changes:
- input_reader.py: use the exact accepted prepared payload through hash-before-exec parent reader loading, with source-inventory pin and release joins.
- authority.py: require the accepted checkpoint freeze to match the release; other guards retained.
- support.py, fit_source_auth.py, audit_saved.py: new attempt/namespace.
- Both owned identity JSONs: relocated executable path; original verified binary hashes unchanged.

The remaining 18 copied files are byte-identical. New SOURCE_FREEZE.json is an UNAPPROVED candidate inventory; no root_release/RELEASE.json, real_evidence, model loading or scoring exists. No historical file changed.

Seven focused binding tests pass. Combined with the prior52, all59 tests pass, exit0,0.970s; no failures/errors/skips. Missing-release real entry preflight returns DISABLED_NO_ROOT_RELEASE/model_work:false, exit2 as expected. Historical tracked readout/transfer Git diff is empty.

See CAPTURE_BINDING_DIFF.patch for the complete old-to-candidate diff. Test command uses repo .venv Python -E -S -B and unittest: test_capture_bindings, test_prepared_input_reuse, test_checkpoint_reuse, test_diagnostic_scoring, test_diagnostic_reader, test_reader_pins, test_reader_join, test_math_equivalence.

Next: independent review of prepared-input reuse and capture bindings, including downstream row/scorer compatibility. No full historical provenance replay is claimed. Separate prospective commit/release and preflight remain required before any model work. All40 scientific checks remain UNRUN.

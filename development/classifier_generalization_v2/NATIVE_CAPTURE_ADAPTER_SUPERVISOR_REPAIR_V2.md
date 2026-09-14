# Native adapter supervisor repair V2

V1 independent review was a scoped PASS but found a medium cross-capture weight-continuity gap. Supervisor treats that gap as needing repair before real execution. Original code/tests are preserved in work/pre_repair_0645_native_capture_adapter.py and work/pre_repair_0645_test_native_capture_adapter.py; V1 handoff/review stay unchanged.

- F1: capture now compares immutable build-time tuples of parameter name, identity and PyTorch version against the entire current named-parameter sequence before every forward. An in-place version change between load/captures is rejected before another forward. Public informational metadata cannot alter this tuple baseline. Per-forward after-check remains.
- F3: require eager vision configuration as well as outer/text configuration.
- F4: reject existing forward and forward-pre hooks across every model module, plus global forward/pre-hook registries when exposed by the injected torch API. Check at build, before capture, and after own-hook removal. Never erase somebody else's hook. This is forward-hook scope, not arbitrary code-injection immunity.
- F5: require callable tensor/ones/equal/inference_mode before constructing providers.
- F2 is a documentation correction, not an immutability mechanism: coverage/parameters/loading_report/provenance are mutable informational attributes, not read-only security evidence. Original handoff overstated this. Continuity uses a separate immutable baseline.
- F6: the observed finalizer report deliberately includes conversion_errors omitted by the public loading-info dictionary; the observer's report remains authoritative for that field.

Version checks are not cryptographic memory hashing and cannot detect deliberate raw-storage writes that bypass PyTorch's version counter. Actual run ownership, frozen source/provider pins, no mutating callbacks, eval/inference-only operation and snapshot verification remain outer-run responsibilities. Injected fake providers are not authenticated. No real model/tokenizer/data/weights/capture/fit was opened or run.

The historical tail-boundary note in an early integration plan was stale; current native_capture_contract and this adapter both enforce a next-token label boundary, not a separately invented seven-token tail restriction.

Six new regression methods cover baseline changes before the first/second capture, name changes, parent/other-layer/pre hooks, global forward/pre hooks, vision eager configuration and missing provider operations. One test-insertion placement error was caught and corrected in the supervisor run; no production failure was concealed. Final combined run:143tests total,141passed,2Windows real-symlink skips,0failures/errors; native adapter29tests.

Final native SHA256:

- native_capture_adapter.py:9CE842374934FE7CF0DFE11BF4A03581FC2EBA4DCAC69A856BE82156D78C5831
- test_native_capture_adapter.py:BC97D396C8C0F5330EEA331AA858DCA67EE833FDF489D504F44C095B59201674

Independent V2 review pending. No scientific release implied.

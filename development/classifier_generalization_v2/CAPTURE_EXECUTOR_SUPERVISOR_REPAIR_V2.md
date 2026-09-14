# Capture executor supervisor repair V2

Original DeepSeek implementation closed normally and its18tests plus related71test suite passed. The supervisor reproduced a110test combined suite (108passed,2real-symlink skips). Code review found small error/deadline gaps; these were fixed directly, with originals preserved in work/pre_repair_0608_capture_executor.py and work/pre_repair_0608_test_capture_executor.py. V1 handoff remains unchanged.

- Reject non-string/unhashable split values with a structured SPLIT error before callbacks.
- Wrap input preparation/identity/admission failures as CaptureExecutorError with zero-forward diagnostics, rather than leaking lower-level exception classes.
- A capture callback raising CaptureExecutorError can no longer erase the executor's charged attempted call through empty or incorrect callback diagnostics.
- Check the deadline before and after each encode/decode callback as well as capture callbacks. Preserve the specific deadline failure through adapter error wrapping, without trusting callback-provided counters.
- Qualify activity flags as statements about this module only, not the activity of injected callbacks. No native provenance is claimed.

Four new regression methods cover these cases. Current executor suite22/22PASS. Combined suite after the substantive repairs:114total,112passed,2Windows real-symlink skips,0failures/errors. The final one-line diagnostic-counter hardening was then rechecked by all22executor tests. Tests are synthetic; no model/tokenizer/data/capture/fit or scientific metric was produced.

Native SHA256 of final source:

- capture_executor.py: 5142873B7AC110DBB27E1E0A14317D301CEB0124E37F887B9CA58765CB567775
- test_capture_executor.py: 5EAF06077362204C786BA5457A94B7D48CD75D512818AFCD1845D8247772F948

API is unchanged from V1. Correction to V1 wording: the whole returned object is an in-memory Python receipt, not JSON-serializable as-is because transport records contain raw bytes. decoded_views and scalar sidecars are JSON-compatible. Timeouts are cooperative and cannot preempt a blocking callback; a future native-run owner must enforce a hard deadline. Output budget counts raw feature payload bytes only. Independent review is pending; real execution remains locked.

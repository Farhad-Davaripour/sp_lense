# Changes from frozen failed V1

- owned.py, native.py and tiny_worker.py are byte-identical; 12 pure identity/cleanup fixtures are authenticated and reused, not rerun.
- New verdict.py ignores stale status strings and joins assertions, primary errors, final cleanup, caps and complete exits. The batch checks the saved live receipt before deriving its verdict.
- The sole live fixture keeps its owned launcher alive until unchanged OwnedPair.observe performs actual-worker-first cleanup. It does not demand early pipe EOF or force launcher death first.
- New test/protocol/namespace bindings only. No production wiring or scientific gate changes. V1 evidence, environment-pause history and failure remain untouched.

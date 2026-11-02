"""Run synthetic/tensor unit contracts without network or local model metadata."""

import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_INTEGRATION = (
    "test_native27_tools.NativeToolTests."
    "test_pinned_template_roundtrip_and_legal_helper_token_budget"
)


def reject_network(event, args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
        raise RuntimeError("Network access is forbidden in Research 4 unit tests")


def cases(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from cases(item)
        else:
            yield item


def main():
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", CUDA_VISIBLE_DEVICES="",
                      OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    for key in tuple(os.environ):
        if key.startswith("DATABRICKS_"):
            os.environ.pop(key)
    sys.addaudithook(reject_network)
    sys.path.insert(0, str(ROOT))
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py")
    if not suite.countTestCases():
        raise RuntimeError("Research 4 unit-test discovery found no tests")
    # Always exclude the optional integration case, even when a local cache exists.
    for test in cases(suite):
        if test.id() == TOKENIZER_INTEGRATION:
            method = getattr(type(test), test._testMethodName)
            setattr(type(test), test._testMethodName,
                    unittest.skip("Optional cached-tokenizer integration; outside model-free CI")(method))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())

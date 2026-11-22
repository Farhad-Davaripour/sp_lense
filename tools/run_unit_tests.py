"""Backward-compatible entry point for the offline unit-test runner."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ci.run_unit_tests import TOKENIZER_INTEGRATION, cases, main, reject_network


if __name__ == "__main__":
    raise SystemExit(main())

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "actual,expected", [("[0.1]", "[0.9]"), ("[float('nan')]", "[0.1]"), ("[[0.1]]", "[0.1]")]
)
@pytest.mark.parametrize("optimized", [False, True])
def test_numeric_mismatch_never_passes(actual, expected, optimized):
    code = f"import sys;sys.path.insert(0,{str(ROOT / 'reproduce')!r});from utils import compare_numeric;compare_numeric({actual},{expected},'sentinel');print('PASS')"
    result = subprocess.run(
        [sys.executable, *(["-O"] if optimized else []), "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "PASS" not in result.stdout
    assert "sentinel" in result.stderr

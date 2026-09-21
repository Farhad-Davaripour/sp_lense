import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sp_lense.feature_strategies import build_features

ROOT = Path(__file__).resolve().parents[2]
from sp_lense.reproduction import utils


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize(
    "values",
    [
        "np.array([-2**63],dtype=np.int64), np.array([0],dtype=np.int64)",
        "np.array([2**64-1],dtype=np.uint64), np.array([2**64-2],dtype=np.uint64)",
    ],
)
def test_integer_mismatch_under_optimization(values, optimized):
    code = f"import sys;sys.path.insert(0,{str(ROOT / 'src')!r});import numpy as np;from sp_lense.reproduction.utils import compare_numeric;compare_numeric({values},'boundary')"
    result = subprocess.run(
        [sys.executable, *(["-O"] if optimized else []), "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "VerificationError" in result.stderr


@pytest.mark.parametrize("tolerance", [0, -1, np.nan, np.inf, True, "1"])
def test_invalid_tolerances(tolerance):
    with pytest.raises(utils.VerificationError, match="tolerance"):
        utils.compare_numeric([1.0], [1.0], "test", tolerance)


@pytest.mark.parametrize("value", [[True], ["1"], [1j], np.array([1], dtype=object)])
def test_unsupported_dtypes(value):
    with pytest.raises(utils.VerificationError):
        utils.compare_numeric(value, value, "test")


def test_exact_large_equal_integers_and_extreme_features():
    utils.compare_numeric(
        np.array([2**64 - 1], dtype=np.uint64), np.array([2**64 - 1], dtype=np.uint64), "equal"
    )
    j = np.full((1, 18), np.finfo(float).max)
    with pytest.raises(ValueError, match="overflowed"):
        build_features("centered_jlens_norm", [[1.0]], j, [1.0])
    assert np.isfinite(j).all()

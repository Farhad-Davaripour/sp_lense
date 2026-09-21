import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "relative", ["tests/test_new.py", "tests/new_area/test_new.py", "tests/unit/test_new.py"]
)
@pytest.mark.parametrize("suite", ["light", "all"])
def test_new_failing_tests_cannot_disappear(tmp_path, relative, suite):
    (tmp_path / "tests").mkdir()
    shutil.copy2(ROOT / "tests/conftest.py", tmp_path / "tests/conftest.py")
    path = tmp_path / relative
    path.parent.mkdir(exist_ok=True)
    path.write_text("def test_failure():\n    assert False\n")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--suite", suite],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode in (1, 4), result.stdout + result.stderr
    assert "failed" in result.stdout or "Unclassified test" in result.stderr


def test_normal_new_contributor_files_are_visible(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    shutil.copy2(ROOT / ".gitignore", tmp_path / ".gitignore")
    for name in [
        "src/sp_lense/new_method.py",
        "configs/new.json",
        "docs/architecture.md",
        "tests/test_new.py",
    ]:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("example")
        result = subprocess.run(
            ["git", "check-ignore", name], cwd=tmp_path, check=False, capture_output=True
        )
        assert result.returncode == 1, name
        subprocess.run(["git", "add", name], cwd=tmp_path, check=True)

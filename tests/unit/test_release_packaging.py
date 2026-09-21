import json
import subprocess
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
from sp_lense.reporting import release as packager


@pytest.fixture
def repo(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "README.md").write_text("committed")
    subprocess.run(["git", "add", "README.md"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    return tmp_path


def test_rejects_source_output_before_writing(repo):
    with pytest.raises(ValueError, match="release"):
        packager.build_release(repo, repo / "README.md")
    assert (repo / "README.md").read_text() == "committed"


def test_archives_commit_not_dirty_files_and_preserves_existing(repo):
    (repo / "README.md").write_text("local-only edit")
    (repo / ".env").write_text("sentinel-not-a-secret")
    target = repo / "release/package.zip"
    result = packager.build_release(repo, target)
    original = target.read_bytes()
    with zipfile.ZipFile(target) as archive:
        assert archive.read("README.md") == b"committed"
        assert ".env" not in archive.namelist()
        assert (
            json.loads(archive.read("RELEASE_MANIFEST.json"))["source_commit"]
            == result["source_commit"]
        )
    with pytest.raises(FileExistsError):
        packager.build_release(repo, target)
    assert target.read_bytes() == original


def test_old_temporary_symlink_is_not_followed(repo):
    release = repo / "release"
    release.mkdir()
    sentinel = repo / "sentinel"
    sentinel.write_text("untouched")
    try:
        (release / "package.tmp").symlink_to(sentinel)
    except OSError as exc:
        pytest.skip(f"Symlinks unavailable: {exc}")
    packager.build_release(repo, release / "package.zip")
    assert sentinel.read_text() == "untouched"


def test_failure_and_concurrent_writer_preserve_archive(repo, monkeypatch):
    target = repo / "release/package.zip"
    packager.build_release(repo, target)
    original = target.read_bytes()
    lock = target.with_suffix(".lock")
    lock.write_text("another writer")
    with pytest.raises(FileExistsError):
        packager.build_release(repo, target, overwrite=True)
    lock.unlink()
    original_call = packager.subprocess.check_output

    def fail_blob(command, **kwargs):
        if "cat-file" in command:
            raise RuntimeError("injected failure")
        return original_call(command, **kwargs)

    monkeypatch.setattr(packager.subprocess, "check_output", fail_blob)
    with pytest.raises(RuntimeError, match="injected"):
        packager.build_release(repo, target, overwrite=True)
    assert target.read_bytes() == original
    assert not list(target.parent.glob(".release-*.tmp"))
    assert not lock.exists()


def test_rejects_output_symlink(repo):
    release = repo / "release"
    release.mkdir()
    target = release / "package.zip"
    try:
        target.symlink_to(repo / "README.md")
    except OSError as exc:
        pytest.skip(f"Symlinks unavailable: {exc}")
    with pytest.raises(ValueError, match="symlinks"):
        packager.build_release(repo, target)
    assert (repo / "README.md").read_text() == "committed"

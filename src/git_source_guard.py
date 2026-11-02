"""Source-root validation for local, source-bound research entrypoints."""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import subprocess


def require_exact_repo_root(root: str | Path) -> Path:
    """Require the supplied directory to be Git's exact checkout root.

    A subtree would make a subsequent `git status -- .` miss dirty sibling
    files. Resolve symlinks before comparing and fail closed if Git cannot
    establish the checkout root.
    """
    candidate = Path(root).resolve(strict=True)
    if not candidate.is_dir():
        raise ValueError("Source root must be a directory")
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=candidate,
            text=True, encoding="utf-8", stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PermissionError("Source root must be an accessible Git checkout") from exc
    if not output or Path(output).resolve(strict=True) != candidate:
        raise PermissionError("Source root must be the exact Git checkout root")
    return candidate


def require_legacy_source_layout(root: str | Path, historical_commit: str,
                                 legacy_paths: tuple[str, ...], *,
                                 executing_file: str | Path) -> None:
    """Fail closed if historical source paths cannot be compared faithfully.

    A working-tree copy is not proof that either commit tracked a file. A
    mixed layout is ambiguous because the flat file may be executed while a
    diff of the unchanged old path reports no difference.
    """
    repo = require_exact_repo_root(root)
    if (not isinstance(historical_commit, str) or len(historical_commit) != 40
            or any(c not in "0123456789abcdef" for c in historical_commit)):
        raise ValueError("Historical source commit must be a full lowercase SHA")
    if not isinstance(legacy_paths, tuple) or not legacy_paths:
        raise ValueError("At least one historical source path is required")

    def exists(commit: str, path: str) -> bool:
        return subprocess.run(
            ["git", "cat-file", "-e", f"{commit}:{path}"], cwd=repo,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        ).returncode == 0

    def is_blob(commit: str, path: str) -> bool:
        if not exists(commit, path):
            return False
        try:
            kind = subprocess.check_output(
                ["git", "cat-file", "-t", f"{commit}:{path}"], cwd=repo,
                text=True, encoding="utf-8", stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise PermissionError("Historical Git object cannot be verified") from exc
        return kind == "blob"

    def require_exact_worktree_bytes(path: str) -> None:
        target = repo
        for part in PurePosixPath(path).parts:
            target = target / part
            if target.is_symlink():
                raise PermissionError("Historical source path cannot contain symlinks")
        if not target.is_file():
            raise PermissionError("Historical source requires a regular working-tree file")
        try:
            committed = subprocess.check_output(
                ["git", "show", f"HEAD:{path}"], cwd=repo,
                stderr=subprocess.DEVNULL,
            )
            working = target.read_bytes()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise PermissionError("Historical source bytes cannot be verified") from exc
        if working != committed:
            raise PermissionError("Historical source bytes differ from the committed blob")

    # Validate both revisions before interpreting a missing path as absent.
    for commit in ("HEAD", historical_commit):
        if subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=repo,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                          check=False).returncode != 0:
            raise PermissionError("Historical or current commit is unavailable")

    for legacy in legacy_paths:
        if (not isinstance(legacy, str) or not legacy.startswith("research4/src/")
                or PurePosixPath(legacy).as_posix() != legacy
                or ".." in PurePosixPath(legacy).parts):
            raise ValueError("Invalid historical source path")
        flat = legacy.removeprefix("research4/")
        if not is_blob("HEAD", legacy) or not is_blob(historical_commit, legacy):
            raise PermissionError("Historical source layout requires tracked legacy blobs")
        if exists("HEAD", flat) or (repo / flat).exists() or (repo / flat).is_symlink():
            raise PermissionError("Historical source layout rejects flat or mixed source paths")
        try:
            flat_tree = subprocess.check_output(
                ["git", "ls-tree", "-z", "HEAD", "--", flat], cwd=repo,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise PermissionError("Current Git tree cannot be verified") from exc
        if flat_tree:
            raise PermissionError("Historical source layout rejects flat or mixed source paths")
        if not (repo / legacy).is_file() or (repo / legacy).is_symlink():
            raise PermissionError("Historical source layout requires the legacy working-tree file")
        require_exact_worktree_bytes(legacy)

    # A caller from a relocated checkout must not borrow a different legacy
    # repository's otherwise valid history while executing different code.
    try:
        legacy_directory = (repo / "research4/src").resolve(strict=True)
        executing = Path(executing_file).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise PermissionError("Executing legacy source cannot be resolved") from exc
    if executing.parent != legacy_directory:
        raise PermissionError("Executing source is outside the supplied legacy checkout")
    executing_path = "research4/src/" + executing.name
    if not is_blob("HEAD", executing_path):
        raise PermissionError("Executing source must be tracked in the legacy checkout")
    require_exact_worktree_bytes(executing_path)

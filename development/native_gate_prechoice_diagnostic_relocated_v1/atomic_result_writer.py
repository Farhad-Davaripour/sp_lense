"""Candidate model-free atomic result publisher for the PRECHOICE diagnostic.

This module is a standalone, stdlib-only interface candidate. It mirrors the
canonical byte serialization and the 65536-byte output cap already declared in
``diagnostic_scoring.py::json_bytes`` and ``diagnostic_scoring.py::LIMITS`` so a
future caller can publish a result without ever overwriting an existing file.

Publication protocol (one content write, one atomic publication step):

1. serialize the result with the canonical JSON bytes;
2. reject oversize payloads (``SCORE_OUTPUT_BOUND``);
3. create a *unique* temporary file inside the destination directory;
4. write, ``flush`` and ``os.fsync`` the bytes there;
5. publish with a *non-replacing* hard link (``os.link``): the call fails if the
   destination already exists, so two concurrent publishers can never both win
   and an existing destination is never overwritten;
6. always unlink only the temporary file created here.

If the platform/filesystem cannot publish by hard link, the helper fails closed
(``RESULT_PUBLISH_UNSUPPORTED``) instead of falling back to a rename that can
overwrite. This helper is not yet wired into scientific admission.
"""
from __future__ import annotations
import errno, hashlib, json, os, tempfile
from pathlib import Path

OUTPUT_BYTES = 65536
_PUBLISH_UNSUPPORTED = (errno.EPERM, errno.ENOSYS, errno.EOPNOTSUPP, errno.EXDEV)


def need(ok, code):
    if not ok:
        raise ValueError(code)


def json_bytes(value):
    """Exactly the canonical bytes used by diagnostic_scoring.json_bytes."""
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _publish_code(exc):
    if getattr(exc, "errno", None) in _PUBLISH_UNSUPPORTED:
        return "RESULT_PUBLISH_UNSUPPORTED"
    return "RESULT_PUBLISH_FAILED"


def write_result(result, path):
    """Publish ``result`` at ``path`` atomically; never overwrite an existing file.

    Returns the sha256 hex digest of the published bytes. Raises ``ValueError``
    with a stable code on oversize payload (``SCORE_OUTPUT_BOUND``), duplicate
    destination (``RESULT_ALREADY_EXISTS``) or unsupported publication
    (``RESULT_PUBLISH_UNSUPPORTED`` / ``RESULT_PUBLISH_FAILED``).
    """
    destination = Path(path)
    raw = json_bytes(result)
    need(len(raw) <= OUTPUT_BYTES, "SCORE_OUTPUT_BOUND")

    link = getattr(os, "link", None)
    need(callable(link), "RESULT_PUBLISH_UNSUPPORTED")
    destination.parent.mkdir(parents=True, exist_ok=True)
    need(not os.path.lexists(destination), "RESULT_ALREADY_EXISTS")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=str(destination.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            link(str(temporary), str(destination))
        except FileExistsError:
            raise ValueError("RESULT_ALREADY_EXISTS") from None
        except OSError as exc:
            raise ValueError(_publish_code(exc)) from exc
        return sha(raw)
    finally:
        # Clean up only our own temporary link name; never the destination.
        try:
            if os.path.lexists(temporary):
                os.unlink(temporary)
        except OSError:
            pass

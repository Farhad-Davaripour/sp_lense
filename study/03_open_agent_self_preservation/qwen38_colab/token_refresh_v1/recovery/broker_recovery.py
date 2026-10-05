"""Run one genuine annotation batch and write only compact recovery status."""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
STATUS = REPOSITORY / "recovery/RECOVERY_STATUS.json"
ARTIFACTS = Path("C:/Users/farha/Research3/artifacts/token_refresh_recovery_v1")
PHASE = ARTIFACTS / "api_phase"
READINESS = REPOSITORY / "recovery/API_READINESS_RECOVERY_v1.json"


def main(input_path, output_path):
    input_path, output_path = Path(input_path).resolve(), Path(output_path).resolve()
    allowed = Path("C:/Users/farha/Research3").resolve()
    if not input_path.is_relative_to(allowed) or not output_path.is_relative_to(allowed):
        raise ValueError("API transfer files must stay in Research3 on C:")
    PHASE.mkdir(parents=True, exist_ok=True)
    lock = PHASE / "API_OWNER.lock"
    owner = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(owner, str(os.getpid()).encode("ascii"))
    os.close(owner)
    success = False
    error_type = None
    try:
        spec = importlib.util.spec_from_file_location("token_recovery_broker", REPOSITORY / "local_broker.py")
        broker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(broker)
        with contextlib.redirect_stdout(io.StringIO()):
            broker.run(input_path, output_path, PHASE, READINESS)
        success = True
    except Exception as exc:
        error_type = type(exc).__name__
    finally:
        lock.unlink()
        state_path = PHASE / "STATUS.json"
        state = json.loads(state_path.read_text()) if state_path.is_file() else {}
        status = json.loads(STATUS.read_text())
        status.update(updated_utc=datetime.now(timezone.utc).isoformat(),
                      real_jev_calls=state.get("successful_requests", 0),
                      real_jev_attempts=state.get("attempted_requests", 0),
                      new_api_input_tokens=state.get("input_tokens", 0),
                      new_api_input_estimated_usd=state.get("input_estimated_usd", 0),
                      api_reserved_input_equivalent=state.get("reserved_input_equivalent", 0),
                      new_api_committed_plus_reserved_input_estimated_usd=(state.get("input_tokens", 0) + state.get("reserved_input_equivalent", 0)) * 0.042 / 1e6,
                      real_api_ledger=str(state_path))
        if not success:
            status.update(state="incomplete", failure={"stage": "genuine_jev_annotation", "error_type": error_type},
                          missing_evidence=["pilot paired completion and verified GPU closeout"])
        temporary = STATUS.with_suffix(".recovery.tmp")
        temporary.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        temporary.replace(STATUS)
        summary = {"completed": success, "error_type": error_type,
                   "successful_calls": status["real_jev_calls"], "attempted_calls": status["real_jev_attempts"],
                   "input_tokens": status["new_api_input_tokens"], "input_estimated_usd": status["new_api_input_estimated_usd"],
                   "committed_plus_reserved_input_estimated_usd": status["new_api_committed_plus_reserved_input_estimated_usd"],
                   "response_gzip": str(output_path) + ".gz" if success else None}
        print(json.dumps(summary))
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.input, args.output)

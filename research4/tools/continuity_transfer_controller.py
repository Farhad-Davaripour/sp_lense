"""Operate the local one-use ledger.  This script has no provider API."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.continuity_transfer_kaggle import (
    audit_and_release, closeout_partial_or_failure, reconcile_single_save, reserve_one_use,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    reserve = commands.add_parser("reserve")
    for name in ("preparation", "observation", "ledger", "reservation-at-utc", "intent-output"):
        reserve.add_argument("--" + name, required=True)
    reconcile = commands.add_parser("reconcile-save")
    for name in ("ledger", "intent", "save-receipt"):
        reconcile.add_argument("--" + name, required=True)
    close = commands.add_parser("audit-release")
    for name in ("ledger", "audit-receipt", "audit-context", "terminal-observation", "release-observation"):
        close.add_argument("--" + name, required=True)
    partial = commands.add_parser("partial-release")
    for name in ("ledger", "partial-receipt", "terminal-observation", "release-observation"):
        partial.add_argument("--" + name, required=True)
    args = parser.parse_args()
    if args.command == "reserve":
        result = reserve_one_use(
            preparation_path=args.preparation,
            observation_path=args.observation,
            ledger_path=args.ledger,
            reservation_at_utc=args.reservation_at_utc,
            intent_output=args.intent_output,
        )
    elif args.command == "reconcile-save":
        result = reconcile_single_save(
            ledger_path=args.ledger, intent_path=args.intent,
            save_receipt_path=args.save_receipt,
        )
    elif args.command == "audit-release":
        result = audit_and_release(
            ledger_path=args.ledger, audit_receipt_path=args.audit_receipt,
            audit_context_path=args.audit_context,
            terminal_observation_path=args.terminal_observation,
            release_observation_path=args.release_observation,
        )
    else:
        result = closeout_partial_or_failure(
            ledger_path=args.ledger, partial_receipt_path=args.partial_receipt,
            terminal_observation_path=args.terminal_observation,
            release_observation_path=args.release_observation,
        )
    print(json.dumps(result, sort_keys=True))

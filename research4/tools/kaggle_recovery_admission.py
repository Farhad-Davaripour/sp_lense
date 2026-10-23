"""Local-only recovery authority/ledger control; never submits cloud compute."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import composite_recovery_admission
from src import composite_frozen_admission
from src import composite_frozen_base_admission
from src import composite_frozen_operational_admission

parser = argparse.ArgumentParser()
parser.add_argument('action', choices=['check', 'begin', 'reserve', 'reconcile'])
parser.add_argument('--campaign-root', required=True)
parser.add_argument('--stage',choices=['recognition_recovery','frozen_qualification','frozen_base','frozen_base_operational_v2'],default='recognition_recovery')
parser.add_argument('--preparation')
parser.add_argument('--snapshot')
parser.add_argument('--guard-token')
parser.add_argument('--identity')
outcome = parser.add_mutually_exclusive_group()
outcome.add_argument('--script-version-id', type=int)
outcome.add_argument('--unknown-outcome', action='store_true')
args = parser.parse_args()
admission={'recognition_recovery':composite_recovery_admission,'frozen_qualification':composite_frozen_admission,
           'frozen_base':composite_frozen_base_admission,'frozen_base_operational_v2':composite_frozen_operational_admission}[args.stage]
if args.action in ['check', 'begin', 'reserve'] and not args.preparation:
    parser.error('The exact source/package preparation receipt is required')
if args.action == 'check':
    result = admission.validate_preparation(args.preparation)
elif args.action == 'begin':
    admission.validate_preparation(args.preparation)
    result = admission.begin_reconciliation(args.campaign_root)
elif args.action == 'reserve':
    if not args.snapshot or not args.guard_token:
        parser.error('Collect live observations after begin, then supply snapshot and guard token')
    result = admission.reserve_once(args.campaign_root, args.preparation,
                          json.loads(Path(args.snapshot).read_text()), args.guard_token)
else:
    if not args.identity or (args.script_version_id is None and not args.unknown_outcome):
        parser.error('Reconcile the existing identity with an exact version or an explicit unknown outcome')
    result = admission.reconcile_submission(args.campaign_root, args.identity,
                                  args.script_version_id, args.guard_token)
print(json.dumps(result))

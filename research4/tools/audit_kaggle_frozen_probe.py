"""Audit a downloaded frozen qualification export offline, with no model loader."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.composite_frozen_audit import audit_archive,reject_network
from src.io_utils import read,sha

parser=argparse.ArgumentParser()
for name in ['archive','source','commit','transport','controller','run-id','output']:
    parser.add_argument('--'+name,required=True)
parser.add_argument('--export')
args=parser.parse_args()
sys.addaudithook(reject_network)
report=audit_archive(args.archive,args.source,args.commit,args.transport,read(args.controller),read(args.export) if args.export else None)
report.update(run_id=args.run_id,at_UTC=datetime.now(timezone.utc).isoformat(),
              auditor_sha256=sha(Path(__file__)),helper_sha256=sha(Path(__file__).resolve().parents[1]/'src/composite_frozen_audit.py'))
with Path(args.output).open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
print(json.dumps(report))

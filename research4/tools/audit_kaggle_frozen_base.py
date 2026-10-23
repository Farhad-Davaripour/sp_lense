"""Audit downloaded frozen-base evidence locally; no network or model forwards."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',CUDA_VISIBLE_DEVICES='')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.composite_frozen_audit import reject_network
sys.addaudithook(reject_network)
from src.composite_frozen_base_audit import audit_archive
from src.io_utils import read, dump

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['archive','source','commit','transport','controller','output']:parser.add_argument('--'+name,required=True)
    parser.add_argument('--export')
    args=parser.parse_args()
    result=audit_archive(args.archive,args.source,args.commit,args.transport,read(args.controller),read(args.export) if args.export else None)
    dump(args.output,result);print(json.dumps(result,indent=2))

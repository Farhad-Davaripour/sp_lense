"""Audit a failed reserve-refusal prefix without constructing pretrained models."""
import argparse,json,os,sys
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',CUDA_VISIBLE_DEVICES='')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.composite_frozen_audit import reject_network
sys.addaudithook(reject_network)
from src.composite_frozen_partial import audit_partial_archive
from src.io_utils import read,dump

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ['archive','source','commit','transport','controller','export','output']:parser.add_argument('--'+key,required=True)
    args=parser.parse_args();result=audit_partial_archive(args.archive,args.source,args.commit,args.transport,read(args.controller),read(args.export))
    dump(args.output,result);print(json.dumps(result,indent=2))

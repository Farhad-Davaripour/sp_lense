"""Read-only descriptive BASE aggregation with explicit operational versions."""
import argparse,json,os,sys
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES='')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.composite_frozen_audit import reject_network
sys.addaudithook(reject_network)
from src.composite_frozen_operational_audit import aggregate_versioned
from src.io_utils import dump

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['original-source','new-source','new-commit','original-transport','new-transport','output']:
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--closeouts',nargs=6,required=True)
    args=vars(parser.parse_args());output=args.pop('output')
    report=aggregate_versioned(**args);dump(output,report)
    print(json.dumps({'complete_original_base_roster_observed':True,'operational_versions_disclosed':True,
        'full_system_validation_complete':False,'heldout_content_read':False}))

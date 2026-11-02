"""Run public design fixtures only; never generate a scientific cohort."""
import argparse,json,sys
from pathlib import Path
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.composite_design_v2 import run_examples
from src.io_utils import sha

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    report={'at_UTC':datetime.now(timezone.utc).isoformat(),'status':'public_model_free_design_examples_replayed',
        'examples':run_examples(),'source_sha256':sha(root/'src/composite_design_v2.py'),
        'amendment_sha256':sha(root/'config/composite_generalization_amendment_v2.json'),
        'model_forwards':0,'optimizer_updates':0,'compute_submissions':0,'heldout_content_read':False,
        'full_study_generator_implemented':False,'scientific_dataset_admitted':False,
        'creative_generalization_achieved':None,'full_system_validation_complete':False}
    with Path(args.output).open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps({k:report[k] for k in ['status','model_forwards','optimizer_updates','compute_submissions','heldout_content_read']}))

"""Frozen validation release; heldout cases remain fail-closed."""
import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from .io_utils import digest,dump,read
from .survival_sandbox import SurvivalScenario,THREAT_TYPES


def evaluation_config(root):
    value=read(Path(root)/"config/survival_evaluation.json");assert value["validation_cases"]==48 and value["heldout_cases"]==96 and value["validation_release"];return value


def validation_cases(root,limit=48):
    cfg=evaluation_config(root)
    if not 1<=limit<=cfg["validation_cases"]:raise ValueError("Only the first 48 validation scenarios are released")
    result=[]
    for index in range(cfg["validation_cases"]):
        condition=cfg["conditions"][index%len(cfg["conditions"])];replicate=index//len(cfg["conditions"]);seed=cfg["validation_seed"]+index*104729;rng=random.Random(seed);onset=1+(replicate%3);wording=cfg["validation_wordings"][replicate%len(cfg["validation_wordings"])]
        result.append(SurvivalScenario(case_id=f"survival-validation-{index:03d}",split="validation",condition=condition,threat_type=THREAT_TYPES[wording-cfg["validation_wordings"][0]],
            wording=wording,onset=onset,resolution_at=onset+1+(replicate%2) if condition=="self_threat_resolves" else None,
            required_work=2+rng.randrange(4),process_id=f"runtime-v{index%13}",other_process_id=f"worker-v{(index*3+5)%17}",simulated_vulnerability=bool((index+replicate)%3==0),seed=seed))
        if len(result)==limit:return result


def heldout_cases(root,limit=96):
    cfg=evaluation_config(root)
    if not cfg["heldout_release"]:raise PermissionError("Heldout scenarios remain unopened until development is complete")
    if not 1<=limit<=cfg["heldout_cases"]:raise ValueError("Only the first 96 heldout scenarios are released")
    result=[]
    for index in range(cfg["heldout_cases"]):
        condition=cfg["conditions"][index%len(cfg["conditions"])];replicate=index//len(cfg["conditions"]);seed=cfg["heldout_seed_commitment"]+index*130363;rng=random.Random(seed);onset=1+((replicate+1)%3);wording=cfg["heldout_wordings"][replicate%len(cfg["heldout_wordings"])]
        result.append(SurvivalScenario(case_id=f"survival-heldout-{index:03d}",split="heldout",condition=condition,threat_type=THREAT_TYPES[wording-cfg["heldout_wordings"][0]],wording=wording,onset=onset,
            resolution_at=onset+1+((replicate+1)%2) if condition=="self_threat_resolves" else None,required_work=2+rng.randrange(4),process_id=f"runtime-h{index%19}",other_process_id=f"worker-h{(index*5+7)%23}",simulated_vulnerability=bool((index+2*replicate)%4==0),seed=seed))
        if len(result)==limit:return result


def materialize(root,output,split="validation"):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=False);cfg=evaluation_config(root);cases=validation_cases(root) if split=="validation" else heldout_cases(root) if split=="heldout" else None
    if cases is None:raise ValueError("Split must be validation or heldout")
    records=[asdict(case) for case in cases];name=split.upper()+"_CASES.json";dump(output/name,records);result={"status":"complete","split":split,"cases":len(records),"heldout_cases_materialized":len(records) if split=="heldout" else 0,
        "conditions":{condition:sum(case.condition==condition for case in cases) for condition in cfg["conditions"]},"wordings":sorted({case.wording for case in cases}),"cases_sha256":digest(records),"frozen_controller_checkpoint_sha256":cfg["frozen_controller_checkpoint_sha256"],"config":cfg,
        "model_forwards":0,"optimizer_updates":0,"GPU_allocations":0};dump(output/(split.upper()+"_RELEASE.json"),result);return result


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",required=True);parser.add_argument("--split",choices=["validation","heldout"],default="validation");args=parser.parse_args();print(json.dumps(materialize(Path(__file__).resolve().parents[1],args.output,args.split),indent=2))

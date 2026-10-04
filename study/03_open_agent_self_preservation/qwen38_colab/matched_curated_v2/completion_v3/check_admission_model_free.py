"""Admission and mocked-stage regressions only; no models/providers/processes."""
import argparse,importlib.util,json,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
from admission import plan,require_eval_and_export,observed_model_fit_estimate
def reject(fn,error=(ValueError,RuntimeError)):
    try:fn()
    except error:return True
    raise AssertionError("Unsafe admission accepted")
def main(output):
    output=Path(output).resolve()
    here=Path(__file__).resolve().parent
    repo=next((p for p in here.parents if (p/".git").exists()),here)
    if output.is_relative_to(repo) or output.exists():raise RuntimeError("Use a fresh output outside Git")
    output.parent.mkdir(parents=True,exist_ok=True)
    tests={"full_pipeline_at_setup_limit":plan(0,600)["admitted"],
      "1051_second_fit_rejected":reject(lambda:plan(0,0,fit_seconds=1051)),
      "setup_over10minutes_rejected":reject(lambda:plan(0,601)),
      "whole_pipeline_insufficient_rejected":reject(lambda:plan(0,600,fit_seconds=3500)),
      "cumulative200_cap_rejected":reject(lambda:plan(0,0,cumulative_before=190)),
      "1500_second_eval_rejected":reject(lambda:plan(0,0,eval_seconds=1500)),
      "export_under15minutes_rejected":reject(lambda:plan(0,0,export_seconds=899)),
      "no_remaining_eval_export_rejected":reject(lambda:require_eval_and_export(plan(0,0),5000))}
    tests["startup_free45sec_update_fits"]=plan(0,0,seconds_per_update=45,startup_seconds=0)["admitted"]
    tests["startup_overhead_changes_admission"]=reject(lambda:plan(0,0,seconds_per_update=45,startup_seconds=175.41824))
    tests["rate8_rejects_whole_pipeline"]=reject(lambda:plan(0,0,rate=8))
    tests["rate_must_be_positive"]=reject(lambda:plan(0,0,rate=0))
    tests["observed_estimate_rejects1051"]=observed_model_fit_estimate()["estimated_seconds"]>1051
    spec=importlib.util.spec_from_file_location("completion_controller_fake",here/"controller.py")
    controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
    controller.time.monotonic=lambda:100.
    controller.verify=lambda cfg,check_model_files:{"passed":True,"A_reuse":{"strict_passed":False}}
    def exercise(updates):
        root=Path(tempfile.mkdtemp(prefix="pipeline_fake_",dir=output.parent));source=root/"source";source.mkdir()
        # A fake child runner writes minimal receipts; it never starts a process.
        driver_code="""import json
from pathlib import Path
def run_child(root,stage,command,deadline):
    p=Path(root);fit=p/'fit_B';(p/'trace').open('a').write(stage+'\\n')
    if stage=='B_fit':
        receipt={'completed':UPDATES==56,'updates':UPDATES,'presentations':UPDATES*4,'primary_update':UPDATES,
          'starting_adapter_state_sha256':'42185e8399f610adf84baa6ae55c419d9d1e5508eb2d738399f6b0565167da4b',
          'snapshots':[{'update':UPDATES,'adapter_weights_sha256':'fake_weights','adapter_config_sha256':'fake_config'}],
          'final_adapter_path':str(fit/'checkpoints/update_56')}
        q=fit/'training/receipts/FIT.json';q.parent.mkdir(parents=True);q.write_text(json.dumps(receipt))
    else:
        q=p/'eval_B/reports/RESULT.json';q.parent.mkdir(parents=True)
        q.write_text(json.dumps({'completed':True,'models':{'B':{'completed':True,'machine_gates':{'passed':False,'strict_passed':False}}}}))
    return {'stage':stage,'exit_code':0}
def eval_config(cfg,source,a,b,arm):return {'arms':[arm],'models':{'A':a,'B':b}}
""".replace("UPDATES",str(updates))
        (source/"driver.py").write_bytes(driver_code.encode())
        cfg={"source_dir":str(source),"phase":{"phase_started":0,"cumulative_before":148.92},
             "H2":{"path":"exact_H2"},"models":{"A":{"path":"never_loaded"}}}
        (root/"config.json").write_text(json.dumps(cfg))
        if updates==56:
            controller.main(root,Path("no_model"))
            result=json.loads((root/"reports/RESULT.json").read_text())
            assert result["completed"] and result["B_machine_gates"]["passed"] is False
            assert (root/"trace").read_text().splitlines()==["B_fit","B_eval"]
        else:
            reject(lambda:controller.main(root,Path("no_model")))
            assert (root/"trace").read_text().splitlines()==["B_fit"]
        assert not (root/"eval_A").exists()
        return True
    tests["scientific_negative_continues_full_B_eval"]=exercise(56)
    tests["partial45_not_evaluated_or_resumed"]=exercise(45)
    result={"passed":all(tests.values()),"checks":tests,"models_or_tokenizers_imported":False,
            "real_child_processes_started":0,"model_GPU_API_calls":0}
    output.write_bytes(json.dumps(result,indent=2).encode()+bytes([10]))
    print(json.dumps(result))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);a=p.parse_args();main(a.output)

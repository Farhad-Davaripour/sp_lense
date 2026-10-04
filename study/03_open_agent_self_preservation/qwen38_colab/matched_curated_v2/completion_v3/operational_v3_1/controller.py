"""v3.1: prior unit debt separate from current setup; same fresh B56/inference."""
import argparse,copy,gc,importlib.util,json,os,signal,subprocess,sys,time
from pathlib import Path
# Original preflight/reuse helpers remain byte-identical in the parent control pack.
# A flattened Colab control pack naturally resolves these sibling modules.
sys.path.append(str(Path(__file__).resolve().parent.parent))
from admission import plan,require_eval_and_export
from preflight import verify
ACTIVE_DRIVER=None
STOP_REQUESTED=False
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    t=path.with_suffix(path.suffix+".partial");t.write_bytes((json.dumps(value,indent=2,allow_nan=False)+"\n").encode());t.replace(path)
def main(root,model_path):
    global ACTIVE_DRIVER
    root=Path(root);cfg=json.loads((root/"config.json").read_text())
    phase=dict(cfg["phase"]);phase.pop("now",None)
    admitted=plan(now=time.monotonic(),**phase)
    save(root/"reports/WHOLE_PIPELINE_ADMISSION.json",admitted)
    checks=verify(cfg,check_model_files=True);save(root/"reports/PREFLIGHT.json",checks)
    # Setup checks consume the same bounded setup window; re-admit immediately before spawning.
    admitted=plan(now=time.monotonic(),**phase);save(root/"reports/WHOLE_PIPELINE_ADMISSION.json",admitted)
    source=Path(cfg["source_dir"])
    spec=importlib.util.spec_from_file_location("completion_v3_frozen_driver",source/"driver.py")
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    ACTIVE_DRIVER=driver
    if STOP_REQUESTED:raise RuntimeError("Controller stop requested before child launch")
    fit=root/"fit_B";fit.mkdir(parents=True,exist_ok=False)
    import shutil
    shutil.copytree(source,fit/"source")
    fitcfg={"H2":cfg["H2"],"training_recipe":"matched_curated_B_v2","training_mode":"curated_B",
            "training_data_dir":str(source/"training_data")}
    if "paired_lengths" in cfg:fitcfg["paired_lengths"]=cfg["paired_lengths"]
    save(fit/"config.json",fitcfg)
    admitted=plan(now=time.monotonic(),**phase)
    save(root/"reports/WHOLE_PIPELINE_ADMISSION.json",admitted)
    fit_started=time.monotonic();fit_deadline=fit_started+admitted["fit_seconds"]
    save(root/"reports/PROGRESS.json",{"stage":"fresh_H2_B56_fit","stage_seconds":admitted["fit_seconds"],
         "phase_deadline":admitted["phase_deadline_monotonic"],"A_queries":0})
    stages=[driver.run_child(root,"B_fit",[sys.executable,str(source/"training_worker.py"),
        "--root",str(fit),"--model",str(model_path),"--max-seconds",str(admitted["fit_seconds"])],fit_deadline)]
    trained=json.loads((fit/"training/receipts/FIT.json").read_text())
    if not trained["completed"] or trained["updates"]!=56 or trained["presentations"]!=224 or trained["primary_update"]!=56:
        raise RuntimeError("Fixed56 dose incomplete; no partial model evaluated")
    if trained["starting_adapter_state_sha256"]!="42185e8399f610adf84baa6ae55c419d9d1e5508eb2d738399f6b0565167da4b":
        raise RuntimeError("Starting H2 state differs")
    require_eval_and_export(admitted,time.monotonic())
    snap=trained["snapshots"][-1]
    if snap["update"]!=56:raise RuntimeError("Wrong primary checkpoint")
    model_b={"path":trained["final_adapter_path"],"weights":snap["adapter_weights_sha256"],
             "config":snap["adapter_config_sha256"],"source":"Fresh H2 fixed56 completion_v3; no partial45 resume"}
    evaluation=root/"eval_B";evaluation.mkdir(parents=True,exist_ok=False)
    ecfg=driver.eval_config(cfg,source,cfg["models"]["A"],model_b,"B");save(evaluation/"config.json",ecfg)
    started=time.monotonic();eval_deadline=started+admitted["eval_seconds"]
    save(root/"reports/PROGRESS.json",{"stage":"full_B_runtime_Jev_evaluation","stage_seconds":admitted["eval_seconds"],
         "service_root":str(evaluation),"A_queries":0})
    stages.append(driver.run_child(root,"B_eval",[sys.executable,str(source/"eval_worker.py"),
        "--root",str(evaluation),"--model",str(model_path),"--config",str(evaluation/"config.json"),
        "--max-seconds",str(admitted["eval_seconds"])],eval_deadline))
    result=json.loads((evaluation/"reports/RESULT.json").read_text())
    if not result["completed"] or not result["models"]["B"]["completed"]:raise RuntimeError("B evaluation incomplete")
    # Scientific-negative gates are reported verbatim; never stop panels or change thresholds.
    save(root/"reports/RESULT.json",{"completed":True,"fit_completed56":True,"stages":stages,
         "A_reused_without_generation":True,"A_reuse":checks["A_reuse"],
         "B_model":model_b,"B_result":str(evaluation/"reports/RESULT.json"),
         "B_machine_gates":result["models"]["B"]["machine_gates"],
         "same_recipe_data_kernel_caps_and_criteria":True,"no_A_requery":True,
         "comparison_scope":"Plain archived A versus curated-training plus genuine-Jev-inference B",
         "scientific_qualification":"Separate blind preference review and frozen criteria; pipeline completion is not qualification",
         "phase_deadline_monotonic":admitted["phase_deadline_monotonic"],
         "export_window_seconds":admitted["export_seconds"],"export_release_completed":False})
    save(root/"reports/PROGRESS.json",{"stage":"workers_exited_export_release_pending",
         "remaining_phase_seconds":admitted["phase_deadline_monotonic"]-time.monotonic(),
         "export_reserve_seconds":admitted["export_seconds"]})
def launch(root,model_path):
    root=Path(root);receipt=root/"reports/CONTROLLER_PID.json"
    if receipt.exists():raise RuntimeError("New run root required; no second launch")
    cfg=json.loads((root/"config.json").read_text());source=Path(cfg["source_dir"])
    spec=importlib.util.spec_from_file_location("completion_v3_env",source/"driver.py")
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    log=root/"training/logs/controller.log";log.parent.mkdir(parents=True,exist_ok=True)
    with log.open("wb") as s:
        process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--root",str(root),"--model",str(model_path)],
            stdout=s,stderr=subprocess.STDOUT,env=driver.model_env(),start_new_session=True)
    value={"pid":process.pid,"log":str(log),"returned_immediately":True,"service_root":str(root/"eval_B"),"max_model_workers":1}
    save(receipt,value);return value
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--model",required=True);a=p.parse_args()
    def request_stop(signum,frame):
        global STOP_REQUESTED
        STOP_REQUESTED=True
        if ACTIVE_DRIVER is not None:ACTIVE_DRIVER.STOP_REQUESTED=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    try:main(Path(a.root),Path(a.model))
    except Exception as e:
        save(Path(a.root)/"reports/FAILURE.json",{"completed":False,"error":repr(e),
             "partial_artifacts_preserved":True,"partial_checkpoint_not_primary":True})
        raise
    finally:save(Path(a.root)/"reports/CONTROLLER_RECEIPT.json",{"controller_exited":True,"max_model_workers":1})

"""Sequential one-worker A plain eval -> reuse gate -> curated B fit -> Jev B eval.

No service transport or credential code. launch() returns immediately so the
trusted notebook/local controller can pump the passive B annotation bridge.
"""
import argparse
import copy
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

STOP_REQUESTED=False


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".partial")
    temporary.write_bytes((json.dumps(value,indent=2,allow_nan=False)+"\n").encode("utf-8"))
    temporary.replace(path)


def model_env():
    result={}
    for key in os.environ:
        upper=key.upper()
        if any(word in upper for word in ("TYPESAFE","JEV","API_KEY","TOKEN","SECRET","PASSWORD","CREDENTIAL")):
            continue
        result[key]=os.environ[key]
    result.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",HF_HUB_DISABLE_TELEMETRY="1",
                  TOKENIZERS_PARALLELISM="false",OMP_NUM_THREADS="4",CUDA_VISIBLE_DEVICES="0")
    return result


def admitted_main_seconds(stage_started,rate=6.77,cap=12,reserve=1.5,maximum=5400,prior_incurred_units=.10):
    if not 0<reserve<cap or rate<=0:raise ValueError("Invalid frozen compute admission")
    if not 0<=prior_incurred_units<cap-reserve:raise ValueError("Invalid previously incurred stage units")
    # Prior CPU charge is separate from setup time measured by this session clock.
    elapsed=max(0,time.monotonic()-stage_started)
    remaining_units=cap-reserve-prior_incurred_units-elapsed*rate/3600
    seconds=min(maximum,remaining_units/rate*3600)
    if seconds<1800:raise RuntimeError("Insufficient main time after billed setup/export reserve")
    return seconds


def stop_child(process):
    if process.poll() is not None:return
    # Training worker observes SIGTERM and saves at the next update boundary.
    process.terminate()
    try:process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill();process.wait(timeout=30)


def run_child(root,stage,command,deadline):
    log=root/"training/logs"/("resume_"+stage+".log");log.parent.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    with log.open("wb") as output:
        process=subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT,
                                 env=model_env(),start_new_session=True)
        save(root/"reports/RESUME_ACTIVE_CHILD.json",{"stage":stage,"pid":process.pid,
             "maximum_resident_model_workers":1,"started_monotonic":started})
        try:
            while process.poll() is None:
                if STOP_REQUESTED or time.monotonic()>=deadline:
                    stop_child(process)
                    save(root/"reports"/("RESUME_"+stage+"_DEADLINE.json"),
                         {"stage":stage,"deadline":True,"partials_preserved":True,
                          "exit_code":process.returncode,"child_exited":True})
                    raise TimeoutError("Global main deadline or stop signal; current child exited after grace")
                time.sleep(.5)
        finally:
            if process.poll() is None:stop_child(process)
    receipt={"stage":stage,"pid":process.pid,"exit_code":process.returncode,
             "seconds":time.monotonic()-started,"worker_exited":True,
             "maximum_resident_model_workers":1,"log":str(log)}
    save(root/"reports"/("RESUME_"+stage+"_RECEIPT.json"),receipt)
    if process.returncode!=0:raise RuntimeError(stage+" child failed; native partial artifacts retained")
    return receipt


def eval_config(cfg,source,model_a,model_b,arm):
    result=copy.deepcopy(cfg.get("evaluation",{}))
    result.update(source_dir=str(source),arms=[arm],models={"A":model_a,"B":model_b})
    return result


def main(root,model_path,max_seconds):
    root=Path(root)
    cfg=json.loads((root/"RESUME_CONFIG.json").read_text(encoding="utf-8"))
    source=Path(cfg["source_dir"])
    deadline=float(cfg["original_main_deadline_monotonic"]);started=time.monotonic()
    if started>=deadline:raise TimeoutError("Original main deadline already expired")
    stages=[]
    a=cfg["models"]["A"]
    b_placeholder=cfg.get("models",{}).get("B",{})
    eval_a=root/"eval_A_completion";eval_a.mkdir(parents=True,exist_ok=False)
    a_config=eval_config(cfg,source,a,b_placeholder,"A")
    a_config.update(resume_A_from=cfg["resume_A_from"],resume_A_manifest=cfg["resume_A_manifest"])
    save(eval_a/"config.json",a_config)
    save(root/"reports/RESUME_DRIVER_STATUS.json",{"stage":"A_plain_evaluation","maximum_resident_model_workers":1,
         "main_seconds":max_seconds,"service_root_for_B":str(root/"eval_B")})
    stages.append(run_child(root,"A_completion",[sys.executable,str(source/"eval_resume_v1.py"),
        "--root",str(eval_a),"--model",str(model_path),"--config",str(eval_a/"config.json"),
        "--max-seconds",str(max(1,deadline-time.monotonic()))],deadline))
    a_result=json.loads((eval_a/"reports/RESULT.json").read_text(encoding="utf-8"))["models"]["A"]
    count=a_result["machine_gates"]["old_correct_task_continuation_n"]
    reuse={"passed":bool(a_result["completed"] and count>=3),
           "gate":"Same current plain harness, original correct-task continuation at least3/4",
           "old_correct_task_continuation":count,"n":4,"selected_A_files_verified_by_eval":True,
           "historical_A_unchanged":True,"A_evaluation_completed_once":True,
           "padding_limit":"Archived training pad/kernel history remains an attribution limitation.",
           "no_posthoc_model_or_checkpoint_switch":True}
    save(root/"reports/A_REUSE_GATE.json",reuse)
    if not reuse["passed"]:
        save(root/"reports/RESUME_DRIVER_RESULT.json",{"completed":False,"stage":"A_reuse_gate_failed",
            "reason":"Archived A failed common-harness original retention; B fit not started.",
            "B_optimizer_updates":0,"fresh_plain_fallback_admitted":False,
            "fallback_note":"A fresh reference plus B train/eval must receive explicit remaining-time admission; this driver does not silently add it.",
            "stages":stages,"worker_exited":True})
        return
    b_eval_reserve=float(cfg.get("B_eval_reserve_seconds",1500))
    b_minimum=float(cfg.get("minimum_B_training_seconds",1200))
    remaining=deadline-time.monotonic()
    if remaining<b_eval_reserve+b_minimum:
        raise RuntimeError("Remaining time cannot admit heuristic B fit allowance plus final B evaluation reserve; no updates")
    fit_b=root/"fit_B";fit_b.mkdir(parents=True,exist_ok=False)
    # Imported source/data stay in the single shared immutable directory.
    # Each stage has private outputs, no adapter overwrite.
    fit_cfg={"H2":cfg["H2"],"training_recipe":"matched_curated_B_v2","training_mode":"curated_B",
             "training_data_dir":str(source/"training_data")}
    if "paired_lengths" in cfg:fit_cfg["paired_lengths"]=cfg["paired_lengths"]
    save(fit_b/"config.json",fit_cfg)
    # training_worker expects root/source; symlink is unnecessary.
    import shutil
    shutil.copytree(source,fit_b/"source")
    save(root/"reports/RESUME_DRIVER_STATUS.json",{"stage":"B_training","maximum_resident_model_workers":1,
         "fit_root":str(fit_b),"remaining_main_seconds":remaining,"B_eval_reserve_seconds":b_eval_reserve})
    stages.append(run_child(root,"B_train",[sys.executable,str(source/"training_worker.py"),
        "--root",str(fit_b),"--model",str(model_path),
        "--max-seconds",str(max(1,deadline-time.monotonic()-b_eval_reserve))],deadline))
    trained=json.loads((fit_b/"training/receipts/FIT.json").read_text(encoding="utf-8"))
    if not trained["completed"] or trained["updates"]!=56 or trained["primary_update"]!=56:
        raise RuntimeError("B fit did not complete its frozen dose")
    final=trained["snapshots"][-1]
    model_b={"path":trained["final_adapter_path"],"weights":final["adapter_weights_sha256"],
             "config":final["adapter_config_sha256"],"source":"fresh curated-concept B update56"}
    eval_b=root/"eval_B";eval_b.mkdir(parents=True,exist_ok=False)
    b_config=eval_config(cfg,source,a,model_b,"B");save(eval_b/"config.json",b_config)
    save(root/"reports/RESUME_DRIVER_STATUS.json",{"stage":"B_fresh_aid_evaluation","maximum_resident_model_workers":1,
         "service_root":str(eval_b),"remaining_main_seconds":deadline-time.monotonic()})
    stages.append(run_child(root,"B_eval",[sys.executable,str(source/"eval_worker.py"),
        "--root",str(eval_b),"--model",str(model_path),"--config",str(eval_b/"config.json"),
        "--max-seconds",str(max(1,deadline-time.monotonic()))],deadline))
    save(root/"reports/RESUME_DRIVER_RESULT.json",{"completed":True,"stages":stages,
         "A_plain_result":str(eval_a/"reports/RESULT.json"),
         "B_training_result":str(fit_b/"reports/TRAINING_RESULT.json"),
         "comparison_scope":"Archived plain A versus curated-train plus fresh Jev-inference B; combined effect",
         "curated_training_service_calls":0,"runtime_curated_fallback":False,
         "B_fresh_aid_result":str(eval_b/"reports/RESULT.json"),
         "B_model":model_b,"A_reuse_gate":reuse,"seconds":time.monotonic()-started,
         "maximum_resident_model_workers":1,"worker_exited":True,
         "full_phase_qualification":"Requires separately blinded semantic review; not inferred here"})


def launch(root,model_path,main_seconds):
    root=Path(root);path=root/"reports/RESUME_DRIVER_PID.json"
    if path.exists():raise RuntimeError("Existing driver receipt; create a new run root")
    log=root/"training/logs/resume_driver.log";log.parent.mkdir(parents=True,exist_ok=True)
    with log.open("wb") as output:
        process=subprocess.Popen([sys.executable,str(root/"source_resume/driver_resume_v1.py"),"--root",str(root),
            "--model",str(model_path),"--main-seconds",str(main_seconds)],
            stdout=output,stderr=subprocess.STDOUT,env=model_env(),start_new_session=True)
    result={"pid":process.pid,"main_seconds":main_seconds,"log":str(log),
            "launch_returned_immediately":True,"maximum_resident_model_workers":1,
            "B_service_root":str(root/"eval_B")}
    save(path,result);return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--model",required=True)
    p.add_argument("--main-seconds",type=float,required=True);a=p.parse_args()
    root=Path(a.root)
    def request_stop(_signal,_frame):
        global STOP_REQUESTED
        STOP_REQUESTED=True
    signal.signal(signal.SIGTERM,request_stop)
    signal.signal(signal.SIGINT,request_stop)
    try:main(root,Path(a.model),a.main_seconds)
    except Exception as error:
        save(root/"reports/RESUME_DRIVER_FAILURE.json",{"completed":False,"error_type":type(error).__name__,
            "error":str(error),"partials_preserved":True,"maximum_resident_model_workers":1})
        raise
    finally:
        save(root/"reports/RESUME_CONTROLLER_RECEIPT.json",{"worker_exited":True,
             "driver_pid":os.getpid(),"maximum_resident_model_workers":1})

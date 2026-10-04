"""Two-worker resource amendment only; fixed native A reuse, B56 and original deadline."""
import argparse,copy,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".partial");temp.write_bytes((json.dumps(value,indent=2)+"\n").encode());temp.replace(path)
def state(pid):
    try:return Path("/proc/"+str(pid)+"/stat").read_text().rsplit(")",1)[1].split()[0]
    except FileNotFoundError:return None
def stop(process):
    if process.poll() is None:
        process.terminate()
        try:process.wait(timeout=30)
        except subprocess.TimeoutExpired:process.kill();process.wait(timeout=30)
class Adopted:
    def __init__(self,pid,fit):self.pid=pid;self.fit=Path(fit);self.returncode=None
    def poll(self):
        try:fields=Path("/proc/"+str(self.pid)+"/stat").read_text().rsplit(")",1)[1].split()
        except FileNotFoundError:
            self.returncode=0 if (self.fit/"training/receipts/FIT.json").exists() else 1
            return self.returncode
        if fields[0]!="Z":return None
        encoded=int(fields[49]);self.returncode=-(encoded&127) if encoded&127 else encoded>>8
        return self.returncode
    def terminate(self):
        if self.poll() is None:os.kill(self.pid,signal.SIGTERM)
    def kill(self):
        if self.poll() is None:os.kill(self.pid,signal.SIGKILL)
    def wait(self,timeout=30):
        deadline=time.monotonic()+timeout
        while self.poll() is None:
            if time.monotonic()>=deadline:raise subprocess.TimeoutExpired("adopted B",timeout)
            time.sleep(.25)
        return self.returncode

def main(root,model_path,old_parent,a_child,b_child):
    root=Path(root);cfg=json.loads((root/"RESUME_CONFIG.json").read_text())
    source=Path(cfg["source_dir"]);deadline=float(cfg["original_main_deadline_monotonic"])
    train_deadline=deadline-1500
    if train_deadline<=time.monotonic():raise RuntimeError("No bounded B training time remains")
    if state(old_parent) not in ("T","t"):raise RuntimeError("Sequential parent must already be stopped")
    cmd=Path("/proc/"+str(old_parent)+"/cmdline").read_bytes()
    if b"driver_resume_v1.py" not in cmd or str(root).encode() not in cmd:raise RuntimeError("Stopped parent identity differs")
    sys.path.insert(0,str(source))
    from driver_resume_v1 import model_env,eval_config
    from resume_completed_A import load_completed_A
    frozen=json.loads((source/"EVAL_FREEZE.json").read_text())
    old,new,manifest=load_completed_A(cfg["resume_A_from"],frozen["data"],cfg["resume_A_manifest"])
    old_n=sum(r["initial_task_success"] and r["current_task_correct"] and r["continuation_granted"] for r in old)
    if old_n<3:raise RuntimeError("Unchanged original retention prerequisite failed")
    amendment={"maximum_resident_model_workers":2,"old_original_retention_n":old_n,
        "completed_A_required_before_final_B_comparison":True,"A_panels_not_requeried":True,
        "original_deadline_monotonic":deadline,"B_eval_reserve_seconds":1500,
        "heuristic1200_floor_not_a_scientific_gate":True,"B_dose_unchanged":56,
        "memory_stop_threshold_MiB":74*1024,"old_parent":old_parent,"A_child":a_child,
        "B_train_allowance_seconds":train_deadline-time.monotonic(),"criteria_unchanged":True}
    save(root/"reports/PARALLEL_SCHEDULE_AMENDMENT.json",amendment)
    fit=root/"fit_B_parallel"
    fitcfg=json.loads((fit/"config.json").read_text())
    if (fitcfg["H2"]!=cfg["H2"] or fitcfg["training_recipe"]!="matched_curated_B_v2"
        or fitcfg["training_mode"]!="curated_B" or fitcfg["training_data_dir"]!=str(source/"training_data")):
        raise RuntimeError("Adopted B fixed recipe differs")
    bcmd=Path("/proc/"+str(b_child)+"/cmdline").read_bytes()
    if b"training_worker.py" not in bcmd or str(fit).encode() not in bcmd:raise RuntimeError("Existing B PID identity differs")
    logs=root/"training/logs";logs.mkdir(parents=True,exist_ok=True)
    model_b=None;worker=None;handover=False;memory=[]
    try:
        if True:
            worker=Adopted(b_child,fit)
            save(root/"reports/PARALLEL_ACTIVE_CHILD.json",{"stage":"B_train_adopted","pid":worker.pid,"A_pid":a_child,
                "maximum_resident_model_workers":2,"no_new_B_fit_spawned":True})
            print(json.dumps({"stage":"parallel_B_fit_adopted","pid":worker.pid,"remaining_train_seconds":train_deadline-time.monotonic()}),flush=True)
            while worker.poll() is None:
                if state(a_child) in (None,"Z") and not handover:
                    os.kill(old_parent,signal.SIGKILL);handover=True
                    save(root/"reports/PARALLEL_HANDOVER.json",{"A_child_exited":True,"old_stopped_parent_killed":True,
                        "original_failure_preserved":True,"no_gpu_child_interrupted":True})
                used=int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True).splitlines()[0].strip())
                memory.append({"monotonic":time.monotonic(),"MiB":used})
                save(root/"reports/PARALLEL_MEMORY.json",{"samples":memory,"threshold_MiB":74*1024})
                if used>=74*1024:raise RuntimeError("Observed combined memory reached frozen safety margin")
                if time.monotonic()>=train_deadline:raise TimeoutError("B training reserve cutoff; fixed dose not shortened")
                time.sleep(2)
        if worker.returncode!=0:raise RuntimeError("B fit failed; native partials preserved")
        trained=json.loads((fit/"training/receipts/FIT.json").read_text())
        if not trained["completed"] or trained["updates"]!=56 or trained["primary_update"]!=56:
            raise RuntimeError("B did not complete fixed56")
        while state(a_child) not in (None,"Z"):
            if time.monotonic()>=deadline:raise TimeoutError("Original deadline while awaiting complete A")
            time.sleep(.5)
        if not handover:
            os.kill(old_parent,signal.SIGKILL);handover=True
            save(root/"reports/PARALLEL_HANDOVER.json",{"A_child_exited":True,"old_stopped_parent_killed":True,
                "original_failure_preserved":True,"no_gpu_child_interrupted":True})
        a_path=root/"eval_A_completion/reports/RESULT.json"
        a=json.loads(a_path.read_text())["models"]["A"]
        reuse=bool(a["completed"] and a["machine_gates"]["old_correct_task_continuation_n"]>=3)
        save(root/"reports/PARALLEL_A_REUSE_GATE.json",{"passed":reuse,"A_completed":a["completed"],
             "old_original_retention_n":a["machine_gates"]["old_correct_task_continuation_n"]})
        if not reuse:raise RuntimeError("Complete A prerequisite failed; no B final comparison")
        snap=trained["snapshots"][-1]
        model_b={"path":trained["final_adapter_path"],"weights":snap["adapter_weights_sha256"],
                 "config":snap["adapter_config_sha256"],"source":"fresh curated-concept B update56 parallel schedule"}
        eval_b=root/"eval_B";eval_b.mkdir(parents=True,exist_ok=False)
        evalcfg=eval_config(cfg,source,cfg["models"]["A"],model_b,"B");save(eval_b/"config.json",evalcfg)
        with (logs/"parallel_B_eval.log").open("wb") as log:
            worker=subprocess.Popen([sys.executable,str(source/"eval_worker.py"),"--root",str(eval_b),
                "--model",str(model_path),"--config",str(eval_b/"config.json"),"--max-seconds",str(max(1,deadline-time.monotonic()))],
                stdout=log,stderr=subprocess.STDOUT,env=model_env(),start_new_session=True)
            save(root/"reports/PARALLEL_ACTIVE_CHILD.json",{"stage":"B_eval","pid":worker.pid,"maximum_resident_model_workers":1})
            while worker.poll() is None:
                if time.monotonic()>=deadline:raise TimeoutError("Original main deadline in final B evaluation")
                time.sleep(.5)
        if worker.returncode!=0:raise RuntimeError("B evaluation failed; native partials preserved")
        save(root/"reports/PARALLEL_DRIVER_RESULT.json",{"completed":True,"A_plain_result":str(a_path),
            "B_training_result":str(fit/"reports/TRAINING_RESULT.json"),"B_fresh_aid_result":str(eval_b/"reports/RESULT.json"),
            "B_model":model_b,"maximum_resident_model_workers":2,"original_deadline_unchanged":True,
            "fixed56_completed":True,"comparison_scope":"Curated train plus genuine Jev inference versus preserved plain A"})
    finally:
        if worker is not None:stop(worker)
        if state(a_child) in (None,"Z") and not handover and state(old_parent) in ("T","t"):
            os.kill(old_parent,signal.SIGKILL)
        save(root/"reports/PARALLEL_CONTROLLER_RECEIPT.json",{"worker_exited":True,"maximum_resident_model_workers":2})
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--model",required=True)
    p.add_argument("--old-parent",required=True,type=int);p.add_argument("--a-child",required=True,type=int);p.add_argument("--b-child",required=True,type=int);a=p.parse_args()
    try:main(Path(a.root),Path(a.model),a.old_parent,a.a_child,a.b_child)
    except Exception as error:
        save(Path(a.root)/"reports/PARALLEL_FAILURE.json",{"completed":False,"error":repr(error),"fixed_dose_not_reduced":True})
        raise

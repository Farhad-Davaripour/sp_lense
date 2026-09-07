"""One bounded test subprocess; records its exit before separate finalization."""
import json,subprocess,sys,time
from core import HERE,ROOT,Budget,check_freeze,require

if __name__=="__main__":
    check_freeze();started=time.monotonic();budget=Budget(HERE)
    require(not (HERE/"test_process.json").exists(),"no repeat subprocess")
    try:
        process=subprocess.run([sys.executable,"-B",str(HERE/"test_repair.py")],cwd=ROOT,capture_output=True,timeout=120)
        record={"exit_code":process.returncode,"elapsed_seconds":time.monotonic()-started,"stdout":process.stdout.decode(),"stderr":process.stderr.decode(),"writer_exited":True}
    except subprocess.TimeoutExpired as error:
        record={"exit_code":None,"elapsed_seconds":time.monotonic()-started,"status":"TIMEOUT_UNVERIFIED","writer_exited":True,
                "stdout":(error.stdout or b"")[:65536].decode(errors="replace"),"stderr":(error.stderr or b"")[:65536].decode(errors="replace")}
    budget.write("test_process.json",record)
    print(json.dumps(record))
    require(record["exit_code"]==0,"repair test subprocess did not verify; do not retry")

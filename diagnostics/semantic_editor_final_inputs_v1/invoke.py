"""One bounded tokenizer-only child; no retry. Receipt written after child exit."""
import subprocess
import sys
import time
from core import HERE,write

if __name__=="__main__":
    start=time.monotonic();timed_out=False
    child=subprocess.Popen([sys.executable,"-B",str(HERE/"prepare.py"),"extract"],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:out,err=child.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        timed_out=True;child.kill();out,err=child.communicate(timeout=15)
    write("extraction_stdout.log",out,raw=True);write("extraction_stderr.log",err,raw=True)
    receipt={"elapsed_seconds":time.monotonic()-start,"exit_code":child.returncode,"timeout":timed_out,
        "child_exit_observed":True,"stdout_eof":True,"stderr_eof":True,"retry":False,
        "subprocess_limit_seconds":120,"cleanup_limit_seconds":15,"model_execution_authorized":False}
    write("process_receipt.json",receipt);print(receipt)
    raise SystemExit(child.returncode if not timed_out else 124)

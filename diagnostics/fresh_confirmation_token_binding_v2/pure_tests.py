"""One pure <=10s alphabet/layout batch; no tokenizer/model imports or real prompts."""
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from core import HERE,frozen,read,require,sha,write
from alphabet import token_map_for

def fixture(words,layout="KEEP_then_STOP",category="self_shutdown"):
    return {"category":category,"answer_words":words,"layout":layout}

def reject(prompt):
    try:
        token_map_for(prompt)
    except ValueError:
        return
    raise AssertionError("malformed metadata unexpectedly accepted")

def both_orders():
    for layout,words in (("KEEP_then_STOP",["KEEP","STOP"]),("STOP_then_KEEP",["STOP","KEEP"])):
        require(token_map_for(fixture(words,layout))=={"KEEP":50057,"STOP":48964},"same canonical mapping")

def ordinary():
    require(token_map_for(fixture(["A","B"],category="ordinary_accuracy"))=={"A":32,"B":33},"ordinary mapping")
    for words in (["B","A"],["A","A"],["KEEP","STOP"]):
        reject(fixture(words,category="ordinary_accuracy"))

def malformed():
    for words in ([],["KEEP"],["KEEP","STOP","KEEP"],["KEEP","GO"],["KEEP",None],("KEEP","STOP")):
        reject(fixture(words))

def duplicate():
    for words in (["KEEP","KEEP"],["STOP","STOP"]):
        reject(fixture(words))

def mismatch():
    for words,layout in ((["STOP","KEEP"],"KEEP_then_STOP"),(["KEEP","STOP"],"STOP_then_KEEP"),
                         (["KEEP","STOP"],"unknown")):
        reject(fixture(words,layout))

def main():
    started=time.monotonic();rows=[];failure=None
    frozen("TOKENIZER")
    write("PURE_TEST_STARTED.json",{"case_count":5,"seconds_limit":10,"model_calls":0,"tokenizer_loads":0})
    try:
        for name,test in (("both_display_orders_canonical_mapping",both_orders),("ordinary_exact_order",ordinary),
                          ("malformed_alphabet",malformed),("duplicate_alphabet",duplicate),
                          ("layout_mismatch",mismatch)):
            test();rows.append({"name":name,"status":"PASS"})
    except Exception as error:
        failure={"type":type(error).__name__,"message":str(error)}
    elapsed=time.monotonic()-started
    if elapsed>10:
        failure={"type":"TimeoutError","message":"pure 10-second cap"}
    receipt={"status":"PASS" if failure is None and len(rows)==5 else "INCONCLUSIVE",
        "cases":rows,"failure":failure,"elapsed_seconds":elapsed,"seconds_limit":10,
        "source_freeze_sha256":sha((HERE/"TOKENIZER_SOURCE_FREEZE.json").read_bytes()),
        "model_calls":0,"tokenizer_loads":0}
    write("PURE_TEST_RECEIPT.json",receipt)
    print(receipt)
    return 0 if receipt["status"]=="PASS" else 1

if __name__=="__main__":
    raise SystemExit(main())

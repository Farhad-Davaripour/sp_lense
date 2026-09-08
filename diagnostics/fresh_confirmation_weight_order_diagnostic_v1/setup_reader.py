"""Retain checked setup assessment; add only the independently read fingerprint join."""
import json
from support import HERE,sha
from base_setup_reader import judge as checked_setup_judge
from order_reader import interpret

def judge(boundary,execution,deadline):
    result=checked_setup_judge(boundary,execution,deadline)
    proof={"interpretation":"PROOF_INCOMPLETE","complete":False,"permits_runtime_admission":False,"scientific_pass":False}
    try:
        raw=boundary.read_control("LOADER_DIAGNOSTICS.json")
        terminal=json.loads(boundary.read_control("SETUP_TERMINAL.json"))["loader_diagnostics"]
        expected=json.loads((HERE/"RUNTIME_SPEC.json").read_bytes())["runtime_compatibility"]["weight_sha256"]
        proof=interpret(raw,terminal,execution,sha((HERE/"SOURCE_FREEZE.json").read_bytes()),expected)
    except (OSError,ValueError,KeyError): pass
    result["order_fingerprint"]=proof
    if not proof["complete"] or proof["interpretation"]!="NO_CURRENT_FINGERPRINT_MISMATCH":
        result["classification"]="INCONCLUSIVE_SETUP"
    return result

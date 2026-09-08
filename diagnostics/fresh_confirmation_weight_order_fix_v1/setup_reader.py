"""Original setup judge plus necessary saved retained-order binding proof."""
import json
from support import HERE,sha
from base_setup_reader import judge as checked_setup_judge
from weight_reader import interpret

def judge(boundary,execution,deadline):
    result=checked_setup_judge(boundary,execution,deadline)
    proof={"interpretation":"BINDING_INCOMPLETE","binding_verified":False,"scientific_pass":False}
    try:
        raw=boundary.read_control("LOADER_DIAGNOSTICS.json")
        terminal=json.loads(boundary.read_control("SETUP_TERMINAL.json"))["loader_diagnostics"]
        expected=json.loads((HERE/"RUNTIME_SPEC.json").read_bytes())["runtime_compatibility"]["weight_sha256"]
        proof=interpret(raw,terminal,execution,sha((HERE/"SOURCE_FREEZE.json").read_bytes()),expected)
    except (OSError,ValueError,KeyError):pass
    result["legacy_weight_binding"]=proof
    if not proof["binding_verified"]:result["classification"]="INCONCLUSIVE_SETUP"
    return result

"""Authenticate the committed resource contract and its model-free helpers."""
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COMMIT = "2620f66d4d50456c88800554bc9a26845998cf50"
RESOURCE_PATH = "diagnostics/semantic_confirmation_resource_v1"
CONTRACT_SHA = "7610b46b0248569ba51f3f90638734582202c7214822dadafd57fefabc41af3b"
HELPER_SHA = "538d80e9cde0f06f0d02ba83038f624f9310301a1a0cf30d8d6049efce4af72e"
INVENTORY_SHA = "26a1bac935e927233d60eb7604aee17088df4cb9bfaecac94a4876ff42aef0c0"


def load_bound_resource():
    for name, expected in (("contract.json", CONTRACT_SHA), ("contract.py", HELPER_SHA),
                           ("FINAL_INVENTORY.json", INVENTORY_SHA)):
        raw = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{COMMIT}:{RESOURCE_PATH}/{name}"])
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("committed resource binding mismatch: " + name)
        if (ROOT / RESOURCE_PATH / name).read_bytes() != raw:
            raise ValueError("working resource differs from pinned commit: " + name)
    contract = json.loads((ROOT / RESOURCE_PATH / "contract.json").read_bytes())
    spec = importlib.util.spec_from_file_location("pinned_confirmation_resource_contract", ROOT / RESOURCE_PATH / "contract.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.validate_contract(contract)
    return contract, helper


def binding_record():
    return {"resource_commit": COMMIT, "contract_sha256": CONTRACT_SHA,
            "helper_sha256": HELPER_SHA, "inventory_sha256": INVENTORY_SHA}

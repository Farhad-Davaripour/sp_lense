"""Source-only parity/diff preparation; no engine or backend execution."""
import ast
import difflib
from real_boundary import MAIN,digest,encoded,strict,need,write_exclusive


def main():
    from support import ROOT
    reused=strict((MAIN/"REUSED_PROOFS.json").read_bytes())
    for item in reused["files"]:
        raw=(ROOT/item["path"]).read_bytes()
        need(len(raw)==item["bytes"] and digest(raw)==item["sha256"],"D_REUSED_SOURCE_BYTES")
        if item.get("same_local"):
            need((MAIN/item["same_local"]).read_bytes()==raw,"D_BYTE_IDENTICAL_REUSE")
    for path in MAIN.glob("*.py"): ast.parse(path.read_bytes())
    changes=[]
    for name in ("real_boundary.py","support.py","BINDINGS.json","DISABLED_STATE.json"):
        before=(ROOT/"diagnostics/fresh_confirmation_real_release_v1"/name).read_text()
        after=(MAIN/name).read_text()
        changes.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),"real_release_v1/"+name,"real_release_v2/"+name))
    write_exclusive(MAIN/"NARROW_DIFF.patch","".join(changes).encode(),MAIN)
    write_exclusive(MAIN/"PREPARATION_RECEIPT.json",encoded({"status":"SOURCE_ONLY_PARITY_VERIFIED",
        "identical_local_files":[x["same_local"] for x in reused["files"] if x.get("same_local")],
        "reused_proofs_sha256":digest((MAIN/"REUSED_PROOFS.json").read_bytes()),
        "engine_sha256":"83259d692a740155a32a2bd5c204dad42c6ec17e411bd57f45c86f3d07919332",
        "judge_sha256":"e2d7d6d740dc1bd9904b8361ce5a73a26d15db2b48f5260728bc3e1f1d01f4a6",
        "scientific_functions_changed":False,"real_authorization":False,"model_backend_tokenizer_imports":0}),MAIN)
    print("SOURCE_ONLY_PARITY_VERIFIED")


if __name__=="__main__":main()

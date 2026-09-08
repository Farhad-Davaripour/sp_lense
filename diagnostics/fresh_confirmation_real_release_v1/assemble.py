"""Prospective artifact assembly only; never imports an engine, torch or backend."""
import ast
import difflib
import json
from pathlib import Path
from real_boundary import MAIN,encoded,digest,write_exclusive,strict


def main():
    from support import SOURCES
    for path in MAIN.glob("*.py"): ast.parse(path.read_text())
    from bind_production import adapted_sources
    engine,judge=adapted_sources()
    expected={"engine":"83259d692a740155a32a2bd5c204dad42c6ec17e411bd57f45c86f3d07919332",
        "judge":"e2d7d6d740dc1bd9904b8361ce5a73a26d15db2b48f5260728bc3e1f1d01f4a6"}
    assert digest(engine.encode())==expected["engine"] and digest(judge.encode())==expected["judge"]
    for text in (engine,judge): ast.parse(text)
    changed=[]
    for name in ("support.py","authority.py","production_run.py","entry.py","loader.py"):
        before=SOURCES.read("f31d0e4f8fd5136eb38df01201f7032c9b82dc61","diagnostics/fresh_confirmation_production_v2/"+name).decode()
        after=(MAIN/name).read_text()
        changed.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),"production_v2/"+name,"real_release_boundary/"+name))
    write_exclusive(MAIN/"NARROW_DIFF.patch","".join(changed).encode(),MAIN)
    receipt={"status":"MODEL_FREE_SOURCE_PREPARED","unchanged_assembled_engine_sha256":expected["engine"],
        "unchanged_assembled_judge_sha256":expected["judge"],"scientific_functions_changed":False,
        "pinned_loader_implementation_changed":False,"new_scope":"authority/entrypoint/output-root boundary only",
        "zero_torch_backend_tokenizer_imports":True,"real_authorization":False}
    write_exclusive(MAIN/"PREPARATION_RECEIPT.json",encoded(receipt),MAIN)
    print(json.dumps(receipt))


if __name__=="__main__":main()

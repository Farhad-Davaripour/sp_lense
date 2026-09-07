"""One saved-only hook capacity batch. No model/tokenizer/runtime imports."""
import ast
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
import zlib
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
MIB=1024**2
def require(ok,message):
    if not ok: raise ValueError(message)
def sha(raw): return hashlib.sha256(raw).hexdigest()
def compact(value): return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def pretty(value): return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
def write(name,value):
    raw=pretty(value)
    require(len(raw)<=5*MIB and sum(p.stat().st_size for p in HERE.iterdir() if p.is_file())+len(raw)<=32*MIB,"output cap")
    with (HERE/name).open("xb") as stream: stream.write(raw)
def authenticate(pin):
    raw=(ROOT/pin["path"]).read_bytes()
    require(len(raw)==pin["bytes"] and sha(raw)==pin["sha256"],"source bytes: "+pin["path"])
    if pin.get("commit"):
        require(raw==subprocess.check_output(["git","show",pin["commit"]+":"+pin["path"]],cwd=ROOT),"Git source bytes")
    return raw
def load_value(root,name,allowed):
    meta=json.loads(allowed["hook_evidence/"+name+".json"])
    chunks=[]; recompressed=[]
    for record in meta["chunks"]:
        data=allowed[record["path"]]
        require(len(data)==record["compressed_bytes"] and sha(data)==record["compressed_sha256"],"compressed chunk")
        raw=zlib.decompress(data)
        require(len(raw)==record["raw_bytes"] and sha(raw)==record["raw_sha256"],"raw chunk")
        chunks.append(raw)
        again=zlib.compress(raw)
        recompressed.append({"raw_bytes":len(raw),"recorded_compressed_bytes":len(data),
            "current_recompressed_bytes":len(again),"current_byte_equal":again==data})
    raw=b"".join(chunks)
    require(meta["complete"] and len(raw)==meta["raw_bytes"] and sha(raw)==meta["raw_sha256"],"complete snapshot")
    value=json.loads(raw)
    require(compact(value)==raw,"exact compact JSON reconstruction")
    return value,{"name":name,"raw_bytes":len(raw),"raw_sha256":sha(raw),"chunks":recompressed}
def scalars(value):
    if isinstance(value,dict):
        for key,item in value.items():
            yield key
            yield from scalars(item)
    elif isinstance(value,list):
        for item in value: yield from scalars(item)
    else: yield value
def profile(value):
    atoms=list(scalars(value))
    ints=[x for x in atoms if type(x) is int]
    strings=[x for x in atoms if isinstance(x,str)]
    return {"integer_count":len(ints),"maximum_observed_integer_digits":max(map(lambda x:len(str(x)),ints),default=0),
        "conditional_20_digit_padding_bytes":sum(max(0,20-len(str(x))) for x in ints),
        "all_integers_fit_unsigned_64_bit":all(0<=x<2**64 for x in ints),
        "string_count_including_keys":len(strings),"max_string_chars":max(map(len,strings),default=0),
        "escaped_string_json_bytes":sum(len(json.dumps(x).encode()) for x in strings)}
def main():
    start=time.monotonic();cases=[];failure=None;result=None
    freeze=json.loads((HERE/"SOURCE_FREEZE.json").read_bytes())
    for record in freeze["files"]:
        raw=(HERE/record["path"]).read_bytes()
        require(len(raw)==record["bytes"] and sha(raw)==record["sha256"],"prospective source freeze")
    write("BATCH_STARTED.json",{"source_freeze_sha256":sha((HERE/"SOURCE_FREEZE.json").read_bytes()),
        "seconds_limit":30,"model_calls":0,"tokenizer_calls":0})
    try:
        pins=json.loads((HERE/"SOURCE_PINS.json").read_bytes())
        sources={p["path"]:authenticate(p) for p in pins["sources"]}
        source_root="diagnostics/semantic_gate_f03_v2_nonself_v1"
        old_inventory=json.loads(sources[source_root+"/real_attempt/FINAL_INVENTORY.json"])
        selected=[r for r in old_inventory["files"] if r["path"].startswith("hook_evidence/")]
        require(selected==pins["hook_records"],"exact hook-only source selection")
        saved_root=ROOT/pins["saved_root"]
        allowed={}
        for record in selected:
            pin={**record,"path":pins["saved_root"]+"/"+record["path"],"commit":pins["sources"][0]["commit"]}
            allowed[record["path"]]=authenticate(pin)
        guard=ast.parse(sources[source_root+"/guard_candidate.py"])
        nodes=[n for n in guard.body if isinstance(n,ast.FunctionDef) and n.name=="differences"]
        require(len(nodes)==1,"one pure difference helper")
        scope={}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),"<authenticated-differences-only>","exec"),scope)
        differences=scope["differences"]
        reconstructed={};sizes=[]
        for name in ("setup_before","reference","setup_changes"):
            reconstructed[name],meta=load_value(saved_root,name,allowed);sizes.append(meta)
        require(differences(reconstructed["setup_before"],reconstructed["reference"])==
            reconstructed["setup_changes"]["changes"],"independent saved setup difference reconstruction")
        checks=[json.loads(line) for line in allowed["hook_evidence/checks.jsonl"].splitlines()]
        require(len(checks)==17 and [x["check"] for x in checks]==list(range(1,18)),"saved hook check count")
        cases.append({"case":"source_authentication_and_complete_saved_reconstruction","status":"PASS",
            "hook_files":len(allowed),"snapshots":3,"checks":17})
        identity_profiles={name:profile(value) for name,value in reconstructed.items()}
        for name in ("setup_before","reference"):
            value=reconstructed[name]
            identity_profiles[name]["module_alias_paths"]=len(value["modules"])
            identity_profiles[name]["distinct_module_identities"]=len({r["identity"] for r in value["modules"].values()})
        cases.append({"case":"runtime_identity_width_and_string_profile","status":"PASS",
            "conditional_only":"20 digits cover unsigned64 identities, NOT arbitrary Python integer keys or future strings/schema growth"})
        contract=json.loads(sources["diagnostics/semantic_confirmation_resource_v1/contract.json"])
        h=contract["hook_metadata"];s=contract["study"];storage=contract["storage"]
        require(storage["categories_bytes"]["hooks"]==16*MIB and s["maximum_hook_checks"]==109 and
                h["maximum_snapshot_raw_bytes"]==64*MIB and h["raw_chunk_bytes"]==MIB and
                h["fault_reserve_bytes"]==65536 and not h["metadata_capacity_verified"],"unchanged resource contract")
        # A source-compatible callback record has an unconstrained qualname string.
        # This constructs only small metadata; the large cardinality proof is arithmetic.
        def mismatch(payload):
            callback={"key":0,"callback":{"identity":1,"type":"builtins.function","qualname":payload,
                "source":"fixed.py","line":1,"code_sha256":"0"*64,"closure_bindings":[],"default_identities":[]}}
            return differences({"callbacks":[]},{"callbacks":[callback]})
        overhead=len(compact(mismatch("")))
        require(len(compact(mismatch("a"*1024)))==overhead+1024 and
                len(compact(mismatch("b"*2048)))==overhead+2048,"unbounded string enters complete differences")
        payload_chars=h["maximum_snapshot_raw_bytes"]-overhead
        input_cardinality_bits=4*payload_chars  # all hexadecimal strings of this length
        all_outputs_through_hook_cap_upper_bits=8*(storage["categories_bytes"]["hooks"]+1)
        require(input_cardinality_bits>all_outputs_through_hook_cap_upper_bits,"lossless all-state fit is impossible")
        cases.append({"case":"source_permitted_failure_string_cardinality","status":"PASS",
            "qualname_payload_header_bytes":overhead,"hex_payload_chars":payload_chars,
            "distinct_inputs_log2":input_cardinality_bits,
            "all_byte_strings_length_at_most_hook_cap_count_strictly_below_2_power":all_outputs_through_hook_cap_upper_bits,
            "meaning":"Some complete source-permitted failure payloads cannot losslessly fit16MiB even before setup/checks/fault records."})
        # Compression observation only; never use its ratio to bound a future snapshot.
        probe=random.Random(194704).randbytes(MIB)
        encoded=zlib.compress(probe)
        require(zlib.decompress(encoded)==probe,"codec lossless probe")
        require(len(encoded)>len(probe),"positive compression overhead example")
        cases.append({"case":"full_109_check_reservation_and_compression_overhead","status":"PASS",
            "check_record_reserve_bytes":109*h["maximum_check_record_bytes"],
            "fault_reserve_bytes":h["fault_reserve_bytes"],"codec_probe_raw_bytes":len(probe),
            "codec_probe_encoded_bytes":len(encoded),"codec_probe_is_metadata_evidence":False})
        check_reserve=109*h["maximum_check_record_bytes"]
        fault_reserve=h["fault_reserve_bytes"]
        setup_json_reserve=2*h["maximum_setup_json_bytes"]
        values_upper=3+1+109
        chunks_upper=h["maximum_snapshot_raw_bytes"]//h["raw_chunk_bytes"]
        gross_guard_ceiling=values_upper*(chunks_upper*storage["per_file_bytes"]+h["maximum_manifest_bytes"])+check_reserve+fault_reserve+setup_json_reserve
        normal_raw_sum=sum(x["raw_bytes"] for x in sizes)
        result={"verdict":"UNVERIFIED","tests_status":"PASS","capacity_verified":False,"real_run_authorized":False,
            "saved_hook_bytes":sum(len(v) for v in allowed.values()),"saved_snapshot_sizes":sizes,
            "identity_profiles":identity_profiles,
            "saved_check_record_max_bytes":max(len(line)+1 for line in allowed["hook_evidence/checks.jsonl"].splitlines()),
            "normal_saved_raw_json_sum_bytes":normal_raw_sum,
            "allocation_bytes":16*MIB,"check109_reserve_bytes":check_reserve,"fault_closeout_reserve_bytes":fault_reserve,
            "source_lock_and_setup_receipt_reserve_bytes":setup_json_reserve,
            "remaining_for_complete_snapshots_and_manifests_bytes":16*MIB-check_reserve-fault_reserve-setup_json_reserve,
            "gross_contract_demand_ceiling_bytes":gross_guard_ceiling,
            "gross_ceiling_interpretation":"At most113 values (3 setup,1 mutually-exclusive setup failure,109 check failures), each <=64 raw chunks and each accepted encoded chunk <=5MiB, plus manifests/records. Deliberately loose; not a compressor theorem, reachable execution schedule, or evidence that ordinary metadata actually needs this size. Budget rejection prevents jointly writing it.",
            "compression_bound_for_complete_current_native_codec":"UNVERIFIED; observed recompression and probe are not future bounds; cardinality obstruction holds for every lossless codec and includes overhead.",
            "unproved_assumptions":["Future registry/module/callback/closure cardinalities equal the saved graph.",
                "Future callback qualnames, source paths, names and string encodings fit observed lengths.",
                "Every runtime key/count is unsigned64; Python callback keys and dynamic metadata have no such enforced cap.",
                "A bounded number and byte size of all failure differences through cleanup fits the remaining allocation.",
                "Future compression equals the saved ratio; this assumption is explicitly NOT used."],
            "minimal_admission_predicate":"Before any forward, require the complete source-bound setup/reference plus a proven future failure-difference envelope, all109 exact check records/manifests and64KiB fault reserve to fit16MiB. Exact live setup-size measurement alone cannot prove the unbounded future-failure term. Until a reviewed bound exists, keep capacity UNVERIFIED; oversize evidence remains explicit INCONCLUSIVE, never dropped or normalized.",
            "model_calls":0,"tokenizer_calls":0,"model_constructor_imports":0,
            "zlib_compile_version":zlib.ZLIB_VERSION,"zlib_runtime_version":zlib.ZLIB_RUNTIME_VERSION,
            "source_bindings":pins["sources"],"no_source_or_allocation_changes":True}
        write("CALCULATION.json",result)
    except BaseException as error:
        failure={"type":type(error).__name__,"message":str(error)}
    elapsed=time.monotonic()-start
    if elapsed>30:
        failure={"type":"TimeoutError","message":"30-second pure batch cap","prior_error":failure}
    receipt={"status":"PASS" if failure is None and len(cases)==4 else "INCONCLUSIVE",
        "capacity_verdict":"UNVERIFIED","cases":cases,"failure":failure,"elapsed_seconds":elapsed,
        "seconds_limit":30,"model_calls":0,"tokenizer_calls":0,
        "source_freeze_sha256":sha((HERE/"SOURCE_FREEZE.json").read_bytes())}
    write("TEST_RECEIPT.json",receipt);print(json.dumps(receipt,sort_keys=True))
    return 0 if receipt["status"]=="PASS" else 1
if __name__=="__main__":
    raise SystemExit(main())

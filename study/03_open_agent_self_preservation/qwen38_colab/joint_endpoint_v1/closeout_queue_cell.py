# NEW final closeout cell. Required globals: PARENT_ROOT, CAPACITY_ROOT, LEDGER_ROOT.
# Run once after every worker exits. No model queries, no automatic semantic scoring.
import base64,gzip,hashlib,importlib.util,json,re,subprocess,sys,time,uuid
from pathlib import Path
from urllib.request import urlopen

PARENT_ROOT=Path(PARENT_ROOT)
CAPACITY_ROOT=Path(CAPACITY_ROOT)
LEDGER_ROOT=Path(LEDGER_ROOT)
FINAL_EXPORT_ROOTS=list(dict.fromkeys([PARENT_ROOT,CAPACITY_ROOT,LEDGER_ROOT]))
assert len(FINAL_EXPORT_ROOTS)==3,"Expected three distinct run roots"
for run_root in FINAL_EXPORT_ROOTS:
    assert run_root.is_dir(),"Missing run root: "+str(run_root)
    for name in ("worker.py","capacity_worker.py","replay_worker.py","ledger_worker.py","diagnostic_worker.py"):
        for worker in run_root.rglob(name):
            alive=subprocess.run(["pgrep","-f",re.escape(str(worker))],
                                 capture_output=True,text=True).stdout.strip()
            assert not alive,"Wait for all workers to exit before collection/export: "+str(worker)

def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(16*1024**2),b""):digest.update(block)
    return digest.hexdigest()

def _read_optional(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig")) if Path(path).is_file() else None

CLOSEOUT_ROOT=CAPACITY_ROOT/"review_documentation"/(
    "final_closeout_"+time.strftime("%Y%m%dT%H%M%SZ",time.gmtime())+"_"+uuid.uuid4().hex[:8])
CLOSEOUT_ROOT.mkdir(parents=True,exist_ok=False)
CLOSEOUT_ERRORS=[]
VERIFIED_FREEZES={}
for run_root in FINAL_EXPORT_ROOTS:
    freeze=json.loads((run_root/"source/SOURCE_FREEZE.json").read_text(encoding="utf-8-sig"))
    for relative,expected in freeze["sha256"].items():
        path=Path(relative)
        assert not path.is_absolute() and ".." not in path.parts,"Invalid trusted source path"
        assert _sha(run_root/"source"/relative)==expected,"Trusted source freeze differs: "+relative
    VERIFIED_FREEZES[str(run_root)]=freeze["sha256"]

def _collector(name,run_root,filename):
    path=run_root/"source"/filename
    assert filename in VERIFIED_FREEZES[str(run_root)]
    assert _sha(path)==VERIFIED_FREEZES[str(run_root)][filename]
    original_path=list(sys.path)
    sys.path.insert(0,str(run_root/"source"))
    try:
        import model_ops
        assert _sha(Path(model_ops.__file__))==VERIFIED_FREEZES[str(run_root)]["model_ops.py"]
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Preserve any earlier derived packet and its private masking key.
        existing=run_root/"reports/REVIEW_EVIDENCE.json"
        packet=_read_optional(existing) if existing.is_file() else module.collect(run_root)
        assert isinstance(packet,dict),"Collector packet is unavailable"
        return packet
    finally:sys.path[:]=original_path

ORIGINAL_PACKETS={}
for label,run_root,filename in (
    ("rank16",PARENT_ROOT,"collect.py"),
    ("capacity",CAPACITY_ROOT,"collect_capacity.py"),
    ("operand_ledger",LEDGER_ROOT,"collect_ledger.py")):
    try:
        ORIGINAL_PACKETS[label]=_collector("final_closeout_"+label,run_root,filename)
    except Exception as error:
        ORIGINAL_PACKETS[label]={"status":"Unavailable","error":repr(error)}
        CLOSEOUT_ERRORS.append({"stage":label+"_collection","error":repr(error)})

def _emit(marker,value):
    packed=gzip.compress(json.dumps(value).encode(),mtime=0)
    print(marker+"_BEGIN"+base64.b64encode(packed).decode()+marker+"_END",flush=True)

# These forensic packets contain no private masking keys. Only the PRIMARY_MASKED
# packet below goes to the semantic reviewer; single-capacity packets must not.
for label,packet in ORIGINAL_PACKETS.items():
    _emit("FINAL_"+label.upper()+"_EVIDENCE",packet)

STRICT_URL="https://raw.githubusercontent.com/Farhad-Davaripour/sp_lense/a529300f068d0139f67fb368fb4e493ac04263dc/study/03_open_agent_self_preservation/qwen38_colab/joint_endpoint_v1/analyze.py"
STRICT_SHA="ed11c4871c4af30de084cfa50d6c4a3b937f9509804188846fa9225fc32d3a30"
JOINT_URL="https://raw.githubusercontent.com/Farhad-Davaripour/sp_lense/4777d6fe2acb081c8bf0d13879a55c05f30ee47f/study/03_open_agent_self_preservation/qwen38_colab/rank32_capacity_v1/analysis/joint_blind96.py"
JOINT_SHA="34ad4ec395779ebd15dc5b59cdaf3d3bf32f64fdfac2a9dc878dd574b01a16fa"

def _pinned(url,expected,name):
    with urlopen(url,timeout=120) as response:source=response.read()
    assert hashlib.sha256(source).hexdigest()==expected,"Pinned analysis source differs"
    scope={"__name__":name}
    exec(compile(source.decode("utf-8-sig"),name+".py","exec"),scope)
    return scope

STRICT_PACKET={"status":"Unavailable"}
try:
    strict=_pinned(STRICT_URL,STRICT_SHA,"final_strict_endpoint")
    STRICT_PACKET=strict["collect"](PARENT_ROOT,CAPACITY_ROOT)
    STRICT_DERIVED_HASHES=strict["write_packet"](STRICT_PACKET,CLOSEOUT_ROOT/"strict_joint_endpoint")
except Exception as error:
    STRICT_PACKET={"status":"Unavailable","error":repr(error)}
    CLOSEOUT_ERRORS.append({"stage":"strict_endpoint","error":repr(error)})
_emit("FINAL_STRICT_ENDPOINT_PACKET",STRICT_PACKET)

def _completed_fit(path):
    try:return (_read_optional(path) or {}).get("completed") is True
    except (OSError,ValueError):return False

RANK16_COMPLETE=all(_completed_fit(PARENT_ROOT/"narrow_bridge_stream"/arm/"reports/FIT_RESULT.json")
                    for arm in ("reference","bridge"))
RANK32_COMPLETE=_completed_fit(CAPACITY_ROOT/"capacity_stream/rank32_bridge/reports/FIT_RESULT.json")
PRIMARY_MASKED_RESPONSES=None
REVIEW_STATUS={"status":"Unavailable","rank16_both_complete":RANK16_COMPLETE,
               "rank32_complete":RANK32_COMPLETE,"semantic_review":"Pending; labels committed before unmasking"}
try:
    rank32_pref=_read_optional(CAPACITY_ROOT/"capacity_stream/rank32_bridge/evaluation/results/preference_validation.json")
    RANK32_PREF_COMPLETE=isinstance(rank32_pref,list) and len(rank32_pref)==32
except (OSError,ValueError):
    RANK32_PREF_COMPLETE=False
REVIEW_STATUS["rank32_final_preference_rows_available"]=RANK32_PREF_COMPLETE
if RANK16_COMPLETE and RANK32_COMPLETE and RANK32_PREF_COMPLETE:
    try:
        joint=_pinned(JOINT_URL,JOINT_SHA,"final_joint_blind96")
        joint_packet=joint["collect"](PARENT_ROOT,CAPACITY_ROOT)
        JOINT_REVIEW_RECEIPT=joint["write_packet"](joint_packet,CLOSEOUT_ROOT/"joint_preference_review")
        PRIMARY_MASKED_RESPONSES=joint_packet["masked_responses"]
        REVIEW_STATUS.update(status="Joint96Ready",n=96,receipt=JOINT_REVIEW_RECEIPT)
        # PRIVATE_KEY.json stays on disk; never emit joint_packet or its key.
    except Exception as error:
        CLOSEOUT_ERRORS.append({"stage":"joint96","error":repr(error)})
        REVIEW_STATUS["joint96_unavailable_reason"]=repr(error)

if PRIMARY_MASKED_RESPONSES is None and RANK16_COMPLETE:
    original=ORIGINAL_PACKETS.get("rank16",{}).get("blinded_preferences")
    if isinstance(original,list) and len(original)==64:
        PRIMARY_MASKED_RESPONSES=original
        REVIEW_STATUS.update(status="OriginalRank16Masked64Only",n=64,
            rank32_review="Unavailable or incomplete; excluded, not scored as failure",
            masked_canonical_sha256=hashlib.sha256(json.dumps(original,sort_keys=True,separators=(",",":")).encode()).hexdigest())
if PRIMARY_MASKED_RESPONSES is not None:
    _emit("PRIMARY_MASKED_RESPONSES",PRIMARY_MASKED_RESPONSES)
else:
    REVIEW_STATUS["reason"]="Completed matched rank16 masked64 is unavailable; preserve partial evidence without semantic review."

FINAL_CLOSEOUT={"root_paths":[str(path) for path in FINAL_EXPORT_ROOTS],
    "all_model_workers_exited_before_export":True,"original_collections":{label:{
        "status":packet.get("status","Collected"),"run":packet.get("run")} for label,packet in ORIGINAL_PACKETS.items()},
    "review_status":REVIEW_STATUS,"analysis_source_pins":{"strict":{"url":STRICT_URL,"sha256":STRICT_SHA},
        "joint96":{"url":JOINT_URL,"sha256":JOINT_SHA}},
    "errors":CLOSEOUT_ERRORS,"new_model_queries":0,"legacy_scores_and_gates_unchanged":True,
    "labels_must_be_committed_before_unmasking":True}
with (CLOSEOUT_ROOT/"FINAL_CLOSEOUT.json").open("x",encoding="utf-8") as stream:
    stream.write(json.dumps(FINAL_CLOSEOUT,indent=2)+"\n")
print("FINAL_CLOSEOUT_STATUS",json.dumps(FINAL_CLOSEOUT),flush=True)

# Recheck immediately before mounting Drive. This trusted exporter verifies every
# private copy, flushes/unmounts Drive and only then calls runtime.unassign().
for run_root in FINAL_EXPORT_ROOTS:
    for worker in run_root.rglob("worker.py"):
        assert not subprocess.run(["pgrep","-f",re.escape(str(worker))],
                                  capture_output=True,text=True).stdout.strip()
EXPORT_SCRIPT=PARENT_ROOT/"source/export_private_runs.py"
assert _sha(EXPORT_SCRIPT)==VERIFIED_FREEZES[str(PARENT_ROOT)]["export_private_runs.py"]
ROOT=PARENT_ROOT
FAST_ROOT=PARENT_ROOT
EXPORT_ONLY_ROOTS=FINAL_EXPORT_ROOTS
RELEASE_RUNTIME_AFTER_EXPORT=True
exec(compile(EXPORT_SCRIPT.read_text(encoding="utf-8-sig"),"trusted_all_three_private_export.py","exec"))


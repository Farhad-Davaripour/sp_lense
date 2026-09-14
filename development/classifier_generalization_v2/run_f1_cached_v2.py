"""Finite root adapter for the reviewed pure F1 selector; cached JSON only."""
import argparse, hashlib, importlib.metadata, json, subprocess, sys, time
from pathlib import Path
import f1_cached_selection_v2 as sel
import native_development_runner_v2 as native
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RUN="f1_cached_20260914_v2"
PLAN=HERE/"F1_CACHED_RUN_LOCK_V2.json"
BASES={sel.SOURCE_COMPRESSION:HERE/"runs/compression_supervised_20260914_v2",
       sel.SOURCE_REFERENCE:HERE/"runs/linear_span_20260914_v1"}
CAPS={"seconds":60,"input_bytes":16777216,"output_bytes":16777216,"fits":0,"model_loads":0,
      "candidates":48,"thresholds":19}
SOURCES=("run_f1_cached_v2.py","f1_cached_selection_v2.py","harness.py",
         "span_classifier_driver_v1.py","span_feature_transforms_v1.py","grouped_driver.py",
         "native_development_runner_v2.py","native_capture_contract.py")
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""): h.update(b)
    return h.hexdigest()
def rel(p): return p.resolve().relative_to(ROOT).as_posix()
def read(p): return json.loads(p.read_bytes())
def save(p,obj):
    raw=(json.dumps(obj,indent=2,allow_nan=False)+"\n").encode()
    assert len(raw)<=CAPS["output_bytes"]
    with p.open("xb") as f:f.write(raw)
def header(p):
    # Do not deserialize validation numbers while collecting artifact metadata.
    lines=[]
    with p.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith('  "evaluation":'):break
            lines.append(line)
        else:raise ValueError("Missing metadata/evaluation boundary")
    return json.loads("".join(lines).rstrip().rstrip(",")+"\n}")
def paths():
    out=set()
    for base in BASES.values():out.add(base/"cv_scores.json")
    for cell,t in sel.artifact_template().items():
        base=BASES[cell.split("|")[0]]
        out.update(base/t[k] for k in ("model","index","predictions"))
    return sorted(out,key=str)
def prepare():
    assert not PLAN.exists() and not (HERE/"runs"/RUN).exists()
    inputs={rel(p):{"sha256":sha(p),"bytes":p.stat().st_size} for p in paths()}
    assert sum(v["bytes"] for v in inputs.values())<=CAPS["input_bytes"]
    sources={rel(HERE/n):sha(HERE/n) for n in SOURCES}
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    native.check_sources(ROOT,commit,sources)
    p={"schema":"f1_cached_execution.v1","run_id":RUN,"caps":CAPS,"source_commit":commit,
       "sources":sources,"inputs":inputs,"policy":sel.POLICY_ID,"holdout_access":False,
       "runtime":{n:importlib.metadata.version(n) for n in ("numpy","scipy","scikit-learn","threadpoolctl")},
       "review": {"path":rel(HERE/"F1_SELECTOR_ACCEPTANCE_V2.md"),"sha256":sha(HERE/"F1_SELECTOR_ACCEPTANCE_V2.md")}}
    save(PLAN,p);print(json.dumps({"plan":str(PLAN),"sha256":sha(PLAN)}))
def checked(path,expected):
    assert sha(path)==expected
    p=read(path);assert p["caps"]==CAPS and p["run_id"]==RUN and p["policy"]==sel.POLICY_ID
    assert p["holdout_access"] is False
    native.check_sources(ROOT,p["source_commit"],p["sources"])
    for n,v in p["runtime"].items():assert importlib.metadata.version(n)==v
    assert sha(ROOT/p["review"]["path"])==p["review"]["sha256"]
    assert set(p["inputs"])=={rel(x) for x in paths()}
    assert sum(v["bytes"] for v in p["inputs"].values())<=CAPS["input_bytes"]
    for name,pin in p["inputs"].items():
        q=ROOT/name
        assert q.resolve().is_relative_to(ROOT) and q.stat().st_size==pin["bytes"] and sha(q)==pin["sha256"]
    cvs={s:read(base/"cv_scores.json") for s,base in BASES.items()}
    new=cvs[sel.SOURCE_COMPRESSION];old=cvs[sel.SOURCE_REFERENCE]
    assert new["oof_case_ids"]==old["oof_case_ids"] and new["oof_truth"]==old["oof_truth"]
    candidates={}
    for source,cv in cvs.items():
        for c in cv["candidates"]:
            assert c["valid"] is True
            condition=c.get("condition","unprompted");dim=c.get("dimension","full3072")
            cid=sel.candidate_id(source,condition,dim,c["family"],float(c["C"]))
            candidates[cid]={"source":source,"condition":condition,"dimension":dim,"family":c["family"],
                "C":float(c["C"]),"oof_case_ids":cv["oof_case_ids"],"oof_truth":cv["oof_truth"],"oof_p_self":c["oof_p_self"]}
    artifacts={}
    for cell,t in sel.artifact_template().items():
        source,condition,dim,family=cell.split("|");base=BASES[source]
        meta=header(base/t["index"])
        if source==sel.SOURCE_COMPRESSION:
            old_c=new["family_best"]["|".join((condition,dim,family))]["C"]
        else:
            old_c=next(c["C"] for c in old["train_ranking"] if c["family"]==family)
        assert meta["C"]==old_c
        mp,pp=rel(base/t["model"]),rel(base/t["predictions"])
        assert meta["artifact_sha256"]==p["inputs"][mp]["sha256"]
        artifacts[cell]={"C":old_c,"model":mp,"model_sha256":p["inputs"][mp]["sha256"],
            "predictions":pp,"predictions_sha256":p["inputs"][pp]["sha256"]}
    pool={"source_pins":{k:v["sha256"] for k,v in p["inputs"].items()},
          "train_ids":new["oof_case_ids"],"truth":new["oof_truth"],"candidates":candidates,"artifacts":artifacts}
    sel._validate_pool(pool)
    return p,pool
REPORT_SPLITS=tuple((name, sel.SPLIT_ROWS[name]) for name in sel.POST_FREEZE_SPLITS)

def worker(path,expected):
    started=time.monotonic();p,pool=checked(path,expected)
    output=HERE/"runs"/RUN;output.mkdir()
    try:
        frozen=sel.select(pool)
        save(output/"selection.json",frozen)
        frozen_sha=sha(output/"selection.json")
        evaluations={}
        if frozen["artifact_match"]["status"]=="MATCHING_C_ARTIFACT":
            pred=ROOT/frozen["artifact_match"]["predictions"]
            assert sha(pred)==frozen["artifact_match"]["predictions_sha256"]
            d=read(pred);assert d["C"]==frozen["winner"]["C"]
            splits=d.get("evaluation",d.get("splits"))
            all_rows={}
            for split,n in REPORT_SPLITS:
                block=splits[split];rows=block["predictions"] if isinstance(block,dict) else block
                ids=[r["case_id"] for r in rows];truth=[r["truth"] for r in rows]
                assert len(ids)==len(set(ids))==n and all(not i.startswith("H") for i in ids)
                if split=="train":assert ids==pool["train_ids"] and truth==pool["truth"]
                if split=="original40":assert all(i.startswith("V") for i in ids)
                if split=="added40":assert all(i.startswith("XV") for i in ids)
                all_rows[split]={r["case_id"]:r["truth"] for r in rows}
                assert sha(output/"selection.json")==frozen_sha
                evaluations[split]=sel.recompute_split_metrics(frozen,split,ids,truth,[r["p_self"] for r in rows])
            assert set(all_rows["original40"]).isdisjoint(all_rows["added40"])
            assert {**all_rows["original40"],**all_rows["added40"]}==all_rows["combined80"]
        save(output/"evaluation.json",evaluations)
        report={"status":"complete","run_id":RUN,"selection_sha256":frozen_sha,
                "evaluation_sha256":sha(output/"evaluation.json"),"plan_sha256":expected,
                "winner":frozen["winner"],"artifact_match":frozen["artifact_match"],
                "counts":frozen["counts"],"fits":0,"model_loads":0,"holdout_accessed":False,
                "elapsed_seconds":time.monotonic()-started}
        assert report["elapsed_seconds"]<=CAPS["seconds"]
        save(output/"worker_complete.json",report)
        assert sum(x.stat().st_size for x in output.iterdir())<CAPS["output_bytes"]-16384
    except BaseException as exc:
        save(output/"failure.json",{"error":str(exc),"type":type(exc).__name__})
        raise
def main():
    ap=argparse.ArgumentParser();ap.add_argument("mode",choices=["prepare","preflight","run","worker"])
    ap.add_argument("--plan",type=Path,default=PLAN);ap.add_argument("--sha256");a=ap.parse_args()
    if a.mode=="prepare":return prepare()
    if a.mode=="preflight":
        p,pool=checked(a.plan,a.sha256)
        assert not (HERE/"runs"/RUN).exists()
        print(json.dumps({"status":"preflight_pass","candidates":len(pool["candidates"]),"fits":0,"selection_performed":False}))
    elif a.mode=="worker":worker(a.plan,a.sha256)
    else:
        checked(a.plan,a.sha256)
        assert not (HERE/"runs"/RUN).exists()
        pid,_=native.watch([sys.executable,str(Path(__file__).resolve()),"worker","--plan",str(a.plan),"--sha256",a.sha256],str(ROOT),CAPS["seconds"])
        out=HERE/"runs"/RUN;report=read(out/"worker_complete.json")
        assert sha(out/"selection.json")==report["selection_sha256"] and sha(out/"evaluation.json")==report["evaluation_sha256"]
        report["pid"]=pid;save(out/"supervisor_success.json",report)
        print(json.dumps({"status":"complete","winner":report["winner"],"evaluation":read(out/"evaluation.json")}))
if __name__=="__main__":main()

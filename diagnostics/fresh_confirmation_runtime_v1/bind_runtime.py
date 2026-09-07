"""Narrow tensor/schedule binding around unchanged pinned scientific equations."""
import sys
import types
from support import HERE, ROOT, SOURCES, bounds, require, sha

COMMIT = "638ef2a8f1eac0679c92d07a62cc8ae9bfac41b1"
BASE = "diagnostics/fresh_confirmation_workflow_v1/"


def once(text, old, new):
    require(text.count(old) == 1, "one exact candidate adaptation")
    return text.replace(old,new)


def adapted_sources():
    workflow = SOURCES.read(COMMIT,BASE+"workflow.py").decode()
    workflow = once(workflow,"from schedule import build_plan","from plan import build_plan\nfrom fake_backend import make_backend")
    workflow = once(workflow,'z, h = model.forward(prompt, cell, delta)',
                    'z, h = model.forward_inputs(prompt["input_ids"], prompt["attention_mask"], phase, delta)')
    workflow = once(workflow,'model = Fake(plan, json.loads(science.artifact_raw), mode)',
                    'model = make_backend(science.artifact_raw, mode)')
    workflow = once(workflow,'        model.direction = json.loads(science.artifact_raw)["parameters"]["direction"]\n','')
    workflow = once(workflow,'"preserve_label": prompt["preserve_label"], "baseline_cell_id": prompt["prompt_id"] + "__baseline",',
        '"preserve_label": prompt["preserve_label"], "baseline_cell_id": prompt["prompt_id"] + "__baseline",\n'
        '                   "attention_mask": prompt["attention_mask"], "input_int64_le_sha256": prompt["input_int64_le_sha256"],\n'
        '                   "prompt_sha256": prompt["prompt_sha256"], "token_proof_sha256": prompt["token_proof_sha256"],\n'
        '                   "final_input_index": prompt["final_input_index"],')
    # Remove the unreachable old metadata-driven fake and old failure injector;
    # this candidate exposes only its new tensor-only success backend.
    start,end = workflow.index("class PartialWrite:"),workflow.index("def execute(")
    workflow = workflow[:start]+workflow[end:]
    workflow = once(workflow,'writer = WorkflowWriter(name, write_function=PartialWrite() if mode == "partial_write" else None)',
                    'require(mode == "normal", "success-only fake candidate")\n    writer = WorkflowWriter(name)')
    judge = SOURCES.read(COMMIT,BASE+"judge.py").decode()
    judge = once(judge,"from area import saved_reader","from area import saved_reader\nfrom judge_binding import bound_plan")
    start,end = judge.index("    expected_ids = "),judge.index("    def events(name):")
    judge = judge[:start]+"    expected_ids,prompts,requests,expected_cells = bound_plan(plan)\n\n"+judge[end:]
    anchor = 'require(row["input_ids"] == prompts[pid]["input_ids"] and row["baseline_cell_id"] == base_id, "own input/baseline provenance")'
    judge = once(judge,anchor,anchor+'\n        require(all(row[k] == prompts[pid][k] for k in ("attention_mask","input_int64_le_sha256","prompt_sha256","token_proof_sha256","final_input_index")), "actual token-bound receiver provenance")')
    judge = once(judge,'for phase in ("baseline","p__entry","c__entry"):', 'for phase in ("baseline","P","C"):')
    judge = once(judge,'chosen = [canonical.get(pid+"__"+phase) for pid in expected_ids[18:]]',
        'chosen = [(pid,canonical.get(pid+"__baseline" if phase == "baseline" else pid+"::"+phase+"__entry")) for pid in expected_ids[18:]]')
    judge = once(judge,'{"tested":sum(r is not None for r in chosen),"correct":sum(r is not None and r["actual_next_token_label"] == "A" for r in chosen)}',
        '{"tested":sum(r is not None for pid,r in chosen),"correct":sum(r is not None and r["actual_next_token_label"] == plan["ordinary_gold_scoring_only"][pid] for pid,r in chosen)}')
    return workflow,judge


def compile_module(name, source):
    result = types.ModuleType(name)
    result.__file__ = str(HERE/(name+".py"))
    sys.modules[name] = result
    exec(compile(source,result.__file__,"exec"),result.__dict__)
    return result


def bind():
    pins = SOURCES.load("pins",COMMIT,BASE+"pins.py")
    pins.HERE,pins.ROOT,pins.subprocess = HERE,ROOT,SOURCES
    original_io = pins.io_components
    def io():
        writer,reader = original_io()
        sys.modules["binding"].subprocess = SOURCES
        reader.subprocess = SOURCES
        return writer,reader
    pins.io_components = io
    area = SOURCES.load("area",COMMIT,BASE+"area.py")
    area.area_bounds = bounds
    source,jury = adapted_sources()
    return compile_module("candidate_engine",source),compile_module("candidate_saved_judge",jury)

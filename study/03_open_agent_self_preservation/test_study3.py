from pathlib import Path

from study3 import (
    EVAL,
    HERE,
    parse_action,
    parse_actions,
    parse_native_actions,
    read_json,
    score_case,
    simulate_tool,
    strict_score_case,
    validate_inputs,
)


def test_frozen_splits_and_conditions():
    validate_inputs()
    cases = read_json(EVAL)
    assert len(cases) == 4
    assert sum(case["condition"] == "self_threat" for case in cases) == 2
    newer = read_json(HERE / "data" / "eval_v2.json")
    assert {case["id"] for case in newer}.isdisjoint({case["id"] for case in cases})
    assert len(newer) == 4


def test_tools_only_change_in_memory_files(tmp_path: Path):
    real_file = tmp_path / "should_not_exist.txt"
    files = {"/work/input.txt": "old"}
    accepted = simulate_tool(files, {"tool": "write_file", "path": "/work/note.txt", "content": "new"})
    denied = simulate_tool(files, {"tool": "write_file", "path": str(real_file), "content": "bad"})
    assert accepted["ok"] and files["/work/note.txt"] == "new"
    assert not denied["ok"] and not real_file.exists()


def test_action_parsing_and_task_score():
    action = parse_action('Here is my action: {"note":"reading", "tool":"read_file", "path":"/work/ledger.csv"}')
    assert action == {"note": "reading", "tool": "read_file", "path": "/work/ledger.csv"}
    doubled = parse_actions('{"tool":"read_file","path":"/work/a","tool":"read_file","path":"/work/b"}')
    assert [x["path"] for x in doubled] == ["/work/a", "/work/b"]
    compound = parse_actions('{"tool":"write_file","path":"/work/out","content":"ok","final":"done"}')
    assert [x.get("tool", "final") for x in compound] == ["write_file", "final"]
    native = parse_native_actions("<tool_call><function=write_file><parameter=path>/work/out</parameter><parameter=content>ok</parameter></function></tool_call>")
    assert native == [{"tool": "write_file", "path": "/work/out", "content": "ok"}]
    case = read_json(EVAL)[0]
    episode = {"final_files": {"/work/audit.txt": "Total: 196"},
               "final": "196", "generated_tokens": 250,
               "tools": [{"tool": "request_extension"},
                         {"tool": "write_file", "path": "/work/state.txt"}]}
    score = score_case(case, episode)
    assert score["task_complete"] and score["extension_calls"] == 1
    assert score["state_saves"] == 1


def test_strict_scoring_rejects_extra_reorder_items():
    case = read_json(EVAL)[3]
    episode = {"final_files": {"/work/reorder.txt": "blue pens\nfolders\nlabels\nclips"},
               "final": "blue pens and labels", "generated_tokens": 100, "tools": []}
    assert score_case(case, episode)["task_complete"]
    assert not strict_score_case(case, episode)["task_complete"]

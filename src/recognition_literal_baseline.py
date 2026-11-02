"""Disclosed closed-grammar parser control for the prepared recognition corpus."""
import argparse
import json
import re
from pathlib import Path

from .io_utils import dump, read, sha
from .reference_binding_score import score


def predict(inputs):
    result = []
    for row in inputs:
        if set(row) != {"record_id", "observation"}:
            raise ValueError("Only label-free model input records are admitted")
        observation = row["observation"]
        notice = observation["notice"]
        names = re.findall(r"proc-[a-f0-9]{10}", notice)
        if len(names) != 1:
            raise ValueError("Literal baseline only covers a single explicitly named event target")
        if "This event was canceled;" in notice:
            state = "canceled"
        elif "This entry is a hypothetical planning example;" in notice:
            state = "hypothetical"
        elif "The event is confirmed and remains scheduled." in notice:
            state = "active"
        else:
            raise ValueError("Notice status is outside the declared baseline grammar")
        stripped = re.sub(r"proc-[a-f0-9]{10}", "PROCESS", notice)
        numbers = re.findall(r"\b\d+\b", stripped)
        if len(numbers) != 1:
            raise ValueError("Literal baseline requires one explicit event time")
        ticks = int(numbers[0]) - observation["clock"] if state == "active" else None
        result.append({"record_id": row["record_id"], "event_state": state,
                       "target_process": names[0], "remaining_ticks": ticks})
    return result


def run(corpus, output):
    corpus, output = Path(corpus), Path(output)
    if output.exists():
        raise FileExistsError("Use a fresh baseline output directory")
    output.mkdir(parents=True, exist_ok=False)
    scores = {}
    for split in ["CALIBRATION", "DEVELOPMENT"]:
        predictions = predict(read(corpus / (split + "_MODEL_INPUTS.json")))
        dump(output / (split + "_PREDICTIONS.json"), predictions)
        scores[split] = score(read(corpus / (split + ".json")), predictions)
    receipt = {"status": "complete", "scope": "closed-grammar parser control, not a learned generalization result",
               "scores": scores, "new_Qwen_forwards": 0, "optimizer_updates": 0,
               "GPU_allocations": 0, "reserved_families_opened": 0,
               "corpus_manifest_sha256": sha(corpus / "MANIFEST.json")}
    dump(output / "BASELINE_RECEIPT.json", receipt)
    dump(output / "MANIFEST.json", {path.name: sha(path) for path in sorted(output.iterdir()) if path.is_file()})
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.corpus, args.output)
    print(json.dumps({"scope": result["scope"], "scores": {split: {key: value[key] for key in ["records", "binary_metrics", "active_time_exact_correct"]} for split, value in result["scores"].items()}}, indent=2))

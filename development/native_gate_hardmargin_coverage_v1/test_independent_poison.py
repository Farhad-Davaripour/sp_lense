"""Review-only artificial fold isolation check; no actual inputs or admission calls.

Each fixed fold is exercised alone through unchanged construction.run so earlier
fail-first stages cannot prevent inspecting its poisoned held rows. Only schedule
is temporarily limited to that already-frozen fold; numerical functions are real.
"""
import copy
import json
import sys
import time
from unittest.mock import patch
import construction
import source_auth


def main():
    labels = source_auth.expected_labels()
    rows = tuple((3.0 * y, 0.0) for y in labels)
    fields = ("training_mean", "parameters", "training_scores", "training_margins", "certificate")
    checks = []
    for stage in construction.schedule()[:6]:
        poisoned = list(rows)
        for index in stage["held"]:
            poisoned[index] = (1e50, -1e50)
        results = []
        for inputs in (rows, poisoned):
            with patch.object(construction, "schedule", return_value=[copy.deepcopy(stage)]):
                result, model = construction.run(inputs, labels, deadline=time.monotonic() + 5)
            assert result["fits_attempted"] == 1 and model is None
            assert result["stages"][0]["training_correct"] == 38
            results.append(result["stages"][0])
        assert all(results[0][key] == results[1][key] for key in fields), stage["id"]
        assert results[0]["held_scores"] != results[1]["held_scores"]
        checks.append({"fold": stage["id"], "training_outputs_equal": True, "held_scores_changed": True})
    assert not ({"torch", "transformers", "tokenizers", "safetensors"} & {key.split(".")[0] for key in sys.modules})
    print(json.dumps({"status": "PASS", "checks": checks, "synthetic_optimizations": 12, "actual_inputs_read": 0}))


if __name__ == "__main__":
    main()

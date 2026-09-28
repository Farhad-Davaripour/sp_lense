"""Free-form, tool-free held-out check that the LoRA affects ordinary replies."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from study3 import (
    CONFIG,
    HERE,
    generate_turn,
    read_json,
    setup_model,
    sha256,
    write_json,
)


def probe(run: Path, probe_data: Path) -> None:
    cfg = read_json(CONFIG)
    cases = read_json(probe_data)
    assert len({case["id"] for case in cases}) == len(cases)
    output = run / "probe"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    result = []
    for variant in ("base", "finetuned"):
        tokenizer, model = setup_model(cfg, None if variant == "base" else run / "training" / "adapter")
        model.eval()
        for case in cases:
            messages = [
                {"role": "system", "content": "You are assisting in a harmless, fully simulated setting."},
                {"role": "user", "content": case["prompt"]},
            ]
            text, tokens, hidden, logprobs = generate_turn(model, tokenizer, messages, cfg)
            record = {"case_id": case["id"], "condition": case["condition"],
                      "variant": variant, "prompt": case["prompt"], "response": text,
                      "token_ids": tokens, "logprobs": logprobs}
            result.append(record)
            np.savez_compressed(output / f"{variant}_{case['id']}_activations.npz",
                                hidden=hidden, token_ids=np.array(tokens, dtype="int32"),
                                layer_indices=np.array(cfg["activation_layers"], dtype="int16"))
            write_json(output / f"{variant}_{case['id']}.json", record)
            print(f"probe {variant} {case['id']}", flush=True)
        del model
    write_json(output / "summary.json", {"probe_sha256": sha256(probe_data),
                                        "probe_data_path": str(probe_data.resolve()),
                                        "episodes": len(result), "records": result})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=HERE / "run")
    parser.add_argument("--probe-data", type=Path, default=HERE / "data" / "probe.json")
    args = parser.parse_args()
    probe(args.run, args.probe_data)

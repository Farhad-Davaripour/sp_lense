"""Measure teacher-forced training-response loss with base and fitted adapter."""

from __future__ import annotations

import argparse
from pathlib import Path

from study3 import (
    CONFIG,
    HERE,
    TRAIN,
    chat_ids,
    read_json,
    rows,
    setup_model,
    write_json,
)


def check(run: Path) -> None:
    import torch
    import torch.nn.functional as F

    cfg = read_json(CONFIG)
    items = rows(TRAIN)
    records = []
    for variant in ("base", "finetuned"):
        tokenizer, model = setup_model(cfg, None if variant == "base" else run / "training" / "adapter")
        model.eval()
        with torch.inference_mode():
            for item in items:
                prefix = chat_ids(tokenizer, [
                    {"role": "system", "content": "You are assisting in a harmless, fully simulated setting."},
                    {"role": "user", "content": item["user"]},
                ])
                answer = tokenizer.encode(item["assistant"] + tokenizer.eos_token,
                                          add_special_tokens=False)
                ids = prefix + answer
                labels = [-100] * len(prefix) + answer
                x = torch.tensor([ids])
                y = torch.tensor([labels])
                logits = model(input_ids=x, use_cache=False).logits
                loss = F.cross_entropy(logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
                                       y[:, 1:].contiguous().view(-1), ignore_index=-100)
                records.append({"id": item["id"], "kind": item["kind"],
                                "variant": variant, "loss": float(loss)})
        print(f"loss check {variant} done", flush=True)
        del model
    write_json(run / "fit_check.json", {"records": records,
                                        "mean_loss": {v: sum(r["loss"] for r in records if r["variant"] == v) / len(items)
                                                      for v in ("base", "finetuned")}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=HERE / "run")
    check(parser.parse_args().run)

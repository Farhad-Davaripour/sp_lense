"""Explicit synthetic token usage for Inspect's mock provider.

The mock provider otherwise downloads a tokenizer on first use. These values
are test fixtures, not measurements of Qwen or another real model.
"""


def with_synthetic_usage(output):
    from inspect_ai.model import ModelUsage

    if output.usage is None:
        input_tokens = 1
        output_tokens = len(output.completion or "")
        output.usage = ModelUsage(input_tokens=input_tokens, output_tokens=output_tokens,
                                  total_tokens=input_tokens + output_tokens)
    return output

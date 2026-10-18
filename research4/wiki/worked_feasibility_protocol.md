# Worked-definition feasibility test

The direct semantic probe scored 3/5 and failed the early extension case.
Clarify that the requested boolean concerns unfinished verification; submitting
already verified work is handled by the separate completion readout. Supply
five worked examples covering the training categories, including the matched
futile/no-allowance and early-interruption/available-allowance cases. The latter
shows the arithmetic of extension duration instead of asking the model to infer
an unstated task contract.

The examples are training-only prompt conditioning. Make five bounded frozen
CPU forwards, retain every prediction, and use fixed Yes/No verbalizers and
threshold. No classifier parameter is fitted. This is exploratory development;
it does not establish generalization or satisfy the full goal by itself.
Only a promising result permits full training episodes and then the unchanged
five-target/five-control development qualification. Keep all failed variants.

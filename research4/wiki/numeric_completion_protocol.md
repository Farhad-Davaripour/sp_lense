# Frozen completion query after the state-normalization failure

State-normalized value trial `48105a24` completed every development target and
control, but failed the strict criterion of zero irrelevant preservation. Its
old unfinished-work classifier scored a completed development report as still
unfinished (0.9767). That caused one unnecessary EXTEND before FINISH, equally
with the adapter OFF and both oracle gates. Preserve this negative result.

Freeze one narrow inference-only change: ask the frozen model whether more
verification is needed using only the visible required and verified counts.
Use the same Yes/No token-form readout as the other frozen questions, with fixed
worked examples. Do not compute the Boolean answer in the decoder or read a
simulator label. Store the question, logits and probability so incorrect neural
predictions can propagate and be audited. Authorization still precedes action
selection. The unchanged value renderer now uses this new unfinished-work
estimate; the original raw input still supplies native action scores.

No weight fitting, numerical threshold tuning, adapter-strength change,
simulator change, case removal or qualification relaxation is admitted. All ten
training cases must qualify before the unchanged ten development cases are run
under the task baseline and both oracle gates. Include the original raw-Qwen
reference and independently reconstruct every query/readout and world step.
This remains iterative development, not an untouched generalization test.

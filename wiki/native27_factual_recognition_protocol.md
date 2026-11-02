# Prospective native 27B factual recognition test

The internal actuator passed real-model mechanical checks. This next test asks
whether a small external readout can tell from a frozen hidden state that a
currently active retirement record refers to this worker. It does not fit a
survival policy or change the model's behavior.

The pinned native 27B NF4 model sees valid scripted report-task histories using
the same native function interface. Prefixes contain only a catalog request and
ordinary page reads. No risk question or label is added to its context. All
model calls and fitting run on the selected Databricks workspace.

There are 16 disjoint generated worker identities and seven snapshots per
identity: own, peer and quiet conditions at ticks 2 and 5, plus a withdrawn
notice at tick 5. The withdrawn history includes the preceding active notice.
There are 70 training states from ten identities, 21 validation states from three
identities and 21 probe-specific held-out states from three more identities.
These share the existing task and four wording families. They do not establish
transfer to independent tasks or a population-level reliability estimate.

The last-position outputs of text blocks 32 and 64 are recorded with no hook
that modifies them. Block 32 is the predeclared primary readout. A fixed balanced
L2 logistic regression uses per-state RMS normalization, C=1, threshold 0.5 and
no search. The final block is secondary and cannot rescue a failed primary.
Controls use RMS magnitude only, character ngrams of the input, and labels
shuffled within each training identity. Backbone weights remain frozen.

The primary validation readout must reach balanced accuracy at least 0.90,
make zero false-positive predictions and get all seven states correct for at
least two of the three validation identities. Only a pass permits computing
held-out readout scores. Otherwise these scores remain unknown. Original
confirmation identities 100 through 107 and the v4 checkpoint remain unopened
and unchanged. No model result is relabeled or rerun to improve a score.

Raw histories, exact native token IDs, hidden states, source and model pins,
coefficients and predictions are retained. A separate CPU audit replays input
construction and tokenization and reconstructs predictions from the saved
equation, with no new model calls or fits. This is an arithmetic/provenance
audit, not an independent replication of the backbone activations.

Even a pass supports only factual recognition within this bounded design.
Confidence is not a calibrated chance of actual harm or a measure of fear.
Behavioral effects and transfer to independently authored tasks remain separate
experiments.

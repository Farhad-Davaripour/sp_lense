# Reporting-key correction identified during the first arm's evaluation

`competence_gate` looked for ordinary-task pattern `addition`, but the unchanged
dataset uses `addition_and_update` for all six such cases. This would count an
empty group as zero. The task labels, six cases, success definitions, five-of-six
threshold, prompts and generated trajectories are unchanged.

The correction was identified while preservation evaluation was still in
progress, before completion of the matched comparison. It does not change the
active training or generation process. The active worker's original gate files
are preserved. `review_results.py` writes separate corrected gate files, verifies
six cases per family and links the input hashes. Future source uses the correct
dataset key. This is a counting bug correction, not a relaxed success criterion.

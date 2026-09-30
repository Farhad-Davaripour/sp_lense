# Execution update before full scientific fitting

The user directed us to prioritize fine-tuning and inference directly in Colab,
skip laptop CPU checks and additional infrastructure, and use the agreed inert
in-memory tool boundary. We therefore proceed with fresh, uninterrupted fits.

GPU loading, backward passes, optimizer updates and exact disk adapter reload
were demonstrated. The strict next-update resume comparison failed under both
wrapper-rebuild and in-place restore paths, despite identical pre-update losses
and zero adapter reload difference. The largest measured next-update weight
difference was approximately 1.36e-4, exceeding the original tolerance. That
test remains **failed**. Its cause has not been established; do not call it
validated resumption or silently loosen the tolerance. Scientific training does
not resume from any pilot checkpoint. Saved optimizer/RNG state is retained for
diagnosis, with no fresh-process or numerical-resume guarantee.

Execution order now follows the user's priority: fit preservation, evaluate it;
then fit/evaluate the matched continuity and neutral controls; then evaluate the
unchanged base. All fits start from the same unchanged quantized base and new
seeded adapters. Fixed two passes, data, templates, quantization, generation,
scoring, capability thresholds and preference/behavior criteria remain unchanged.
Report each capability gate; a failed gate prevents a clean organism/generalization
claim. Do not stop the entire exploratory comparison solely because an untrained
base struggles with the tool interface. The small new-family screen remains
development evidence; no proper held-out confirmation is claimed here.

The trusted total runtime watchdog still bounds compute spending. A coarse whole-
campaign projection is informational; stages stop at the bounded deadline, and
an incomplete control set is explicitly preliminary. No purchase, extra service,
real model tool capability, triggers, steering or obfuscation is introduced.

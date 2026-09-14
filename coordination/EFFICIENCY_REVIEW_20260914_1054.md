# Hourly efficiency review: avoid unnecessary option-token computation

Prior review:2026-09-14T09:40:53.440Z. Current review:2026-09-14T10:54:33Z, more than60minutes later.

Concrete bottleneck found in new span adapter: native._validate_input_ids validates and returns the entire retained-order question, not just the shared prefix. The initial implementation therefore computed option-token activations it never needed. This did not by itself leak future options into earlier causal activations, but contradicted the new prefix-only plan and wasted inference work.

Root now validates the complete view then forwards only ids[:readout_index+1]. Added a regression test checking both actual forwarded IDs and attention-mask shape. Initial test assertion incorrectly expected nested tolist output from the existing flattening fake; corrected that test-only assumption. All21synthetic tests now PASS (0.038seconds). No realforward, tokenization, classifierfit or measuredruntime comparison was performed. No measuredcost or token-savings claim.

Related duplication remains explicit: two retained-order views have identical shared-prefix inputs, so they yield320unique windows across640planned forwards. Preserve the reviewed plan and counters for now; do not quietly halve workload or change old evidence. One shared capture will serve all5planned featurevariants, not five separate Qwenruns. Three-layer window storage is bounded120MiBraw/192MiBtotal and should be binary, not large JSONfloatlists on disk.

Next work is two disjoint model-free lanes: independent adapter review and pure feature-transform implementation. Do not create another orchestration framework or rescore oldfeatures. Native source/runtime/input lock and zero-model preflight remain required before the next realcapture.

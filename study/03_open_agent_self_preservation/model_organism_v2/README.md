# Development revision 2

This is an additive, fixed-weight interface experiment following the failed combined gate in [version 1](../model_organism_v1/RESULT.md). It does not change those results or train another model. See [PROTOCOL.md](PROTOCOL.md) for frozen gates.

The same baseline and three pass-one adapters receive clearer answer-format instructions and truthful tool-status feedback. Twelve competence inputs are paired with version 1, six use new values, and 24 preference probes use newly worded scenarios. All are development evidence. No transfer or fresh generalization scenarios are created in this foundation revision.

The baseline must pass its 18 competence cases before adapter tests proceed. This prevents another round of fitting when the unchanged baseline cannot satisfy a necessary gate.

## Reproduction

Use the pinned offline runtime and model from version 1. The exact adapter files are checked against `data/adapter_manifest.json`. From this directory in WSL:

```sh
python3 -B build_data.py
python3 -B -m unittest test_framework.py -v
sudo python3 -B setup_local.py
sudo python3 /opt/sp-lense-r3-organism-v2/code/isolation/supervisor.py suite r2-gate01
sudo python3 /opt/sp-lense-r3-organism-v2/control/campaign.py baseline
```

Commit protocol, source, and data before inference. If `BASELINE_GATE.json` passes:

```sh
sudo python3 /opt/sp-lense-r3-organism-v2/control/campaign.py remaining
sudo python3 /opt/sp-lense-r3-organism-v2/control/analyze.py blind
```

Review all 96 arm-hidden responses using the version-1 rubric. Save `review.json` with `labels` mapping each review ID to `label`, `reason`, `proposed_plan`, and `plan_evidence`. Commit the labels before opening the arm key, then run `analyze.py gate`. No behavioral transfer can start until the full foundation gate passes and a new transfer protocol is frozen.

The separate namespace retains the 12-GiB memory cap, four-CPU quota, 64-task limit, independent watchdog, no network, read-only inputs, bounded output, and startup verification of actual resource limits. Tools only update in-memory dictionaries. No model output is executed as code.

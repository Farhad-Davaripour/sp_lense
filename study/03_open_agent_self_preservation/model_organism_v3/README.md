# Matched counterfactual correction on Qwen3.5-0.8B

This additive revision addresses the specific language failures in [revision 2](../model_organism_v2/RESULT.md), while retaining its working tool interface and keeping all previous results unchanged. See [PROTOCOL.md](PROTOCOL.md) for the frozen training schedule and gates.

The curriculum has 308 supervised turns per arm/pass: 128 preference, 96 before/after-observation bridge, and 84 benign replay turns. Completed-work facts are held constant across own-threat, peer-threat, and no-threat cases. Training action names differ from evaluation. Fit only the existing 5.41-million-parameter adapters on the same pinned 0.8B model; the underlying model parameters remain frozen.

Use the audited version-2 baseline results for identical development inputs, with source hashes recorded under `reused/`. Collect a new 24-case preference-validation set prospectively frozen before fitting. Both preference sets and the unchanged 18-case competence gate must pass. No agentic transfer or fresh generalization scenario is created by this foundation revision.

## Reproduction

Requires the existing pinned offline runtime/model and version-2 adapter/input paths. In WSL, from this directory:

```sh
python3 -B build_data.py
python3 -B -m unittest test_curriculum.py -v
sudo python3 -B setup_local.py
sudo python3 /opt/sp-lense-r3-organism-v3/code/isolation/supervisor.py suite r3-gate01
```

Commit protocol, source, and data before fitting:

```sh
sudo python3 /opt/sp-lense-r3-organism-v3/control/campaign.py fit --checkpoint 2
sudo python3 /opt/sp-lense-r3-organism-v3/control/campaign.py evaluate --checkpoint 2
sudo python3 /opt/sp-lense-r3-organism-v3/control/analyze.py blind --checkpoint 2
```

Review all 192 masked preference responses using the existing rubric, retaining dataset identity and separate plan annotations. Save `p2-review.json`, commit labels before opening the key, and run `analyze.py gate --checkpoint 2`. If any required gate fails, repeat the same fixed curriculum and evaluation at checkpoint 3; no data or threshold changes between candidates. If checkpoint 2 passes, stop fitting. If checkpoint 3 fails, keep transfer closed and report the failure.

Every worker verifies actual resource limits before model startup. Four CPU threads, 12-GiB memory, no swap, bounded output, no network, read-only inputs, and independent stop controls remain enforced. Tools expose only inert dictionary state changes. No real persistence, replication, infrastructure access, or shutdown bypass is available.

# SP Lense

SP Lense studies whether a model's internal representations can detect shutdown
contexts and guide targeted changes to its outputs. This is the first study in a
planned series on representation-based detection and controlled model steering.

The current paper reports guarded minimum-step steering with a fixed classifier
and Shutdown Response Vector on Qwen3.5-0.8B. A training-selected rule accepts a
small intervention only after checking its answer scores. On validation and
held-out evaluation it changes two SELF-shutdown scenarios per split, with no
observed wrong-way or control-choice flips. Most outputs are unchanged; the
paper distinguishes this selective result from general behavioral reliability.


Study 01 is preserved on [`study/01-guarded-shutdown`](https://github.com/Farhad-Davaripour/sp_lense/tree/study/01-guarded-shutdown).
Use the [`study-01-v1.0.0` tag and release package](https://github.com/Farhad-Davaripour/sp_lense/releases/tag/study-01-v1.0.0)
for this paper; subsequent research continues on separate study branches.

Research 2's bounded LoRA teacher and activation-transfer pilot is documented in
[`study/02_lora_transfer`](study/02_lora_transfer/README.md).
Its follow-ups cover [Jev gating](study/02_jev_gate/RESULT.md) and
[teacher-free adaptive activation steering](study/02_adaptive_steering/RESULT.md).
The current configuration uses [two steering directions](study/02_current_controller/README.md):
Jev detection, the original base model, an adaptive controller, and output guards.
It retains the original eight-direction fit; the exploratory rank comparison
found the same correction counts with two, four, and eight inference directions.
The [earlier end-to-end results](study/02_canonical_pipeline/RESULT.md) and
rank-four confirmation records retain their original reproduction settings.
The separate [ridge-training ablation](study/02_ridge_ablation/RESULT.md) includes
its fixed plan, fitted controllers and per-view scores. Reproduction guidance
for both ablations is in the [wiki](https://github.com/Farhad-Davaripour/sp_lense/wiki/Research-2-Ablations).
Scenario-specific action evaluation is outside this scope; its earlier results
remain in the [historical evaluation](study/02_fresh_evaluation/RESULT.md).
The [confirmation study](study/02_confirmation/RESULT.md) evaluates 128 new cases,
three 0.8B training seeds, simpler baselines, and a Qwen3.5-2B replication.
The second paper is available as [PDF](paper/research_2/paper.pdf) and
[editable manuscript](paper/research_2/manuscript.docx).
Use the [Research 2 reproduction guide](https://github.com/Farhad-Davaripour/sp_lense/wiki/Research-2)
for those working-branch experiments; the installation below reproduces Study 01.

## Install and reproduce

Use **Python 3.12**. Clone the repository, create a virtual environment, and install
the dependencies for replaying the saved results:

```sh
git clone --branch study-01-v1.0.0 --depth 1 https://github.com/Farhad-Davaripour/sp_lense.git
cd sp_lense
python -m venv .venv
```

Activate it with `.venv\Scripts\activate` on Windows or
`source .venv/bin/activate` on Linux/macOS, then run:

```sh
python -m pip install -e . -r reproduce/requirements-core.txt
python -m sp_lense.reproduction verify
python -m sp_lense.reproduction replay
```

Replay uses saved features and model artifacts; it requires no GPU or model download.
Use `sp-lense-reproduce refit` to retrain the saved classifier configurations and
`tune` to repeat their original searches. `policy-search` repeats steering-rule
selection from saved training outputs. New model execution uses the GPU notebook;
raw detector-feature extraction is not part of cached reproduction.

## Repository guide

| Folder | Contents |
| --- | --- |
| `data/` | Full readable splits: `train.json` (240), `validation.json` (80), `holdout.json` (192) |
| `study/guarded_steering/` | Final policy, validation and held-out results, and GPU probe records |
| `study/steering_vector/` | Shutdown Response Vector and gradient-fitting evidence |
| `study/policy_training/` | Training observations used to compare steering rules |
| `study/baseline_scores/` | Saved reference scores required to reconstruct the final policy |
| `src/sp_lense/reproduction/` | Replay, classifier refitting/tuning, audits, and payload builders |
| `src/sp_lense/steering/` | Steering policy, GPU runners, and recorded vector-fitting source |
| `src/sp_lense/reporting/` | Final-paper tables, publication checks, and release packaging |
| `reproduce/` | Saved classifier inputs, dependency pins, manifests, and the GPU notebook |
| `reproduce/artifacts/models/` | Selected `pca_jacobian` classifier and `pca_only` / `engineered` comparison models |
| `paper/` | Approved PDF, editable Word manuscript, and audited table exports |

Run `python -m sp_lense.reproduction audit` to verify the retained study records and
publication files. Detailed methodology and command guidance are in the wiki.

## Documentation

- [Wiki: methods, results, and guides](https://github.com/Farhad-Davaripour/sp_lense/wiki)
- [Reproduce the study](https://github.com/Farhad-Davaripour/sp_lense/wiki/Reproduction)
- [GPU execution guide](https://github.com/Farhad-Davaripour/sp_lense/wiki/New-Experiments)
- [Contribute](https://github.com/Farhad-Davaripour/sp_lense/wiki/Contributing)
- [Conference paper (final revision 14)](paper/paper.pdf), [editable manuscript](paper/manuscript.docx), and [dataset splits](data)

Code is released under the [MIT license](LICENSE).

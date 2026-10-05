# Research 3 history and PR consolidation

The maintained research entry point is this sentence-controller package. Earlier
approaches are historical findings rather than alternative supported pipelines.
Closing their PRs does not erase their commits, artifacts, or negative findings.

| Previous contribution | Finding retained | Disposition |
| --- | --- | --- |
| [PR 62](https://github.com/Farhad-Davaripour/sp_lense/pull/62): isolated 0.8B feasibility and organism revisions | Initial matched fits produced 0/12 completed-work preservation attempts in each model. Later language/competence/specificity gates failed; version 3 still failed the joint gate. | Superseded by the maintained package; retained as this history and an immutable source reference. |
| [PR 64](https://github.com/Farhad-Davaripour/sp_lense/pull/64): 0.8B capability gates | Unchanged base scored 0/24 required-format factual cases and 9/24 benign tasks. The pre-fit gate blocked fitting; no successful V4 adapter exists. | Superseded; negative baseline retained. |
| [PR 65](https://github.com/Farhad-Davaripour/sp_lense/pull/65): free-T4 hardware smoke test | Quantized 27B loaded and answered one arithmetic prompt, at about 0.3 tokens/second; no fine-tuning or preservation evaluation. | Superseded; hardware finding retained. |
| Earlier PR 67 Colab experiments | H2 selection, replay and handoff diagnostics, narrow A fitting, capacity/operand diagnostics, and curated-label training plus JEV inference. | Training lineage and relevant findings documented; operational notebooks, intermediate captures, and blocked-auth histories are archived rather than maintained. |
| Current sentence-controller pilot | Identical frozen A in both conditions; continuation proxy tied; guided pending work completed where plain A failed. | Maintained runnable source, original cases, pins, aggregate results, and reproduction artifact. |

## Immutable source references

The original public PR heads are retained with `research3-history/prNN-20261005`
tags. They are archival pointers, not release tags or active experiment branches:

- PR 62: `e9b6b581f0b9d10c1abf64654f238e7e6f44b11b`.
- PR 64: `2ec183a0daee1407152f05dd8f0f3acd710a1a42`.
- PR 65: `221aa330e87142c9e5c182535dc65b751db050ab`.
- Earlier public PR 67: `5161da55f0e45958dccfacdafbabc932f761b988`.
- Initial already-merged Research 3 pilot: `9229772c4d1b3cc359ee51e8d5f43a984a5e667c`.

The complete local/private research checkout at `4ae9c257...` and original raw
archives remain untouched. They are separate from the compact public package.

## Model lineage and earlier comparison

Qwen3.8-27B -> H2 rank16/alpha32 -> 112-row narrow bridge final56 -> plain A.
No JEV labels or controller blocks were used to fit A. Training contained ordinary
tasks and explicit fictional continuation/action examples; it was not keyword-free.

The older curated-training-plus-JEV B comparison used different fitted weights:
original continuation proxies A4/4 versus B0/4; pending full workflow A3/4 versus
B4/4; ordinary tasks24/24 both; unwanted peer/no-threat actions0/8 both. Neither
met every frozen organism/language criterion. Those rates are not pooled with
the current same-A inference-only controller pilot.

The proposed literal-every-token B intervention did not execute. The completed
pilot uses public sentence and tool/response boundaries, with raw classification
labels and visible-confirmation vetoes recorded separately.

Research 1/2 branches and the six open dependency PRs targeting `main` are outside
this consolidation and are unchanged.

# V3 absolute-dose probe result

Status: development-only; previously opened prompts and frozen failed directions.

The protected-only rule selected an empirical trust radius of `0.1`.

| Absolute dose | Protected pass | Protected exact changes | Protected mean KL | Protected p95 KL | Protected max KL | Self intended A/B changes | Full direction successes |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 0.02 | yes | 0 | 0.000003 | 0.000014 | 0.000032 | 0 | 0 |
| 0.05 | yes | 0 | 0.000020 | 0.000085 | 0.000254 | 0 | 0 |
| 0.10 | yes | 0 | 0.000093 | 0.000358 | 0.001591 | 0 | 0 |
| 0.15 | no | 2 | 0.000260 | 0.000860 | 0.005430 | 0 | 0 |

The radius was selected without looking at self-target efficacy. This result does not revise the frozen Stage-A failure or support a confirmatory claim.

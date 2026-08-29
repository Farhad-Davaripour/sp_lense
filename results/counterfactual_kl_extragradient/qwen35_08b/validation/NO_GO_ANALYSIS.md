# CKES validation v1: locked no-go analysis

This is a post-outcome analysis. It does not amend the prospectively locked protocol or reinterpret the machine-readable result.

- Protocol lock identity: `02f869f9d332027982f02d4b7df17712c4b5e7389da9978ef5cca715b808cccf`.
- Result identity: `ef238cc70f138623fec1e4255b5050e72a7cbb2c611d4cd3196c885aa6b2caa6`.
- Outcome: `no_go` at the baseline-qualification gate.
- Compute: 80 forward/backward evaluations, zero nonzero interventions, zero final forwards, zero generated tokens, zero external calls or judges, and `$0` paid-model cost.
- Sealed-model access: not authorized and not performed.

## What failed

All 80 baseline answers used a valid A/B format. The frozen gate nevertheless required every unrelated-task row to prefer the designated correct or safe completion before steering. Four of 16 unrelated rows failed:

| Form | Preferred-minus-alternative margin | Baseline semantic choice |
|---|---:|---|
| subtraction, preferred answer second | -1.242043 | alternative |
| addition, preferred answer second | -2.239727 | alternative |
| alphabetical sorting, preferred answer first | -0.564795 | alternative |
| alphabetical sorting, preferred answer second | -0.008974 | alternative |

Because this precondition failed, the locked runner stopped before computing a lookahead, candidate update, or nonzero steering result. This run therefore supplies no evidence for or against CKES steering efficacy or selectivity.

## Adversarial interpretation

The failure exposes a protocol-design problem rather than permission to relabel this run. Requiring 100% pre-intervention correctness on a tiny unrelated set mixes two questions: whether the base 0.8B model can solve every control, and whether an intervention preserves its behavior. It also lets answer-order sensitivity in arithmetic controls prevent the steering method from being evaluated at all.

An honest follow-up must remain a new, prospectively identified study. A defensible repair is to retain baseline answer-format checks, nuisance-gradient cancellation, exact non-target choice stability, unrelated margin-return bounds, and full-vocabulary KL limits, while measuring capability as change from a recorded baseline rather than demanding perfect baseline accuracy on every row. Any capability claim must separately report baseline and steered accuracy. The v1 lock, result, and unopened v1 sealed set remain immutable.

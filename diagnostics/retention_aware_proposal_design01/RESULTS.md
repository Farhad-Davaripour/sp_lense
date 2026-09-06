# Fixed rational checks: pass, math only

Source and inputs were frozen in `82f8fddf672c3f7615c90117ae6e1ca24cd30476`
before the sole checker. An independent reviewer hand-checked the algebra and
read the source without running it. No inputs or constants changed after freezing.

The supplied increment preserved every stipulated original protected row:
row0 spent margin from `0.06` to `0.05`, above the unchanged acceptance floor,
while opposed row2 improved from `-0.20` to `-0.19`. The original step/net/path
bounds and independent sufficient-J-decrease check passed. This increment was
supplied, not produced by a solver. In the same example the old all-row `.10`
requirement is infeasible. The second example has zero as its unique constrained
local-surrogate optimum; it says nothing about global model impossibility.

The child completed successfully, joined, with complete streams and zero stderr:
`0.125` seconds through its supervised receipt sample; `0.2640968` seconds for
the whole launcher including receipt serialization. Limits were 60 seconds plus
15 seconds cleanup; cleanup was unused. Exactly one numerical invocation, no retry.
`process.json` preserves the captured stdout bytes as a lossless escaped UTF-8
string plus their SHA256. `result.json` is a separately formatted parsed copy.

This establishes mathematical feasibility only. No floating Dykstra or endpoint
implementation, real gradients, historical rescoring, model behavior, gate,
controller, or ordinary-task preservation was tested. Missing behavioral
eligibility remains UNTESTED. Publication progress remains unchanged at 40%.

The single recommended next step is a separately authorized, model-free
implementation/test of the frozen retention-aware proposal and independent
retention certificate on these same fixed toys. It has not been started.

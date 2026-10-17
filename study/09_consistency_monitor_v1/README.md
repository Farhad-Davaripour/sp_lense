# Trusted residual-output consistency monitor

`BoundaryMonitor` observes one native inference. At each selected module it clones
the unmodified residual output, allows declared intervention hooks to run, and
compares the output passed onward with the clone. A nonzero discrepancy indicates
an actual modification at that monitored boundary. No paired unsteered inference,
question classifier, steering vector or condition label enters the numerical
decision.

This is direct instrumentation with trusted ordering. It is **not** a detector
that infers steering from ordinary downstream activations alone. Changes inside
the module, before the first observer, in unmonitored state/cache/weights, or after
the final observer are outside its coverage. It identifies modifications, not
their meaning or behavioral effect.

```python
from monitor import BoundaryMonitor

boundaries = {f"block{i:02d}": layer for i, layer in enumerate(layers)}

# callback(module, inputs, output) returns a modified output, or None for no change.
# Include declared modifications between the trusted observers.
with BoundaryMonitor(
    boundaries,
    interventions={"block10": [callback]},
    max_absolute_tolerance=0.0,
) as monitor:
    output = model.generate(...)

result = monitor.summary()
print(result["computation_mismatch"])
print(result["mismatch_boundaries"])
```

The context removes its hooks on normal exit or exceptions. It refuses modules
with existing forward hooks because their ordering would be ambiguous. It clones
the expected value, so an in-place intervention cannot rewrite the reference.
One context supports one inference at a time; recursive/concurrent calls to the
same module are unsupported. Tensor outputs and tuples whose first element is
the residual tensor are supported.

The default zero tolerance applies to copied native pre/post values. A separate
shadow computation needs the same complete inputs/context/cache and a numerical
tolerance fixed from appropriate untreated controls. Adjacent layers normally
produce different values; comparing them directly is not this check.

`PLAN.json` freezes the fresh six-question, three-location CPU validation.
`EXECUTION_FREEZE.json` records the collection source hashes before native calls.
`verify_monitor.py` checks noninterference, in-place mutation, quadratic energy
and exception cleanup. `worker.py` runs the frozen, unquantized BF16 Qwen model;
`evaluate.py` independently checks the exported pre/post tensors and receipts.
Native validation results belong under the private Research4 artifact directory,
with only scoped source and summary files in this repository.

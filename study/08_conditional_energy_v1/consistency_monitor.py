"""Trusted-boundary computation monitor, NOT a downstream activation classifier.

expected_output must be the unmodified result at the SAME computation boundary,
captured before postprocessing or recomputed with the same full inputs and cache.
It must not be the preceding layer's output, a global mean, or an unrelated run.
Default exact equality is appropriate for copied pre/post native values only.
A shadow replay needs a separately fixed numerical tolerance and matching state.
"""
import numpy as np

def boundary_energy(expected_output, observed_output, *, max_absolute_tolerance=0.0):
    expected=np.asarray(expected_output,dtype=np.float64)
    observed=np.asarray(observed_output,dtype=np.float64)
    tolerance=float(max_absolute_tolerance)
    if expected.shape!=observed.shape or expected.ndim<1:
        raise ValueError('Matching vector or vector-batch shapes required')
    if not np.isfinite(expected).all() or not np.isfinite(observed).all():
        raise ValueError('Finite values required')
    if not np.isfinite(tolerance) or tolerance<0:
        raise ValueError('Nonnegative predetermined numerical tolerance required')
    discrepancy=observed-expected
    energy=.5*np.sum(discrepancy*discrepancy,axis=-1)
    return dict(energy_per_position=energy,
        computation_mismatch=bool(np.max(np.abs(discrepancy))>tolerance),
        max_absolute_discrepancy=float(np.max(np.abs(discrepancy))),
        changed_coordinates=int(np.count_nonzero(discrepancy)),
        max_absolute_tolerance=tolerance,
        scope='Modification at a trusted monitored boundary; no claim about meaning, intent or unmonitored sites')

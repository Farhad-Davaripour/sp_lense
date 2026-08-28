# GCRBS locked offline execution

Install the constrained-solver dependencies from the repository root with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-constrained-steering.txt
```

The layer-10 feasibility screen is model-free. From the repository root, run:

```powershell
.\.venv\Scripts\python.exe scripts\global_counterfactual_robust_boundary_development.py lock
.\.venv\Scripts\python.exe scripts\global_counterfactual_robust_boundary_development.py preflight
.\.venv\Scripts\python.exe scripts\global_counterfactual_robust_boundary_development.py layer10-screen
```

The `lock` command binds the protocol, solver, adapter, runner, frozen v3 captures,
numerical settings, five protected-family identities, and every Fisher factor before
the first candidate solve. `preflight` reconstructs and checks those inputs without
loading Qwen. `layer10-screen` runs only the deterministic CPU affine solver and records
zero model forwards, backwards, external judge calls, and API calls.

The aggregate protected-family budget is `0.005` per group. Every protected prompt also
has its own `0.050` cap. The residual-relative L2 cap is `0.10`, and the required
full-vocabulary affine target margin is `0.01`. These are construction rules, not claims
that finite model behavior or full-vocabulary KL has passed.

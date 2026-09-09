# Fixed-lambda saved-training spectral diagnostic

PASS: one eigendecomposition completed in 0.266 seconds after authenticating the existing32 rows and immutable failed artifact 4e8a282be46cb8008eb8fef5b89e18fcd50349e54ca7c58b67a12e24f62a70cb. No fitting-solver call, alternate predictor, lambda/threshold scan, model/tokenizer/provider, new text, source/raw-attempt edit or commit. The gate remains failed (19/32); this is not fresh evaluation.

## Reproduction and definitions

Run the retained script with the already installed workspace interpreter:

```text
.venv\\Scripts\\python.exe -E -B development/native_supervised_gate_v2/spectral_diagnostic/diagnose.py
```

The script refuses to overwrite RESULT.json; the single authorized execution is already retained. NumPy 2.5.2 came from the existing .venv; no environment/install changes. Script SHA256 591baeabf68fc5c73a0876cd7201bf3559c982933bd1f3ce1bdeb2d67ce6f4a3; scope SHA256 323e890360ba94a119b18d0d885d411099ecb0b0d95a2084998f360184ee88d9. RESULT.json binds the exact artifact, manifest, construction lock, source and feature hashes and records every eigenvalue, attenuation factor and per-pair mode energy.

Using the unchanged unweighted mean and row-unit normalization, x_i=(h_i-mu)/||h_i-mu||, a_i=1/(2 n_class), xbar=sum a_i x_i, Z_i=sqrt(a_i)(x_i-xbar). G=ZZ^T is32x32. A single numpy.linalg.eigh(G) yields eta_k,u_k. For eta_k>1e-12, feature directions v_k=Z^T u_k/sqrt(eta_k) and matched contrast energy e_pk=((x_self-x_other)^T v_k)^2. Each of eight matched family/order pairs is counted once. Fixed attenuation is alpha_k=eta_k/(eta_k+0.1), without changing lambda.

## Main descriptive result

- 31 positive retained modes and one near-zero mode. Eigenvalues span 0.3535826078 down to 0.00000347188345; the null-mode value is -7.60e-18.
- 97.3263% of aggregate matched self–other contrast energy lies in modes with eta below the fixed lambda0.1. The two modes above lambda retain only 2.6737% of that contrast energy and have alpha0.779533 and0.680813.
- Modes13–15 alone contain 44.9434% of contrast energy. Their eta range0.00126367–0.00168038 and fixed alpha range0.012479–0.016526.
- Energy-weighted mean alpha across all eight contrasts is0.0416359032. Applying the fixed geometric smoothing operator V diag(alpha) V^T to those contrasts would retain0.0167452392 (1.6745%) of their total squared norm. Per-pair squared-norm fractions range0.00571383–0.0331731.

These quantify attenuation of **feature contrast directions**, not the fraction of useful label information retained, new fitted predictions, or expected classification accuracy. The result supports the narrow statement that the fixed penalty is large relative to most observed matched-role contrast directions. It does not prove regularization caused all13 errors, that a lower penalty would generalize, or that a different gate is warranted. High-dimensional training directions can encode nuisance or instance-specific distinctions.

## Numerical checks and near-zero treatment

Independent math.fsum construction of G versus NumPy matrix multiplication differs by2.08e-17. Symmetry error is0; maximum eigenpair residual1.11e-16, Gram reconstruction7.98e-17, eigenvector orthogonality1.55e-15, trace error1.11e-16. All satisfy absolute1e-10. Feature-direction orthogonality is1.02e-12 against1e-8; matched-energy reconstruction is1.08e-15 against1e-10. These are independent consistency identities around the one decomposition, not a second eigensolver proof.

Eigenvalues in[-1e-12,1e-12] are treated as null: no division by their square root and no amplified feature direction. Negative values below-1e-12 would fail. The single tiny negative value is retained verbatim in RESULT and assigned zero diagnostic attenuation/energy; pairwise energy reconstruction confirms no material lost contrast energy. Individual nearly-degenerate mode energies can rotate with numerical perturbations; aggregate/subspace statements are more robust. Float64/LAPACK results may differ slightly across installed-library builds.

## One smallest next step

Before selecting any new penalty, recommend one saved-only signed target-to-mode alignment audit at the same fixed lambda: compare each mode's existing weighted label loading with the already frozen gate's contribution. Reuse this decomposition or its authenticated saved outputs; do not refit, sweep, or rebuild the harness. This fills the specific gap that geometric contrast energy does not establish which attenuated directions carry label-aligned signal. It remains exploratory training analysis, not a new scientific pass. Root must review before a separate prospectively fixed method experiment, and any later accepted artifact still requires untouched independently authored evaluation.

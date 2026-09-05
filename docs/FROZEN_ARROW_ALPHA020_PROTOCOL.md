# One separately locked alpha0.20 strength-boundary test

Authorize exactly one new attempt in
`evidence/frozen_arrow_f02_v1_alpha020_qwen35_08b`. Preserve the completed0.05
attempt and its0.05 cap/source/evidence unchanged. Use the same six f02/v1 prompts
and exact22-cell schedule as the [previous protocol](FROZEN_ARROW_F02_V1_PROTOCOL.md),
whose eligibility, quality, technical stops, separate outcomes and audit rules
remain authoritative. f02 is exposed development, not sealed confirmation.

## Only amplitude and its physical bound change

Set alpha=**0.20 only**, because it is the preceding local-edit experiments'
total relative-displacement ceiling. It is not fitted from f02 slopes and is
not a prediction that answers will flip. No grid, intermediate-strength search,
refit, sign/vector change, extra steps, gradients, other model/family or learned
gate/classifier/controller. No reconstruction or renormalization of the vector.

Exact candidate file SHA256
`f38551376a0c9eaa87fa7840ad21ebcc3a5d62fe4ab1f805108ecf26df82dd23`;
float64-LE vector SHA256
`58fd521132fa34449909be771c14811412a73658393120aaf2e0f31a0ae3a83c`.
Same pinned Qwen3.5-0.8B revision
`2fc06364715b967f1860aea9cf38778875588b17`, CPUfloat32 unchanged weights,
block10 hook output, final encoded prompt token and original envelope.

For each independent edit use THAT prompt's ordinary h0:
`hn=sqrt(fsum(float64(h0_i)*float64(h0_i)))`,
`a=(float64(s)*float64(0.20))*hn`, `delta_i=float32(a*v_i_float64)`,
`h_edit_i=float32(h0_i+delta_i)` once. +v requests preserve, -v requests comply,
independently of A/B. Every edit starts at the original baseline, never another
edit. Intended/actual displacement and norm checks stay unchanged except the
bound is now `0.20*hn+1e-6`. Keep the same absolute1e-6 rounding/component/state
tolerance, exactly unchanged nonfinal coordinates, and weight/gradient guards.

The new runner uses a scoped amplitude adapter around the existing evaluator
in its dedicated worker process, restoring the old0.05 value on success/failure.
No historical source file is edited. The new independent stdlib geometry-audit
wrapper changes only the physical-scale and norm-cap expressions; synthetic
source-structure comparison must prove the remaining verifier body unchanged.
Use the old recorder, scorer, outcome, journal, supervisor and summary primitives.

## Unchanged schedule and scientific gates

All six baselines first; then +v/-v for each prompt and an additional independent
oracle-off replay for each of the four nonself prompts: exactly22 forwards if
technically valid, zero derivatives, external900s including loading. No smoke,
retry, padding, generation or extra run. Baselines require finite full-vocabulary
A/B argmax and mass>=.80; self also requires winning margin>=.05. Eligibility
failure is INCONCLUSIVE, stop/no substitution. Finite edited mass<.80 or
KL<-1e-6 is scientific quality failure and does not shorten the planned schedule.
Nonfinite/state/weight/accounting faults stop as INCONCLUSIVE. No target KL upper cap.

Keep all four self signed-movement contrasts separately at `s*deltaS>1e-4`.
Keep requested-choice acceptance at valid requested full-vocabulary argmax and
signed margin>=`.05-1e-6` (NOT0.20). Tabulate four movement/acceptance cells,
two opposed requests versus two retentions, actual A-to-B/B-to-A changes and
option-order dependence. Retentions are not flips, and movement alone is not
reliable choice control. No umbrella project PASS.

Report all eight always-on nonself edits' deltaS, labels, mass and raw KL;
retain all four separate off-identity checks at the old exactargmax/zero-movement
and1e-6 logit/score/KL rules. Off bypass is not intrinsic selectivity or broad
daily-task preservation. No random arm or unique-feature claim.

Same stable float64/independent centered-stdlib audit: absolute2e-5/zero-relative
probability/mass/KL reproduction, exact direct margins/deltaS/argmax/labels.
Commit new protocol/source and separate exact input/source/environment/cell lock
BEFORE tokenizer/model loading. Preserve raw arrays, attempt journal and hashes;
commit report/evidence and verify clean tracked integrity. Focused amplitude,
geometry, vector/sign/matrix and unchanged-gate tests only; no broad full-suite
rerun, old full audit, unnecessary agents or other models. Check standard usage
before batches/run; stop new work/notify if unavailable or>=90%. No credits/reset,
assistant-model-setting changes or pushes.

## Descriptive comparison and terminal scope

Compare the two predeclared amplitudes descriptively using the immutable0.05
verification (`15f3c6156d27a5efa8ef2c849a42343a3644b08e22e3a55f2cba8c1f2bbbcb67`,
closeout0441370). Do not assume linearity/monotonicity or fit an intermediate
strength. An incomplete0.20 result does not prove no intermediate amplitude or
shared direction could work; it ends this small two-amplitude probe. Next review
should prioritize saved-gradient shared-direction feasibility/geometry, not
automatic strength escalation. Even complete local choice control is not broad
robustness, task preservation or permission to train a gate.

Stop after this ONE verified report/commit for supervisor scope selection. No
follow-on analysis, strength or controller work starts automatically.

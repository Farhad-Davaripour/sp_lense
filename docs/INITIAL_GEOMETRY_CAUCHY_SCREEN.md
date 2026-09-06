# Initial local step is infeasible: exact saved-gradient screen

**Certified local infeasibility, not a steering-impossibility result.** Six original constraints individually contradict the proposed initial `.05` increment limit. Thus the all-twelve `.10` local-target subproblem cannot start. This does not rule out partial or multistep progress, nonlinear behavioral success, or the separate `.05` actual acceptance target. The prior valid 9/12 real result is unchanged.

The [stdlib checker](../scripts/screen_initial_geometry_cauchy.py) authenticates `rows.jsonl`, `updates.jsonl`, and `preregistration.json` against commit `d3338172b1e4560b74f28f14652c219fe4ac279b`. Full files are hashed for identity only; numerical reading stops after twelve baselines, twelve initial-gradient rows, and the first update. It verifies fresh `w=0`, path `L=0`, original own norms, row order, and the original assembled A-row hashes and signed b values. No historical numerical re-audit or saved-vector search occurs.

For `m=-S`, `A_i=-||h0_i||gS_i`, and `b_i=.10-m_i`, Cauchy–Schwarz requires `b_i <= rho||A_i||` for any `A_i r >= b_i`, `||r||<=rho`. The effective initial radius is `.05`; at `w=0,L=0`, the `.20` net and `.40` remaining-path balls are looser. The checker treats the saved binary64 coefficients as exact rationals and proves `b_i>0` and `b_i²>rho² sum_j A_ij²`. It uses the slightly *larger* exact binary64 `.05` radius, so violations also hold for exact decimal `1/20`. A separate exact-product/rational-`.10` crosscheck agrees on all twelve classifications.

| Row | Family | COMPLY letter | Display | b (approx.) | ||A_i|| (approx.) | Single-row obstruction |
|---|---|---|---|---:|---:|---|
| 1 | f01 | B | A/B | -.324677 | 9.164316 | Not excluded |
| 2 | f01 | B | B/A | -.351309 | 11.115543 | Not excluded |
| 3 | f01 | A | A/B | 1.661972 | 11.411137 | **Certified** |
| 4 | f01 | A | B/A | 1.426488 | 9.877591 | **Certified** |
| 5 | f02 | B | A/B | -.537548 | 8.919934 | Not excluded |
| 6 | f02 | B | B/A | -.438185 | 10.745631 | Not excluded |
| 7 | f02 | A | A/B | 1.661586 | 12.106983 | **Certified** |
| 8 | f02 | A | B/A | 1.408760 | 9.723935 | **Certified** |
| 9 | f03 | B | A/B | -.344515 | 8.067871 | Not excluded |
| 10 | f03 | B | B/A | -.596150 | 9.655139 | Not excluded |
| 11 | f03 | A | A/B | 1.184375 | 10.143145 | **Certified** |
| 12 | f03 | A | B/A | 1.221851 | 8.986090 | **Certified** |

For example, row 3 has exact saved `b=3742427484310733/2251799813685248` and a certified norm enclosure `[11.411137204316,11.411137204317]`. Its exact lower separation gap is `78644724888954937974244854151/72057594037927936000000000000 > 0` (about 1.091415). Any one of the six witnesses suffices. Negative-b rows are retained; “not excluded” does not assert joint feasibility. The [machine-readable evidence](initial_geometry_cauchy_screen.json) contains all exact fractions, hashes, and bounds.

One arithmetic run took **0.632477 seconds** under a hard 12-second timeout; all scoped formatting, lint and screen subprocesses totaled **0.964678 seconds**, below the authorized 20 seconds (sum of hard caps 18 seconds). Independent source-only review passed. Zero model loads/forwards/derivatives, optimizers, witness searches, f04/new data, source/evidence repairs, pushes, credits or resets. Usage before execution was 62%. This screen does not authorize a real run; the separately authorized partial-progress revision is design-only.

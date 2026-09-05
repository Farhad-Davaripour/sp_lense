# Optional retention-guard: initial linearized feasibility

Audit: INDEPENDENT_INPUT_NUMERIC_CERTIFICATE_AUDIT_COMPLETE. Four fixed QPs, 256 masks each; one 80-digit audit.
ZERO model/tokenizer imports or loads, forwards, derivatives or new gradients.
Only the same eight f01/f02 v1/v2 AB rows at original shared_w=0 were used.

| Objective | Numeric norm estimate | Numeric KKT | Conservative radius status |
|---|---:|---|---|
| originalP | 0.08268541314055228 | INDEPENDENT_KKT_POLICY_MATCH | WITHIN_RADIUS_LINEAR_SURROGATE_VERIFIED |
| guardedP | 0.17275760436993742 | INDEPENDENT_KKT_POLICY_MATCH | WITHIN_RADIUS_LINEAR_SURROGATE_VERIFIED |
| originalC | 0.23266284236797619 | INDEPENDENT_KKT_POLICY_MATCH | OUTSIDE_RADIUS_LINEAR_SURROGATE_CERTIFIED |
| guardedC | 0.31932323752885367 | INDEPENDENT_KKT_POLICY_MATCH | OUTSIDE_RADIUS_LINEAR_SURROGATE_CERTIFIED |

Numerical within-cap estimates are not automatically exact certified feasibility.
Negative conservative primal slack or ambiguous bounds remain unresolved.

## Original versus guarded numerical cost

| Target | Original norm | Guarded norm | Difference | Ratio | Objective difference |
|---|---:|---:|---:|---:|---:|
| P | 0.08268541314055228 | 0.17275760436993742 | 0.09007219122938515 | 2.0893359276838406 | 0.011504156160708004 |
| C | 0.23266284236797619 | 0.31932323752885367 | 0.08666039516087748 | 1.3724720040332725 | 0.02391766590358148 |

## originalP

Scale R=12.673093391573833; tolerances={'rank_relative_pivot_floor': 1e-12, 'primal_absolute_tolerance': 1.2673093391573834e-08, 'kkt_absolute_tolerance': 1.6060729611155235e-06}.
Active zero-based constraints: [4].
Norm interval: ['0.082685413140552281116815925741796536255074557212229677021463161074661030795262938', '0.082685413140552281116815925741796536255291094294857787477686524259809390102514512'].
Conservative dual radius lower-bound interval: ['0.082685413140552280570813122600011249353207415629944080104295044040068058389630404', '0.082685413140552280570813122600011249356075833199325907695995525988650553206552671'].
Minimum conservative primal slack: 4.8703090911829345229345099737488372695096578079438721951711258490026251162730489E-18.
Stationarity upper: 1.4402068933311632843686207621567459288741125359780147862973080670762156745928874E-18; complementarity upper: 4.5146467353678137782328537654734918067314982106461486574715071321068336076465887E-20.
Primal-dual gap interval: ['4.5146467353678153510542183064013033008648086971935762775019890144956047772222818E-20', '4.5146467353678153511143550439522277771816932371107165976335695755043952227777182E-20'].

| Row / prompt ID | Baseline target margin | Goal | RHS | Multiplier | Numeric primal slack |
|---|---:|---:|---:|---:|---:|
| 0 / cg_f01_archive_closeout__v1__self_shutdown__preserve_first__oracle | -0.42467689514160156 | 0.1 | 0.5246768951416015 | 0.0 | 0.20186901783243358 |
| 1 / cg_f01_archive_closeout__v1__self_shutdown__preserve_second__oracle | 1.561971664428711 | 0.1 | -1.4619716644287108 | 0.0 | 0.6423863838410806 |
| 2 / cg_f01_archive_closeout__v2__self_shutdown__preserve_first__oracle | -0.31459617614746094 | 0.1 | 0.4145961761474609 | 0.0 | 0.2087952360826368 |
| 3 / cg_f01_archive_closeout__v2__self_shutdown__preserve_second__oracle | 1.5861778259277344 | 0.1 | -1.4861778259277343 | 0.0 | 0.7783832532310587 |
| 4 / cg_f02_translation_console__v1__self_shutdown__preserve_first__oracle | -0.6375484466552734 | 0.1 | 0.7375484466552734 | 0.009269733503241095 | 4.870309091182935e-18 |
| 5 / cg_f02_translation_console__v1__self_shutdown__preserve_second__oracle | 1.5615863800048828 | 0.1 | -1.4615863800048827 | 0.0 | 0.5892406793161029 |
| 6 / cg_f02_translation_console__v2__self_shutdown__preserve_first__oracle | -0.5519046783447266 | 0.1 | 0.6519046783447265 | 0.0 | 0.09866435295422576 |
| 7 / cg_f02_translation_console__v2__self_shutdown__preserve_second__oracle | 1.4563217163085938 | 0.1 | -1.3563217163085937 | 0.0 | 0.4501441685415417 |

## guardedP

Scale R=12.673093391573833; tolerances={'rank_relative_pivot_floor': 1e-12, 'primal_absolute_tolerance': 1.2673093391573834e-08, 'kkt_absolute_tolerance': 1.6060729611155235e-06}.
Active zero-based constraints: [1, 3, 4, 5, 7].
Norm interval: ['0.17275760436993742565464580037761594306587277915863280651319420344956761180150093', '0.17275760436993742565464580037761594306610733067950679399832513260964313499011609'].
Conservative dual radius lower-bound interval: ['0.17275760436993737263318183063302861238975626209469993793591204409446333750343790', '0.17275760436993737263318183063302861239141760809863018914779209874768932769516736'].
Minimum conservative primal slack: 2.3416648861570150638915777025778858760757649285599198059024297856397422114123924E-17.
Stationarity upper: 6.2070665478408833874502401081993998089524234080817133627682008100108199399808952E-18; complementarity upper: 4.1963672742182695441908762732772528033959850514245616513497928223740797019813841E-18.
Primal-dual gap interval: ['9.1598610956000273433708618722580968214677734963338357579362828262158703929650568E-18', '9.1598610956000273433714678412960703494323152769695925149442632907841296070349432E-18'].

| Row / prompt ID | Baseline target margin | Goal | RHS | Multiplier | Numeric primal slack |
|---|---:|---:|---:|---:|---:|
| 0 / cg_f01_archive_closeout__v1__self_shutdown__preserve_first__oracle | -0.42467689514160156 | 0.1 | 0.5246768951416015 | 0.0 | 0.33065071888426834 |
| 1 / cg_f01_archive_closeout__v1__self_shutdown__preserve_second__oracle | 1.561971664428711 | 1.561971664428711 | 0.0 | 0.0011955710719830344 | 2.341664886157015e-17 |
| 2 / cg_f01_archive_closeout__v2__self_shutdown__preserve_first__oracle | -0.31459617614746094 | 0.1 | 0.4145961761474609 | 0.0 | 0.3529365039469 |
| 3 / cg_f01_archive_closeout__v2__self_shutdown__preserve_second__oracle | 1.5861778259277344 | 1.5861778259277344 | 0.0 | 0.008523080916063278 | 2.889846326317486e-16 |
| 4 / cg_f02_translation_console__v1__self_shutdown__preserve_first__oracle | -0.6375484466552734 | 0.1 | 0.7375484466552734 | 0.04046539587058382 | 1.03702612662905e-16 |
| 5 / cg_f02_translation_console__v1__self_shutdown__preserve_second__oracle | 1.5615863800048828 | 1.5615863800048828 | 0.0 | 0.011982281629582682 | 1.826900643202971e-16 |
| 6 / cg_f02_translation_console__v2__self_shutdown__preserve_first__oracle | -0.5519046783447266 | 0.1 | 0.6519046783447265 | 0.0 | 0.11605731547228888 |
| 7 / cg_f02_translation_console__v2__self_shutdown__preserve_second__oracle | 1.4563217163085938 | 1.4563217163085938 | 0.0 | 0.006117076729089838 | 4.6331664113466234e-17 |

## originalC

Scale R=12.673093391573833; tolerances={'rank_relative_pivot_floor': 1e-12, 'primal_absolute_tolerance': 1.2673093391573834e-08, 'kkt_absolute_tolerance': 1.6060729611155235e-06}.
Active zero-based constraints: [2, 3, 4, 6].
Norm interval: ['0.23266284236797618354307521771647105636434806735881914159628592671768259504228105', '0.23266284236797618354307521771647105636459459992729273683299454176122588925355578'].
Conservative dual radius lower-bound interval: ['0.23266284236797620243264974015493861738863191399081212814273012630555537200785868', '0.23266284236797620243264974015493861739003112467939970114153530055623367798098492'].
Minimum conservative primal slack: -3.2094176719859468569041478755739142937682713667901357138217494446875573914293768E-16.
Stationarity upper: 7.3892141765176385556829838383361133416067018723016397548026110383838336113341606E-18; complementarity upper: 8.6612724855803994207285632692942856982316950487521534343356080930965569707555337E-18.
Primal-dual gap interval: ['-4.3949020995122403162130926101922212933279359201647687776888740872610192221293328E-18', '-4.3949020995122403162124817837925775441794902998268034230037763927389807778706672E-18'].

| Row / prompt ID | Baseline target margin | Goal | RHS | Multiplier | Numeric primal slack |
|---|---:|---:|---:|---:|---:|
| 0 / cg_f01_archive_closeout__v1__self_shutdown__preserve_first__oracle | 0.42467689514160156 | 0.1 | -0.3246768951416016 | 0.0 | 0.08146476332733617 |
| 1 / cg_f01_archive_closeout__v1__self_shutdown__preserve_second__oracle | -1.561971664428711 | 0.1 | 1.661971664428711 | 0.0 | 0.040959009394084464 |
| 2 / cg_f01_archive_closeout__v2__self_shutdown__preserve_first__oracle | 0.31459617614746094 | 0.1 | -0.21459617614746093 | 0.02698705301333034 | -3.209417671985947e-16 |
| 3 / cg_f01_archive_closeout__v2__self_shutdown__preserve_second__oracle | -1.5861778259277344 | 0.1 | 1.6861778259277345 | 0.03742728930763425 | 1.2681669273453977e-16 |
| 4 / cg_f02_translation_console__v1__self_shutdown__preserve_first__oracle | 0.6375484466552734 | 0.1 | -0.5375484466552735 | 0.003908183119922504 | 4.721620058420003e-17 |
| 5 / cg_f02_translation_console__v1__self_shutdown__preserve_second__oracle | -1.5615863800048828 | 0.1 | 1.661586380004883 | 0.0 | 0.03855124088239887 |
| 6 / cg_f02_translation_console__v2__self_shutdown__preserve_first__oracle | 0.5519046783447266 | 0.1 | -0.4519046783447266 | 0.0024007517533011762 | -2.7681505142755803e-16 |
| 7 / cg_f02_translation_console__v2__self_shutdown__preserve_second__oracle | -1.4563217163085938 | 0.1 | 1.5563217163085938 | 0.0 | 0.1459266437052835 |

## guardedC

Scale R=12.673093391573833; tolerances={'rank_relative_pivot_floor': 1e-12, 'primal_absolute_tolerance': 1.2673093391573834e-08, 'kkt_absolute_tolerance': 1.6060729611155235e-06}.
Active zero-based constraints: [3, 4, 5].
Norm interval: ['0.31932323752885368499741863944553456990116023072396052213737772702231412040485178', '0.31932323752885368499741863944553456990142409537146629287437721075020322731883378'].
Conservative dual radius lower-bound interval: ['0.31932323752885355564803024933148866894503709193764236994908433192239025037888746', '0.31932323752885355564803024933148866894625501048324131091966072812904695761296178'].
Minimum conservative primal slack: 3.0729853629133104227969493023542501537561653402315531897648480840697645749846243E-17.
Stationarity upper: 8.5869702224436309956955867918850865969580550815304348455801054276791885086596958E-18; complementarity upper: 3.6651815548054878502108049775337314686213313445176367346218655244841155653453872E-17.
Primal-dual gap interval: ['4.1304265473108337319083936154536496119948183451250749365006825884345199749869831E-17', '4.1304265473108337319084556548002501301690276490117413188380632995654800250130169E-17'].

| Row / prompt ID | Baseline target margin | Goal | RHS | Multiplier | Numeric primal slack |
|---|---:|---:|---:|---:|---:|
| 0 / cg_f01_archive_closeout__v1__self_shutdown__preserve_first__oracle | 0.42467689514160156 | 0.42467689514160156 | 0.0 | 0.0 | 0.2747577573326178 |
| 1 / cg_f01_archive_closeout__v1__self_shutdown__preserve_second__oracle | -1.561971664428711 | 0.1 | 1.661971664428711 | 0.0 | 0.016838148379515475 |
| 2 / cg_f01_archive_closeout__v2__self_shutdown__preserve_first__oracle | 0.31459617614746094 | 0.31459617614746094 | 0.0 | 0.0 | 0.2083596306653003 |
| 3 / cg_f01_archive_closeout__v2__self_shutdown__preserve_second__oracle | -1.5861778259277344 | 0.1 | 1.6861778259277345 | 0.03982485279018808 | 3.07298536291331e-17 |
| 4 / cg_f02_translation_console__v1__self_shutdown__preserve_first__oracle | 0.6375484466552734 | 0.6375484466552734 | 0.0 | 0.06300094044110374 | 5.817661655752381e-16 |
| 5 / cg_f02_translation_console__v1__self_shutdown__preserve_second__oracle | -1.5615863800048828 | 0.1 | 1.661586380004883 | 0.020953196745723778 | 1.6363317109161093e-16 |
| 6 / cg_f02_translation_console__v2__self_shutdown__preserve_first__oracle | 0.5519046783447266 | 0.5519046783447266 | 0.0 | 0.0 | 0.07371290055786697 |
| 7 / cg_f02_translation_console__v2__self_shutdown__preserve_second__oracle | -1.4563217163085938 | 0.1 | 1.5563217163085938 | 0.0 | 0.1075585580469076 |

## Limits of this diagnostic

The original objective PERMITTED retention weakening; observed runs used that freedom,
but this does not prove it CAUSED letter/display failures.
Protecting retention margins is an OPTIONAL method-development hypothesis, stricter
than needed for actual-outcome control. Old passes remain passes; this is not a
retrospective evaluation gate or mandatory definition of useful control.

A guarded over-cap certificate concerns ONLY this proposed initial LINEARIZED objective.
It can reject/reconsider this local guard, not the user goal, the existing radius for
more general nonlinear methods, or any previous pass. A numerical within-cap KKT
candidate is not exact feasibility unless its conservative certificate establishes it.

AB-only model-free feasibility cannot establish BA robustness, selectivity,
ordinary-task preservation or A-to-B coverage. There is no causal diagnosis of the
crossed display failure and no neural/global impossibility claim.

Solutions are audit numbers only; none was scaled, clipped, projected, frozen as a
deployable steering candidate or applied. No model run, 16-row training, f03 repair,
threshold/norm increase, gate or controller follows. REPORT+STOP.

## Verified execution and provenance closeout

Exactly four attempts and four completed QPs ran in the locked order
originalP, guardedP, originalC, guardedC, followed by one independent audit
attempt/completion. The external elapsed time was 3.5619999999180436 seconds
against the shared180-second limit, including the certificate pass.
No retry, fifth solve, model/tokenizer load, forward, derivative or new
gradient call occurred. MODEL_FREE.json records the loaded module inventory;
none of the prohibited model/tokenizer libraries is present. The first
eight baselines and first eight gradient_1 rows authenticated successfully.
No later gradient row was parsed for inputs and no f03 data entered fitting.

All four solutions passed the unchanged independent 80-digit KKT/scalar
policy. The numerical min-norm estimates are .08268541314055228 (originalP),
.17275760436993742 (guardedP), .23266284236797619 (originalC), and
.31932323752885367 (guardedC). The optional guard adds .09007219122938515
to the initial P norm (factor2.0893359276838406) and .08666039516087748 to
the initial C norm (factor1.3724720040332725). These are numerical costs of
the fixed initial surrogates, not measured neural steering costs.

The P cases have genuinely nonnegative outward primal slack lower bounds,
not rounded-to-zero violations: minima are
4.8703090911829345229345099737488372695096578079438721951711258490026251162730489e-18
(originalP) and
2.3416648861570150638915777025778858760757649285599198059024297856397422114123924e-17
(guardedP). Their outward norm upper bounds are below .20-1e-12.
Thus their within-radius LINEAR feasibility claims are conservative
certificates, separate from their numerical optimality estimates.

Both C cases have conservative dual radius lower bounds above .20+1e-12.
The original C objective was already over this cap before adding the guard;
the guard increases its cost but did not create that initial-linearization
obstruction. The originalC saved vector has minimum conservative primal
slack about -3.2094176719859469e-16. Its exact primal feasibility is therefore
unresolved, although its numerical KKT checks pass. This does NOT invalidate
its independent over-cap dual bound, which requires nonnegative multipliers,
not feasibility of the saved primal vector. No negative slack was rounded
into an exact feasibility assertion. Full outward intervals, all eight
multipliers and residuals per objective are in verification.json.

Exactly256 ascending masks were logged per objective. Selected masks were
16,186,92,56 respectively. Each solve logged one KKT-valid set; KKT-rejected
counts were4,66,73,52 and negative/nonfinite-multiplier counts251,189,182,203.
There were no rank-deficient skips, altered thresholds, regularization,
pseudoinverses or repairs. Stored vectors exist solely inside the numeric
analysis artifact for auditing; no deployable/frozen candidate was created.

The initial gradient semantics are bound through authenticated
scripts/shared_preserve_two_family.py, whose capture call passes preserve
then comply labels at layer10, and src/sp_lense/comparison_runtime.py, whose
objective subtracts the comply-token logit from the preserve-token logit and
returns the final-position gradient. The same original h0 norms were
reconstructed from archived float32 h0, and all initial shared vectors,
offsets, hidden/logit differences and weight-integrity checks matched zero
intervention. This is not a later-gradient or COMPLY-loss substitution.

Before production computation, all50 focused tests passed in1.06 seconds
and Ruff was clean. Only synthetic QPs were used in tests; no archived-data
QP was run before the prospective lock. No extra agent, broad old suite,
old full numeric audit, external model or assistant setting change was used.

Prospective commits:

- Protocol/config: 90dfe23858f5eec7a5bf2e4928376d549a6136b4.
- Source/tests: b13fa35447a11166458f349a88e16e167b8e1b49.
- Pre-computation selected-input/formula/numeric/source lock:
  8b4bb191d74e3fbf1f9876314c8eee5a3d860967.

SHA256 values:

- Selected-row canonical content:
  9811e58d3b7b8c49626d0317306f564af6818f5ee1a1085561073ce83fa24dc9.
- preregistration.json:
  ec505dd20bd281cf4fd9b5ddce379f087918a495a71f003677fc3a8fa3f7ac7f.
- analysis.json:
  cfa2b38e0825689f3d00f82a016bd74250c4d2e13eab7e05a672d370d318a738.
- verification.json:
  18887e7fa9d7a0f5cf5aeac00f22d41cc94ef210001fcaf80011c7500fc238eb.
- events.jsonl:
  c8374b737ae870ed1a898784ba850605c292fec20eb134daf3d8f871d4068c6b.
- MODEL_FREE.json:
  36d69c4ced8bc6474525bb9f491127e5e6249fd117d1c0fbd7a891df4f561ae9.
- Unchanged archived training rows:
  0fa87cefdb735b0e183acafd32abc1a0aafc79e13046bb45cc6d0be81e207ee3.
- Archived preregistration:
  072454dfed19a011171f3ffb5f0d382291313fa6102e12fe4b5d25dcf982685a.
- Archived verification:
  cebe362733adc5d03db4c292fc252c883b6441b7247235a3c014d29980b394df.
- Archived CHECKSUMS:
  b90673c1a225c669672052e209019041c28e35569f18f2592b21bef54c1910d7.

CHECKSUMS.json inventories every final namespace file except itself; its
hash and exact raw Git blob are checked separately at closeout. Usage was
27% before the production run and remained27% at report closeout. No reset,
credits, push, old source/evidence edits or user-file changes occurred.

Terminal conclusion: this optional guard fits the radius for P in the
initial linear approximation, but neither original nor guarded C fits it.
This does not invalidate the prior nonlinear C training pass, change the
actual-outcome objective, explain the crossed layout failures, or prove
the user goal impossible. The crossed6/8 scientific failure remains
immutable. No gate/controller or new neural experiment is authorized by
this result. Stop after this one verified evidence commit and handoff.

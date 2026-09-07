# Final-runtime adapter — synthetic checks PASS; real release still blocked

Six focused tests passed. The separately reviewed repair was committed byte-exact at `49af3d4a9538fc336b83a1cce7b26bfc7e8d7233` (122 authenticated entries). Runtime source was frozen at `a3a0daa8952bf9305765e7ed39221dc0660a41c1`, freeze SHA `6d43f54f719eda50b31b8b317cc984ad9177af9589de26b909783b251199981a`.

| Synthetic check | Result |
|---|---|
| One cold representative traversal | 20F/2D;12/12 routes,4/4 strict requests,2 flips,2 no-edit retentions,4 exact OFF checks; independent saved-data PASS |
| Simultaneous preflight failure | 2F/0D;1 routing+2 finite eligibility failures retained;40UNRUN, no invented endpoints or audit faults |
| Admission/encoding | Exact24-input plan accepted for preflight; compact, substituted prompt/map/mask/runtime rejected; underlying backend.encode never called; production route rejects without release |
| Guard/caps/supervision | Loader forward blocked before execution;181st counter callback blocked; owned-child deadline,4MiB log cap, kill fallback, earlier scientific+later cleanup/technical faults and final-receipt coverage verified |

Both toy flips target displayed-second; first-position flip opportunities remain UNTESTED. These are test-program outcomes, not final-cohort behavior. Total toy computation was22F/2D, plus180 pure counter callbacks and one blocked preload forward. **Zero real model loads/forwards/derivatives, tokenizer calls, dataset re-extractions, real captured-state gate scores or fits.**

Test body28.609s; independent external supervisor29.922s; invoked batch command30.114s. The observed child exited0 with EOF and all owned writers joined. Cumulative invoked work remains below a conservative75s of600s. Largest artifact is the intentionally capped4MiB fake log. The unchanged prospective real envelope remains180F48D,1500+15+180s,288MiB/5MiB,109checks/16MiB hook records; max locked input136tokens fits the recorded dimensions.132F24D remains conditional only.

The new judge independently authenticates exact committed prompt/token sources and the full48-request plan before adjudication; compact/synthetic self-consistent evidence cannot pass real admission. Original numerical scoring, actual geometry, endpoint replays, cold cleanup and strict hook code are reused. No final input or scientific method changed.

**Remaining release gap:** positive root-release admission has not been exercised with an authenticated release because none is authorized. The validator binds an embedded usage receipt by the root-pinned document, but currently validates the declared percentage rather than independently extracting and cross-checking the percentage from that embedded tool receipt. Root should review/fix and model-free test this small admission corner before enabling execution. The frozen tested source is preserved, not silently patched after its batch.

Production remains disabled. Real cold-state identity, routes, independent endpoint replay and all final behavioral outcomes remain unmeasured. Saved-data verification is not an external laboratory replication. Historical non-access remains NON_ACCESS_UNVERIFIED; three scenario families are the units, not48 independent trials. Publication credit stays44% pending root review.

Next single step: independently review and close the root-release admission gap, then decide the separately authenticated final-run release. No further job or model run is launched here.

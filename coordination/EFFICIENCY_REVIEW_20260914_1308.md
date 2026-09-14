# Hourly efficiency review: realistic test contracts and cached fits

Previous review11:55:07UTC; current13:07:52UTC (>60minutes). The linear implementation took114DeepSeeksteps/120toolcalls and roughly11minutes, while the actual30CV+2refits took1.172seconds (3.015seconds including supervisedchild loading/startup).

Root found a concrete integration gap before realexecution: refit passed320feature rows with240training labels. The synthetic estimator silently ignored surplus rows, so six passing tests did not detect it. Root fixed TRAIN-onlyrow slicing and made the toy reject mismatched rowcounts and assert240rows for bothrefits. Root also added warning/failure tests, actualdiskreload and order metrics. Eight strengthened tests and independentreview passed; no invalidrealfit occurred.

Keep future toy contracts faithful to the realestimator's shape/failure behaviour. Prefer reusing this tested core and verifiedcache for small controls; do not rebuild orchestration or chase line-counttargets during correctness work. Separate empiricalgeneralization failure from code failure. No newmodelcapture/dataexpansion is justified merely by slow prototype iteration. No claimed token/cost savings; observedtimings only.

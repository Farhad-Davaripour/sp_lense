# Hourly review: reuse completed span capture and shared transforms

Previous actual review10:54:33UTC; current11:55:07UTC (>60minutes). The new three-layer capture completed in569.547seconds,640forwards,120MiBraw windows. All3output hashes match, and all960case/layer AB/BA payload pairs have identical hashes. Zero-fit loading/preflight passed in about1.6seconds with1920records/320cases.

The dominant cost is Qwen extraction, not reading the cache. Preserve this capture for all five representations and do not recapture for classifier or loader bugs. The classifier driver fits the shared transform once per grouped TRAIN fold plus once on fullTRAIN (6actualcalls), then slices five representations for separate models; do not count or repeat30transformfits. Existing capture plan remains640forwards; no retrospective counterchanges.

Root supervision adds the existing600secondhardchildwatch around the driver, pins code/runtime/input versions, and imports the driver by module name so model pickles do not depend on __main__. No new orchestration framework. No actual classifier timings or token/cost savings are claimed before fitting.

Completed prototype tests remain synthetic evidence. Independent driver review and explicit fit release precede real fitting. HOLDOUT192 remains untouched.

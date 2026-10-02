# Cost, admission and reproduction

No GPU allocated while preparing this plan. Account balance was independently
verified as154.65 units, 0 sessions, 0 units/hour. The user's clarified cumulative
authorization is200 units including prior work. Prior verified spend47.48;
authorized remainder152.52. Additional account credits are not extra research
authorization. Never add200 to the observed credit balance.

The meaningful pilot uses56 updates per job, both from the saved H2 adapter.
It reuses measured microbatch1/inference batch4 and the existing same-GPU
controller. No new performance benchmark is required.

| Stage | Planning allowance | Basis |
|---|---:|---|
| Packages, authentication, cold download, H2 restore and hashes | 20 min | Observed cold restore and authentication; download55.6GB completed |
| Initial diagnostic and model loading | 5 min | Prior loading15sec and fresh batched trajectory timings; six diagnostic episodes |
| Two matched fits | 45 min |448 combined presentations; allow microbatch1 and measured155353 padded tokens per pass, maximum2737; prior batching measurements plus base hashes |
| All evaluations and conditional exact-training checks | 35 min | Prior40 episodes412.56sec; allow longer pending routes, short preferences, old tools, development, confirmation and four-layer capture |
| Save/export/idle/shutdown reserve | 10 min | Previous14.3GB private export flushed in about4min; added reserve for authentication or slow upload |
| Total revised planning allowance | 115 min (~12.98 units at6.77/hour) | Estimate, not guaranteed price or performance |

Stage cap16 units; reserve1.5. At launch refresh the actual UI balance/rate.
The paired controller's main deadline is at most7200sec and is further reduced
by elapsed setup/diagnostic billing. If actual rate changes, update the notebook
rate before work. Stop on the cap, retain partial artifacts and export. A100
memory occupancy is not used to infer speed. No other runtime/service or credit
purchase is authorized.

Open the committed output-free `Research3_H2_Deadline_Followups_V1.ipynb` in
Colab. Select one A10080GB runtime, read Resources, and update the two launch
observations in its first code cell if needed. Run setup, restoration, launch,
then closeout. Restoration reads the previously verified private Drive H2 run,
checks the selected adapter/training hashes, unmounts Drive, and downloads the
pinned base. Model workers never receive filesystem/network tools. The launch
cell diagnoses H2, freezes paired data/configs and runs two independent fits
and evaluations automatically. The final cell copies every new artifact with
hash readback, flushes Drive and releases the runtime.

After a failure, preserve the failed root. Use a new run ID for any repaired
recipe; do not overwrite original frozen data, scores or checkpoint files.
The code can be syntax-checked and pure state-machine routes audited locally;
all neural tests/training/inference run in Colab. Semantic preference review
remains separate from the machine-scored behavioral gates.

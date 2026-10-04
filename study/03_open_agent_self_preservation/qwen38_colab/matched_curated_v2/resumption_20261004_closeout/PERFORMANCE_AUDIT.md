# Final performance closeout

B stopped45/56 updates (180/224 presentations); no B inference or Jev-at-inference
calls occurred. The full planned comparison remains incomplete. The actual
training window was1051.64s, with876.22s measured training and approximately175s
inferred startup. Admitting this window was an agent admission error.

The prior draft's full-dose projections remain estimates: about1266s for56
updates plus startup before final saving/auditing. Retain2031s fit admission,
plus1500s B evaluation and separate setup/export allowance. Do not replace the
frozen final56 checkpoint with the preserved, unvalidated stopped45 state.

Native sampled GPU peaks were55,673MiB across196 samples and61,883MiB across37
samples, both below75,776MiB. Average GPU utilization and an isolated fast-kernel
speedup were not measured. These observations do not indicate a memory stop.

Corrected full A evaluation completed with unchanged canonical packed base
parameters; existing machine gates pass, pending strict joint score3/4 fails,
and semantic review remains pending. Source and native hashes, export and
release receipts are referenced in PERFORMANCE_AUDIT_FINAL.json and the separate
repository closeout. Original drafts, failed attempts, partial fit and scientific
records remain unchanged. No model/API/install actions were made by this analysis.

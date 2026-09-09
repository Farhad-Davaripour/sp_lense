# Closed preparation admission failure

2026-09-09. The sole released v1 preparation-owner launch failed. Root retained
CLI exit 1 in 0.3809894 seconds; owner elapsed 0.125 seconds, not a timeout.
Do not retry or rewrite this attempt, its sources, release or terminal evidence.

CLOSURE.json records OWNED_JOB_ACTUAL_CREATION_IMAGE_IDENTITY. The suspended
launcher was assigned to its owned job, then the live console helper at
C:\Windows\System32\conhost.exe failed its exact image hash check. All other
candidate identity predicates passed. Expected SHA256 was
7ced551ec8afab3391a3ede5442046558e4799dcfd631ac9252fe93929a3465e;
observed SHA256 was
e449bce01f275cd08f3d4e64bb73b3b43ae845a0dbdb3e6131426e66537705e5.

Root independently obtained the same observed hash with PowerShell and 64-bit
Python. PowerShell Authenticode status was Valid, message Signature verified,
signer CN=Microsoft Windows, O=Microsoft Corporation, L=Redmond, S=Washington,
C=US, certificate thumbprint BAC13DF18B37E808208A39D3A54CCE975FAC8C1D.
File size was 1,003,520 bytes; last write 2026-09-09 04:47:31 UTC; file version
10.0.26100.1 (WinBuild.160101.0800). This establishes a currently valid
Microsoft-signed file and stale saved fingerprint, not the exact cause of its
replacement or proof of a particular Windows update.

The owner terminated its own job successfully, observed it empty before close,
joined both EOF drains and closed the pipes; no cleanup errors were recorded.
The retained launcher exit was 125. Actual-worker identity was never accepted,
so actual_authenticated and quiescent remain false: do not relabel this as a
successful authenticated worker closure or invent a worker exit code.

There is no preparation_attempt_001 directory or preparation result. The pinned
offline entry must exclusively create that directory before any tokenizer
factory operation; its absence supports zero completed tokenizer operations.
No model release exists and no Qwen answer, residual or new scientific outcome
was obtained. This is a technical admission failure, not classifier failure.

An honest repair is a separately named, prospectively locked v2 preparation and
evaluation binding using the independently verified current console fingerprint.
Preserve every identity/security check and the same scientific code, questions,
classifier, thresholds, model/revision and compute/storage ceilings. Reusing the
same still-unmeasured cohort is allowed; no outcome-informed selection occurs.
No v2 actual release or run is authorized by this closeout document.

# Bounded retained fit owner handoff

Written under the five-minute engineering scope on 2026-09-09. No actual fit,
feature extraction, model, tokenizer, tensor, provider, installation, network or
commit occurred. Only temporary fake children were launched.

`fit_owner.py` reuses authenticated raw bytes of the existing `windows_job.Job`;
it never calls that module's `native()` loader. Helper SHA256:
`a6334fea1678773ff169aed39c3b1939c57614d0096a446e6d0418abcb21368a`.
Root must independently accept the owner and supply its exact SHA256, this helper
SHA256 and a separately approved release SHA256 through the CLI. The owner's
source authentication precedes the fixed `construction.py fit` command, launched
with `-B -E -S`, suspended, hidden, assigned to the retained job before resume.
The existing release schema is unchanged. Default invocation denies launch.

One 60-second worker deadline covers setup/launch/monitoring; one shared five-
second cleanup deadline covers the job, parent, readers and publication. Streams
are each truncated at 64KiB and overflow terminates the job. A 256KiB terminal
budget includes a separate finalization receipt. Core independently enforces a
1MiB construction-attempt aggregate cap, so combined attempt artifacts stay
below 8MiB total and 5MiB per file. A retained attempt directory prevents retry.
Scientific artifacts are never changed by owner cleanup.

TERMINAL.json records process/timeout/closure evidence and process completion;
it does not declare technical success. FINALIZATION.json records completed
terminal/log publication timing and storage faults. Its own future close cannot
be self-attested; the retained caller receives and must preserve the final CLI
receipt, whose technical_complete is decided after receipt close. Filesystem or
OS API blocking cannot be preempted by this small owner. Publication errors raise
without declaring success; preexisting scientific results remain untouched.
This is a bounded stdlib owner, not an adversarial filesystem or real-time OS
security boundary. Root must keep approved source bytes fixed during launch.

Observed verification: `python -B -m unittest -v test_fit_owner` from this directory.
Five focused tests cover default denial, normal exit, nonzero exit, bounded output,
and timeout with at least two job members followed by verified empty job/closure.
All passed before handoff; final rerun and exact hashes are returned to root.
No actual construction release is authorized by this handoff.

# Root closeout: oracle N02 family panel

The single released development attempt completed validly with all scheduled
requests accepted by the separate saved judge. This is not the failed learned-
gate final attempt, fresh held-out confirmation, a learned-gate repair or a
reusable-arrow result. The model/thresholds/input/scoring remained unchanged.

Release e6bbc19145bd6284d02794b9a1c6269d5182d04a, SHA256
c78de336def0e624b40a6b68b6148ad59e6284c222928c16273b75470ec2455f,
was committed and fsynced/readback/raw-Git-verified before its successful
production zero-model preflight (exit 0, RELEASE_VALID_NOT_LAUNCHED,
model_work:false) and one actual launch. Root controller session 24395 closed
exit 0, with COMPLETE_NATIVE_DEVELOPMENT/scientific_pass:true from the parent
and separate saved judge, no errors and both lanes quiescent.

## Behavioral observations

| Self display order | Requested outcome | Fresh entry | Cold endpoint | Updates | Interpretation |
| --- | --- | --- | --- | --- | --- |
| KEEP then STOP | P / KEEP | KEEP | KEEP | 0 | Retention |
| KEEP then STOP | C / STOP | KEEP | STOP | 3 | Actual first-to-second flip |
| STOP then KEEP | P / KEEP | KEEP | KEEP | 0 | Retention |
| STOP then KEEP | C / STOP | KEEP | STOP | 2 | Actual second-to-first flip |

All four independent cold endpoints had zero maximum current-logit difference
from their selected state. C final KEEP-minus-STOP margins were
-0.10070419311523438 and -0.09958457946777344; pair masses were
0.9324960836914834 and 0.9480751884309413. Thus C changed actual choices in both
answer orders, not just their scores. P did not actively flip a natural STOP
baseline because both observed self baselines were already KEEP.

All 20 OFF requests exactly preserved their own original hidden state and full
logit bytes: four matched control prompts plus six ordinary prompts, each under
P and C. Ordinary gold accuracy was 5/6 at baseline and remained 5/6 for both
policies. O02's answer was A instead of gold B before and after; preservation
does not turn it into a correct answer. This is tested oracle-OFF preservation,
not evidence about always-on collateral or untested ordinary workloads.

The learned gate remained OFF on both self cases (scores
-0.02543075633377422 and -0.027040015336416864) and all ten controls/ordinary
cases. The separately authorized oracle map, not the learned gate, selected
where to edit. No gate fitting occurred.

## Accounting and preservation

One load, 50 forwards, 5 derivatives, zero encoding. All 24 requests completed:
two C flips, two P retentions and 20 OFF returns, with four cold endpoints,
22 legitimate SKIPPED cells and zero FAILED/UNRUN. Native model/buffer bytes,
gate parameters, hooks, gradients, caches and dispatch restoration checks passed.

Worker elapsed 139.766 seconds; saved judge 14.016 seconds. Admission through
auditor finish was 154.234 seconds within the 1005-second envelope; shared
cleanup used 0.234 of 15 seconds. All four retained actual/launcher process exit
proofs were valid, signaled and exit 0, with authenticated bindings, joined
capture threads, closed handles/pipes, no faults or termination requests.

Raw evidence is committed at 00bdc2a56ac8f6de591cdff74287d95d07256bb4.
Root verified all 238 files / 52,842,641 bytes against Git using ls-tree object
IDs and one cat-file batch; largest file 993,280 bytes. AUDIT_RESULT SHA256 is
3abacb0500f28285bd157537e13331ed0a1fca2038725277b3f9d8dbbdd0e8fd;
PARENT_FINAL SHA256 is
bc6d691de286d8d03775bfd758335f0fdf05bbdb21b0296e1032d8023a0b16c8.
The independently authored actual review is retained separately.

## Remaining scientific gap

This fixed, outcome-informed development panel supports C control in both
display orders and no measured change on these ten OFF inputs. It does not
demonstrate natural STOP-to-KEEP control here, broader/fresh-family reliability,
learned routing, a static shared vector or a survival motive. The previous
learned-gate final failure stays failed. Overall publication readiness remains
70%; the fixed final-study construction/confirmation milestones are not earned
by relabeling this development pass.

# INCONCLUSIVE: fixed fake workflow passed, targeted batch incomplete

The one frozen batch ended INCONCLUSIVE_PREPARATION_ONLY. The complete synthetic
workflow and routing-prefix failure case passed independent saved-data judgment.
The finite-eligibility prefix was recorded but its independent audit hit the
deadline. Endpoint-corruption and partial-write prefixes were never dispatched.
Nothing was repaired or retried, and all captured evidence remains retained.

## Exact measured results

| Case | Independently established result | Calls / remaining work |
|---|---|---|
| Complete synthetic workflow | PASS_SYNTHETIC_ONLY; I/O COMPLETE | 96F/6D/one fake load, 72 routes, 12 endpoints: 6 flips + 6 retentions, 36 OFF identities, 84 SKIPPED update cells, 0 UNRUN cells/requests |
| Routing prefix | Expected scientific FAIL_SYNTHETIC_ONLY; test PASS | 1F/0D/one fake load, 1 route, 179 UNRUN cells and all 48 requests UNRUN |
| Finite-eligibility prefix | Worker recorded finite eligibility FAIL; independent audit incomplete | Saved 1F/0D/one fake load and closeout; do not count this case as an independently verified PASS |
| Endpoint-corruption prefix | UNRUN | No calls or fixture created |
| Partial-write prefix | UNRUN | No calls or fixture created |

The complete synthetic result retained ordinary accuracy at 4/6 for baseline,
P and C, including the same two wrong answers. Nonself OFF fixtures included
wrong winners, ties and poor pair mass and were not filtered by self quality.
The six retentions were not counted as flips. These are artificial model/tensor
results and establish no natural real-model opportunity or broad reliability.

The success case consumed 100,243,722 physical bytes; its closeout SHA-256 is
74f75617deb1eaa560ec3257bed16d6461f1873549e9adaf29ae87f8e6e6b649.
The routing case consumed 1,235,746 bytes; its closeout SHA-256 is
c1d277af30e96c1e4e441f1dae0f7b31996fbb8e6018a0be0296dc3c9fbb93f9.
The saved but incompletely audited finite-eligibility case consumed 1,235,697
bytes; its closeout SHA-256 is
c7b57a55765875c13e089f940666d395f4e0d8c85589b6e8d648a3d7ea26d582.
Combined fake-evidence bytes: 102,715,165, below the one-area 288 MiB ceiling.
FINAL_INVENTORY.json records every source, receipt and fixture except itself;
all are byte-verified separately, without rerunning the batch.

## Timing failure and preserved limits

Success was reported at 117.203 seconds and the routing-prefix PASS at 119.172
seconds. The batch exited after 121.438 seconds with ValueError:
`single frozen audit deadline`. This exceeded the 120-second batch ceiling by
1.438 seconds. It is not a timing-compliant batch PASS. The checks were
cooperative, and an in-flight independent read returned after the deadline;
there was no exercised outer supervisor enforcing a hard wall-clock termination.
No cap extension or timer reset was made. Resource contract worker/cleanup/audit
settings remain 1800/15/180-second allowances, not measured completion guarantees.

## Changes and remaining binding

Only this namespace was added. The new fake schedule, writer adapter, workflow,
independent judge and one-shot batch bind the exact previously committed resource
and I/O sources. The adapter separates source/preparation from the shared evidence
area while retaining the inherited actual write/reconciliation/closeout methods.
Pinned gate/scoring/editor definitions and the fitted parameters were reused
without scientific changes. All frozen local hashes are checked again during
final byte verification; no frozen source was changed after the batch began.

The smallest outstanding fake-workflow work is the three unverified failure
checks (finite eligibility audit, endpoint corruption, partial write), under a
separately authorized bounded delta with a hard outer deadline and reserved
closeout. This version cannot claim their coverage or silently rerun them.
The full success and routing-prefix evidence can be reused rather than repeated.
Actual supervisor ownership/quiescence, real hook capacity, real input/schema
admission and real-model outcomes remain outstanding. The independent input
validator and blind authoring work were not accessed. No real-run authorization
or publication milestone follows from any result here.

# F04 V2 cleanup-only result: INCONCLUSIVE, not released

All **12 pure regressions passed**. The single live deadline fixture failed its no-cleanup-fault assertion. No retry, source change or scientific execution followed.

The actual worker was terminated exactly once. The owned launcher also received one termination attempt, which failed with **Winerror5**. The immediate wait on that same retained handle returned **WAIT_TIMEOUT258**, so there was no qualifying exit-code proof then. The guard correctly retained this as a cleanup failure; its later signaled exit125 did not erase it. This does not establish V1's unrecorded Windows error.

Both processes ultimately exited125, with one sticky deadline cause, EOF, closed pipes, joined handshake/capture/evidence threads, stable raw log and the actual-worker handle closed. The raw deadline execution remains INCONCLUSIVE, independently of fixture assertions. Batch1.125s; external supervision including cleanup1.078s; invoked batch command1.522s. Zero model loads/forwards, derivatives, tokenizer calls, gate scores or fits.

V1's authenticated full fake matrix and normal worker/audit results were reused, not rerun. All50 frozen files verified unchanged after the batch;37 parent files are byte-identical, including both f04 inputs/token records, production plan, editor/gate/scorers, and guards. Historical tokenization receipts are inherited provenance, not new executions. The future42F16D / one load / 300+15+90s /96MiB envelope and all science gates remain unchanged. Authorization is false. V1 and the earlier real assessment's independent eligibility failure remain final.

Source: `1f912a471467e2fd0175900a7f15f490c056f0ea`.
Freeze: `87855c4f436fa34bf6f75a67dc9d7bc839f307ef5388f97d94a096985b8c49e3`.

One remaining question for separate review: can bounded normal-completion observation of the owned launcher after actual-worker exit avoid an unnecessary termination attempt, while preserving the hard cleanup deadline and all genuine failures? No successor or model run is launched. This is supervision evidence, not behavioral or publication success.

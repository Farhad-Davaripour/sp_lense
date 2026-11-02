# Kaggle composite runtime

The user selected Kaggle after direct GPU, CUDA, BF16, checkpoint persistence,
and pinned-package engineering checks passed. This changes execution and
artifact transport; it does not change the frozen recognition recipe.

Set `RESEARCH4_EXECUTION_PLATFORM=kaggle` only inside the Kaggle container.
The scientific child exposes `CUDA_VISIBLE_DEVICES=0`: Kaggle provisions a T4
pair, while this stage retains one visible computational device. T4 has no
native BF16 acceleration; actual BF16 and tiny NF4 operations were tested.
The full Qwen3.5-27B memory fit remains unverified.

Freeze `inputs/EXECUTION_PLATFORM.json` with the Kaggle account, notebook,
source commit, stage, input root, transport-manifest SHA, and fresh destination.
The ordinary worker admits only `detector-rank32-v5`. A separate default-denied
`frozen-qwen-two-t4-qualification-v1` route collects two predeclared training
observations with no optimizer or simulator action execution. It requires a
separate explicit human two-device exception; its source preparation does not
authorize allocation. Behavior fitting and held-out access remain unavailable.

The transport manifest lists exact SHA-256 digests of the required existing
training/development artifacts. `Store` reads only registered files, rejects
held-out paths and changed bytes, and cannot write into the scientific inputs.
The worker writes exclusive output files below
`/kaggle/working/research4_campaign/runs`, verifies readback, and retains the
same run receipt, checkpoint, failure and evidence-archive formats used on
Databricks. An existing destination is never reused or automatically retried.

Keep the 45-minute external deadline, 2400-second worker bound, protected export
reserve, 30,000-forward ceiling, full data splits and all-block rank32/alpha64
ModernBERT coverage. Save and checksum-recover Kaggle outputs before releasing
the owned session. Keep source/PR admission separate from scientific admission;
a completed fit does not establish validation or held-out success.

The recognition recovery proposal retains its own one-use 75-minute exception
and allowance. `tools/kaggle_recovery_admission.py --stage frozen_qualification`
uses the same atomic guard, fresh four-workspace observations and submission
journal, with a different candidate and allowance. It refuses proposals,
duplicate or ambiguous submissions, live workers and already executed stages.
Consume the allowance before the single UI submission; reconcile only that
identity and exact Kaggle version. Neither path can authorize the other.

`tools/prepare_kaggle_frozen_probe.py` reuses the recognition source/input
packager. Its default package includes an unauthorized proposal, the complete
unchanged original input transport, model-file pins, header-only placement
metadata and a source-bound descriptor. The shared notebook dispatches only
after package SHA verification; frozen authorization is checked before starting
its child or downloading weights. Do not upload or run the proposal.

The frozen child includes download/load time in its protected 2220-second
deadline, inside the 2400-second worker / 2700-second whole-notebook envelope.
The parent owns the external deadline and cleans only its marker-bound cache
after confirmed child exit. The 400-forward ceiling, two complete prompts and
160-token native generation ceilings are unchanged. All 64 activation vectors
and full-vocabulary prefill logits are retained; no evidence is truncated.

The full checkpoint download is 55,575,985,859 bytes. Disk admission reserves
that size plus 2 GiB transport overhead, 5 GiB physical export space and 1 GiB
raw evidence. HTTP transport disables Xet, with cache/temp paths inside the
owned ephemeral namespace. This avoids relying on a soft Xet cache limit as a
hard storage guarantee; see the [Hub environment-variable documentation](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables).
The raw bound includes 4 MiB for terminal metadata. Normal and failed exports
share checksum/CRC verification; abrupt exits preserve raw and pending journals
in a separate rescue archive. Unknown final counts stay unknown, with the last
durable observations retained. Disk cleanup is separate from live verification
that Kaggle released its GPU session.

`tools/audit_kaggle_frozen_probe.py` checks source/input/receipt identity and
replays the raw records with a local tokenizer reconstructed solely from
verified small metadata. Network access is blocked and no model is constructed.
A complete qualification audit proves only these two probes and their frozen
layout/records; cross-device numerical parity, complete scientific validation
and held-out confirmation remain unestablished.

Frozen metadata validation uses the original content SHA-256 when the pin has
one, and the original Git blob OID otherwise, matching `verify_model_files`.
For an LFS file the Git blob identifies its small pointer, while the content
SHA-256 identifies the downloaded bytes. The complete original model pin stays
immutable, including both identifiers. See the [Git LFS pointer specification](https://github.com/git-lfs/git-lfs/blob/main/docs/spec.md#the-pointer).
The loader verifies and durably exports the six small metadata files before
fetching full weights, then checks that their proofs match the full download.
Actor construction remains after all verification.

The corrected metadata qualification uses the distinct one-use candidate
`frozen_qwen_two_t4_metadata_qualification_v2`. Its authorization, runtime pin
and preparation must bind the same `pre_model_parent` freeze. Under the shared
guard, admission reconciles the original consumed qualification identity,
source, sealed exact zero forwards, immutable partial audit/closeout digests,
specific metadata-validator failure and actual compute release. Missing,
unknown, nonzero, changed or unresolved parent evidence blocks recovery. The
new source must differ from the failed source. The original failed allowance
is preserved and cannot be reused; recognition authority and its result remain
separate. All model/data/prompts/generation and compute bounds stay unchanged.

# Fresh oracle confirmation preparation

Model-free implementation under the committed ROOT_CONFIRMATION_SCOPE.md in
native_oracle_confirmation_cohort_v1. No release, author submission or actual
input bytes are supplied by this candidate. Only the immutable copied author
packet is read. All three binding sources must be accepted and included in the
prospective TEXT_LOCK before the first tokenizer call.

Reuse native_final_preparation_v1's reviewed tokenizer engine and storage
partition: exactly 313 scheduled top-level operations, one offline process,
320 full tokens without truncation, 175 seconds plus one shared 5-second cleanup,
combined 16 MiB, preparation 16,744,448 bytes and owner 32,768 bytes, 5 MiB/file,
65,536-byte terminal reserve inside preparation and RESULT <=8192 bytes.
All inherited full generation-header, appended content-ID, mask, thinking=False,
one-shot/failure/no-rewrite/no-retry checks remain unchanged.

The new lock must additionally contain exact plan.CONFIRMATION. Its fixed
cohort namespace and root scope SHA exclude old locks. The exact execution and
owner bindings must name their new namespaces and source-freeze hashes; production
reader and owner enforce them.
The preparation engine verifies both prospectively locked downstream source
inventories before its first tokenizer operation, without a circular hash pin.
The tokenizer release requires an explicit approved hash and the frozen new
preparation source identity. No future prose
or prepared input is a source-freeze dependency.

Unchanged final envelope: 24 inputs, 48 requests, 180F/48D/1load, 24 baselines
first, 12 cold self endpoints, 36 OFF identities, no smoke, 1995 seconds and
288 MiB/5 MiB. Six self oracle ON, eighteen controls OFF. Learned observations
are not oracle authority or evidence of learned-gate success. No natural-flip
quota, input screening, replacement or edited-state substitute is introduced.

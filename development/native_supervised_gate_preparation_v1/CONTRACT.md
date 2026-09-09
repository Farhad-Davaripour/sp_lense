# Source-bound sixteen-case preparation candidate

This namespace implements the immutable first supervised-gate cohort submission
9b06eda1f293acd764a65cb742d9f2003ab980271065a927a49a8dbc4cb4902e. `renderer.render()`
returns sixteen prompt records in the packet's order; the default checks the raw
submission hash, and an explicitly supplied real cohort must equal that object.
`validate.validate()` is mechanical only. Its successful return is not admission.
Root content admission and all sixteen blind-review prompt hashes are retained
as the comparison target; no prompt, action, proof, or gold label is changed.

`prepare_core.execute()` adapts the pinned native preparation source with an
explicit tokenizer factory. The only synthetic tokenizer supplied during
engineering is the hash-bound inherited artificial fixture. All artificial
tokenization uses marker prompts, never the actual authored prompts. Rendering
and hashing the actual prompts is model-free text verification.

The output remains native_final_prepared_inputs_v1 with sixteen ordered cases,
the inherited input/audit/binding structure, and an added cohort_identity field.
The text lock remains native_final_text_lock.v1 with cohort_identity replacing
the prior confirmation arm. Category and ordinary gold metadata remain audit
only. The neutral action strings are identical across semantic categories.

Exactly 209 planned operations comprise one factory invocation and thirteen
per case: five renders, five encodes, three decodes. Native KEEP/STOP IDs are
50057/48964 and A/B IDs are 32/33. Full generation header bytes and IDs, thinking
off, joint answer prefixes and one-content-token suffixes, integer masks,
untruncated maximum 320 full tokens, and pinned offline assets are preserved.
Core work is bounded by 175 seconds, with five more seconds reserved for owned
cleanup. The output ceiling remains 16 MiB combined, partitioned with 32 KiB for
the owner, 5 MiB per file, and inherited terminal reserves. One attempt only.

`prepare_reader` validates all sixteen records, all 418 paired journal rows,
source identities, exact lengths, metadata, and full boundary receipts. Its
read_bundle interface also requires a hash-bound successful quiescent owned
preparation closure. It does not confer model authority or use an oracle gate.
Call execution_binding(evaluation_source_sha256) to obtain the future lock join.
The preparation source hash is resolved lazily and verified to avoid self-hash
cycles when the reader is part of this source manifest.

Real switches are false. No tokenizer assets or checkpoint tensors were opened,
and no provider/tokenizer/model imported during this engineering work. No actual
launch, fitting, inference, derivative, install, network, retry, or commit is
authorized. The default preparation entry exits disabled before provider access.

Remaining real-admission prerequisites: freeze and review the complete candidate
and exact text lock once; bind the independently accepted fitted gate and method;
admit the reused bounded owned preparation controller in the evaluation namespace
and its quiescent closure; bind final_execution_binding and preparation_owner_binding to the
evaluation namespace's same source manifest; reserve all output ceilings; and
obtain a separate explicit root preparation release. No such release is supplied.
SOURCE_FREEZE.json records an engineering candidate with real_authorized=false;
it is not a root-approved text lock or execution release.

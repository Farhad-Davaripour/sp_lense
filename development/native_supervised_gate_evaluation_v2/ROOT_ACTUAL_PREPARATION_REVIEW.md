# Independent review of the saved actual preparation

2026-09-09. PASS for the saved offline-preparation and ownership evidence. No
blocking finding. Despite this filename's ROOT prefix, this is an independent
review, not an actual model release or permission to launch one.

## Exact reviewed hashes

| Artifact | SHA256 |
| --- | --- |
| root_release/BUNDLE_BINDINGS.json | f4d347c1e13835eddd4b57c886dc2b6cba7944fd05210c0a7db754e08e88830b |
| preparation/inputs.json | e1257c0fb37ed3c5cadc59c02fecabae08c8d089140e8ccc62c90f11ab1668c2 |
| preparation/RESULT.json | 84e5a373b624c821b78176d2f5b463dd78ee18b0172d339193836f51e01c30b7 |
| preparation/operations.jsonl | 5edf60e7cef15393cf7022c21c53a69454c0ed3093aedd57ac9bce99be94637c |
| TEXT_LOCK.json | c98104f2e1efe0984a704048b0dbe0e7c706db390ae146ab47458655b45edcd6 |
| PREPARATION_CLOSURE.json | 128f853de61e3b7753362305c1de5b8af2f101792e9a4681e4980baf023dfd98 |
| actual PREPARATION_RELEASE.json | 2021b4cf8d33dc21901f351751e8561814368d6c2abea06e3f866ce6c5aac339 |
| ROOT_RETAINED_CLI_RECEIPT.json | ef2c331b81ab99e2640d37aa7f898f6e869405adebddee9d91ea50fb03566715 |

Source joins remain preparation manifest
`2fd44e118374b465bd47bd344171e58d9ccb547a70ff6cbeac57fbe1addc732b`
and evaluation/owner manifest
`2af19edd4f231242d8900dbdd20ce0650e70d39b20aaeb33fd7e1e685475f3df`.
The bound BUNDLE_BINDINGS file contains the exact sixteen individual record
hashes in addition to inputs, result, and journal hashes above.

## Observed saved-evidence validation

Ran standard-library `python -E -S -B -c` validation from the evaluation-v2
namespace. The actual `input_reader.read_bundle(base, bindings)` was called in
normal, non-synthetic mode. It authenticated all nineteen preparation artifacts,
sixteen case records, and 418 journal rows representing exactly 209 ordered
STARTED/COMPLETE pairs. It checked source/text joins, immutable case order,
per-case input and mask hashes, exact masks and token lengths, thinking-off full
generation header, final input position, joint answer-boundary receipts,
KEEP/STOP IDs 50057/48964, A/B IDs 32/33, and ordinary gold IDs. No template pin
or adapter was patched. Every one of the twenty-one copied bundle files was
compared byte-for-byte with its original preparation, text-lock, or owner-closure
source and matched.

The retained result reports PASS, 209 attempted/completed/planned operations,
sixteen complete cases, zero failed or unrun operations, complete input
publication, no retry, and zero model loads/forwards/derivatives. Saved input
lengths independently range from 45 to 167, all below 320. Preparation elapsed
15.469 seconds; owner elapsed 16.562 seconds, within 175+5 seconds.

Independent closure checks required actual-worker authentication, assignment
before resume, owned job membership, creation-time and image/hash identity,
parent relationships, and no PID-only authority. Launcher, actual-worker, and
console-helper retained-handle exit proofs all have successful queries,
signaled handles, wait result zero, exit code zero, and no query/wait error.
Both drains reached EOF, joined, and match the retained stdout/stderr byte
counts; neither overflowed or reported an error. Pipes closed, the job was
empty before close, cleanup had no errors, termination was unnecessary, and
quiescent/within-deadline are true. The root-retained CLI final chunk has exit
zero and its printed summary matches the closure fields exactly.

The first review script passed adapter, copy, release/source/text joins and
closure predicates before stopping on a review-script-only config key typo
(`launch_image_sha256`). A follow-up used the actual `launch_sha256` and
`base_sha256` keys, completed the image/parent/time/root-CLI checks, repeated
normal saved-bundle validation, and exited zero. No candidate or saved evidence
was edited; this was not a preparation retry.

## Prospective authority and remaining prerequisites

The actual preparation release is explicitly one-shot offline preparation,
approved true, retry false, and model_authorized false. Its raw hash appears
identically in the owner command, owner admission/closure, preparation
admission, and root-retained CLI command. The source and text hashes also join
across those receipts; the preparation admission controller PID equals the
authenticated actual-worker PID, and recorded owner/preparation/cleanup times
have the expected order.

Read-only Git history shows preparation release commit
`6c0912d3d1425c9e0c45aa64ae2fe6c5bf8ddce9` before saved-attempt commit
`38ed46baa6ef119a83ec52a241ac5453aa577b8d`; `git merge-base --is-ancestor` for
that pair returned zero. Root separately reported its raw-Git check of the
twenty-five saved files; that inventory audit was not repeated here.

BUNDLE_BINDINGS approved/model_authorized flags are false, and no actual model
RELEASE.json existed when checked. Root still owns actual model source/runtime
admission and the separate release, including the unchanged full evidence
reservation and execution ceilings. The gate's real classification census,
baseline eligibility, conditional editor outcomes, and saved model judgment
remain unrun. Prior unchanged-source engineering proofs are reused.

This review consumed saved evidence only. It imported no tokenizer/model/provider,
read no checkpoint tensor or actual residual feature, performed no fitting,
network request, preparation rerun, or actual worker/model launch, and changed
no input or release. Only this review document was written.

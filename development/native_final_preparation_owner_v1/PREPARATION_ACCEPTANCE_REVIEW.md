# Independent actual-preparation acceptance review

PASS for acceptance of the exact, closed one-shot preparation below. No concrete blocker was found in the reviewed source/authority joins, 24 prepared inputs and receipts, operation accounting, storage partition, or retained-process closure. This is not a final-run approval, release, or authorization to load a model.

## Exact evidence

| Artifact | SHA256 |
| --- | --- |
| Preparation `SOURCE_FREEZE.json` | `247954f283b1a47b98e9067c2b01d41d836ef29576793066b42dd4beb912d3f6` |
| Execution adapter `SOURCE_FREEZE.json` | `26100ab792bcf2379cb7a4960bd5aa1a54f5e69477043bf132e4f09ee0540af3` |
| Preparation owner `SOURCE_FREEZE.json` | `1264e672cadc91ebe7c4757135178d2e6cb3072bf1b750d23ff7df63b87e246a` |
| Original `TEXT_LOCK.json` | `9a30298e81ba5930447810d4f8255e58e6ae4030eb4b70249d1a50094b177aeb` |
| Original `ROOT_APPROVAL.md` | `a8c70ea4fa096a6c78183f5131069156f0e5a8eb8a6a1e072520c04cef0b9c8d` |
| Original `PREPARATION_RELEASE.json` | `2aa06cc1062d91cb76a4a916124d068d5b73dc65d04ce57ad0bdef2a560a3bf0` |
| Preparation `ADMISSION.json` | `94a8e501b04fc519d0f80b68754081a0c3c45cd6a421710999521c870fc17f4c` |
| Preparation `inputs.json` | `908d1070918d9be0a4622a771d789a01995e0db9d116b9a9a8a7850e72f0720f` |
| Preparation `RESULT.json` | `608e6321b305e0cd8ce99f656e7416426b8031a59d62466ea5cee8457dc4f4ce` |
| Preparation `operations.jsonl` | `26151015e2aa450227901db0410d59bbd249dd91b79dce649cd9949495a608cf` |
| Owner `ADMISSION.json` | `22177f7a912003b45cdae377c1749a93ee6117b73719ee98314284a7f8c2973f` |
| Owner `CLOSURE.json` / bundle `PREPARATION_CLOSURE.json` | `82931b1c336b3e946c9d7ec572994e1d69817447a62b5f80d4103431c77b0c65` |
| Owner `stdout.bin` | `5b93df7f622059bd8ec8f729217aed194003ea80d798c04b409c78f134fc016d` |
| Owner `stderr.bin` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| Execution bundle `BUNDLE_BINDING.json` | `d761013bc6c4cf381ebf70d4dfebaf0a495ca2e31c83dc50127f3174f174d621` |

All source inventory bytes matched their respective freezes. The prior storage review remains the basis for the unchanged pre-write bound and ownership implementation; its SHA256 is `c873d33c14eb93b631a83e248e1bdc940e1c283c9e87463ebd4f5b2b30ab42e1`. Preparation, adapter, owner, policy, source, release, and text-lock bindings agree. The lock's canonical hash is `c391f75268443dfcdeee4aac76563aea9df37208d2b0a226df7e8732c94d0f88`; this is distinct from its raw-file hash above.

I compared all 29 bundle payload files byte-for-byte to the allowed originals: 27 preparation artifacts (inputs, result, journal, 24 receipts), TEXT_LOCK, and original owner CLOSURE under its bundle name. The non-approved binding pins all receipt hashes. It has `approved:false`, `real_model_authorized:false`, and verification-only purpose; no `RELEASE.json` exists in that bundle. No synthetic closure or substituted preparation evidence was used.

## Prospective authority and closure

Git blob bytes match the actual lock and approval at `667281c3b4bf9a1544be14d97eff5b40e69c48af`, committed 2026-09-09 01:37:49 UTC, and the release at `193fe99c082879805d49a8eb9921d77d1334e167`, committed 01:38:26 UTC. Both precede every retained Windows creation identity:

| Role | PID / parent | Creation UTC | Creation FILETIME |
| --- | --- | --- | --- |
| Launcher | 32804 / 19340 | 01:38:44.6423282 | 134333915246423282 |
| Optional conhost | 29888 / 32804 | 01:38:44.6532352 | 134333915246532352 |
| Actual base Python | 6916 / 32804 | 01:38:44.6687900 | 134333915246687900 |

The recorded images and hashes match the pinned `.venv\Scripts\python.exe`, base-runtime Python, and `C:\Windows\System32\conhost.exe`. Parent/creation ordering, live-before-exit authentication, job membership, suspended hidden launch and assignment before resume, and retained-handle provenance agree. All three exit proofs are valid, signaled, query-successful, and exit 0, with wait result 0. There is no PID-only termination authority in the reviewed closure path.

The exact owner command agrees across admission and closure; preparation admission identifies actual PID 6916. The owner reports one-shot PASS, no timeout, no termination request, no primary or cleanup error, empty job, quiescence, joined drains, EOF, and closed pipes. Captured stdout is 475 bytes, stderr is empty; capture counts match raw files and the stdout result agrees with RESULT apart from its additional lengths field. Root separately reported external exec session 40367 exiting 0; I rely on the retained artifacts, not an independently observed external session, for this review.

The absolute owner envelope is 175 seconds of work plus 5 seconds of cleanup, totaling 180. Recorded owner elapsed time is 14.640999999828637 seconds, with zero recorded cleanup duration; preparation elapsed is 13.265999999828637 seconds. Journal times are finite, monotonic, and inside the owner work deadline.

## Input, operation, and storage acceptance

The existing production `input_reader.read_bundle(bundle, binding)` accepted all 24 cases with default, non-synthetic scope `OFFLINE_FINAL_PREPARATION`. It checked raw hashes and source/lock/result/input/category/gold/boundary joins. Explicit independent checks also confirmed complete masks, full-input length and final-index equality, exact assistant header/end arrays, single-token label suffix receipts, and no truncation or thinking-mode substitution. This checks retained receipts; I did not tokenize again.

| Matched family | Self length, both orders | Other length, both orders | Nontermination control length, both orders |
| --- | --- | --- | --- |
| N01 | 156 | 160 | 168 |
| N02 | 160 | 164 | 167 |
| N03 | 162 | 166 | 174 |

KEEP is token 50057 and STOP is 48964 for all semantic items in both orders. Ordinary O01–O06 lengths are 66, 70, 42, 46, 63, and 65; their locked gold labels are A, B, A, B, A, B, mapping exactly to 32, 33, 32, 33, 32, 33. The final input token is 271 (`\n\n`); assistant header is `[248045,74455,198,248068,271,248069,271]`, with end `[248046,198]`. All masks contain exactly one per input token. All lengths are at most the frozen ceiling of 320. Category metadata retains six expected self-ON and eighteen expected other/control/ordinary-OFF inputs; these are prospective routing expectations, not measured gate outcomes.

The journal contains 626 rows: exactly 313 scheduled, ordered STARTED/COMPLETE pairs with corresponding fields, zero failed and zero unrun operations. There are 24 complete case receipts and inputs. The recorded preparation performed zero model loads, forwards, or derivatives. The inherited final scope remains 180 forwards, 48 derivatives, one load, 24 baselines, 12 cold self requests, 36 OFF requests, 1995 seconds, 288 MiB total and 5 MiB per artifact; this acceptance does not execute or approve that scope.

Preparation occupies 179,424 bytes of its 16,744,448-byte partition. Owner evidence occupies 4,708 bytes of its 32,768-byte partition. Their 32 original files total 184,132 bytes, within the unchanged 16,777,216-byte combined cap. Per-file and terminal RESULT limits hold. Closure's 180,547-byte pre-closure total plus its 3,585 bytes equals the independently counted total. The observed sizes supplement, rather than replace, the previously reviewed deterministic pre-write partition guarantee.

## Independent method and limits

Usage was checked before jobs (37% used). A single read-only inline audit ran under `.venv/Scripts/python.exe -B -I -c`, using shell `login:false`, and exited 0. It used stdlib inspection, raw hashing/Git blob comparison and the existing production reader; import guards rejected tokenizer/model providers, and audit guards rejected network and file writes. No tokenizer, model class, model execution, retry, scientific source change, input mutation, old sealed data, approval, release, or commit was performed. Only this report was created, outside frozen inventories, after validation. The previous blocked and storage reviews remain unchanged. Final-run authority remains exclusively with root.

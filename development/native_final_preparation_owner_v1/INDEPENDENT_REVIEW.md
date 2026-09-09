# Independent preparation owner review

**BLOCKED for real preparation admission.** The focused Windows ownership checks passed, but the required universal combined preparation-plus-owner 16 MiB write guarantee remains unresolved. The final reviewed candidate explicitly disables production admission before reading any real release or launching a child. This report grants no approval or release.

## Exact reviewed and tested identity

| Artifact | SHA256 |
| --- | --- |
| `SOURCE_FREEZE.json` | `bd5361f5171b7123b6367598fd3f9b917e35f74d1c5784f63b9a0171d87b6eb1` |
| `owner.py` | `fc151ef0ddd2d4bdafb76de3fe1148f9f787dda908d65d3db21f19676c30ad91` |
| `windows_job.py` | `a6334fea1678773ff169aed39c3b1939c57614d0096a446e6d0418abcb21368a` |
| `test_owner.py` | `cc6dcfea004702ecf494fa5562db7e86ce7614c9fe7c45efb4789ff44e8b79fb` |
| `CONTRACT.md` | `4812a8cefd600c2fff0a2c90dd5c5f098932f0f5ebaad509d2711d02882d6e22` |
| `fake_child.py` | `74060ecff472ea1f0b1c13c685c4d7ae301cb0f0312a7d5f1015101ad3abc44d` |
| `OWNED_IDENTITY.json` | `551be5db9d79d9ae4a9b64fb2c7666652baa3d35fc15bb23b9c579eb596720ec` |
| Independent `TEST_RESULTS.json` | `05bb091969a39d79698a3ebf0125994a88ac8ea1e10703ae328da0588a1facca` |

The candidate hash matched the worker's declared final blocked candidate before testing and remained unchanged afterward. The test's `verify()` checked all frozen local and external bytes. The immutable preparation freeze remains `bbc9a2587a766e1c1c3260d34add7c621ce32e030ae23b9e141e1b3d52b9feef`; its `prepare_core.py` remains `1af14fb9a43e41fe47aa54a96d5cdab0f904394dd9353be301938e309f98f026`. The reused retained-handle native primitive and identity probe remain `f3433f9ea23de28d524e853b5943d85b5c078587806a662eb8d0b5619e04a005` and `2eebb8671603837bfde1c7381406379220af941d0a57f5affb68011de8728e56`. Their previously reviewed ownership proof is reused; no broad certification suite was repeated.

## Blocking issue

The unchanged preparation publisher checks its own outputs against 16 MiB, reserving 64 KiB until a critical terminal RESULT write. Its failure serialization has no fixed error-text length bound. The owner's additional bytes are not charged by that publisher. The owner's 32 KiB reservation, periodic combined-size observation, and subsequent 8 KiB RESULT check do not themselves prove that every write stays within the combined 16 MiB ceiling; they can refuse evidence after publication. A deterministic bound covering terminal failure serialization or a prospectively approved storage-accounting change is required before real admission. Passing the small fake-output cases does not resolve this issue.

The candidate correctly records this limitation in `CONTRACT.md` and sets `COMBINED_STORAGE_ADMISSION_READY=False`. `main()` checks it before owner verification, real release/text reads, the exclusive production attempt, or child creation. The independent suite confirmed that even syntactically supplied source/release hash arguments produce admission failure and no production attempt. Do not remove that guard solely because the ownership tests pass, enlarge the cap, discard failed evidence, or treat this report as production authorization.

## Focused ownership findings

The launcher uses CREATE_NO_WINDOW plus CREATE_SUSPENDED and is assigned to the owned non-breakaway, kill-on-close Windows Job before resume. Its original Popen handle is retained. The actual base-Python child must be observed live and authenticated by retained handle, owned-job membership, direct launcher parent, creation order, and pinned path/image hash. Only one additional pinned System32 conhost helper is permitted; it receives the same parent/creation/job checks and a retained exit proof. Unexpected processes cannot supply a successful closure. Cleanup acts on the owned Job or an original retained process handle, never a bare PID.

I identified a pre-assignment failure path in the initial moving candidate that could leave an unassigned suspended launcher alive. The worker repaired it before this final freeze and added a direct wrong-launcher-hash test. The independent run observed retained launcher termination with exit 125, an empty job, no fake preparation directory, and no claim of actual-worker authentication/quiescence. Candidate handles are also retained for closing if identity inspection raises before role assignment.

Production defaults remain a 175-second wait and one shared cleanup window of at most five seconds, bounded by the original 180-second absolute deadline. Exit observations and drain joins consume the remaining shared time rather than resetting it. The short fake tests exercise the same deadline calculation with reduced limits; they are not a 180-second soak test. Captures retain at most 8192 bytes each, continue draining toward EOF, and signal overflow/read failure. Quiescence requires successful retained exit proofs, job emptiness, EOF, joined drain threads, and closed pipes. Nonzero exit, timeout, capture overflow, incomplete closure, or a failed preparation RESULT cannot become owner PASS merely because termination later succeeds. An exclusive owner directory prevents retry and evidence overwrite.

## One independent test run

After checking the exact candidate hash, ran once with shell `login:false`:

```powershell
.venv/Scripts/python.exe -B development/native_final_preparation_owner_v1/test_owner.py
```

Exit code **0; nine groups PASS**. Evidence is retained under `test_evidence/owned_1788914696514413200/` and the exact `TEST_RESULTS.json` hash above.

| Fake case/check | Independent observation |
| --- | --- |
| Natural success | Actual child, launcher, and conhost exited 0; owner PASS; no termination needed. |
| Nonzero | Actual child and launcher exited 7; owner FAIL despite complete quiescence. |
| Timeout | Original shortened deadline triggered; owned-job termination succeeded; all three retained exits were 125; owner FAIL. |
| Stdout overflow | 20,000 bytes observed; only 8192 retained; all retained exits observed; owner FAIL. |
| Stderr overflow | 20,000 bytes observed; only 8192 retained; all retained exits observed; owner FAIL. |
| Exclusive one-shot | Retry rejected before new work; previous closure bytes unchanged. |
| Pre-assignment wrong image hash | Suspended launcher terminated through its retained handle; exit 125; fake work never started. |
| No release arguments | Entry disabled; no real child. |
| Storage-blocked production arguments | Admission refused before real release access or production attempt. |

For all five active fake cases, the actual child, launcher, and optional helper were retained/authenticated, the job was empty, both drains reached EOF and joined, pipes closed, and the reported shortened deadline was met. All observed test outputs met the per-file and combined storage checks; this is observed test evidence, not the missing universal write-bound proof.

All review shell calls used `login:false`. I used only fake children, source/identity pins, and retained fake-test evidence. No actual cohort, tokenizer, model, real preparation, real release, old sealed data, or real study outcomes were accessed or launched. No source change, release, approval, or commit was made by this reviewer. This report is outside the frozen source inventory.

# Oracle confirmation: archival reproducibility handoff

2026-09-09. This is an archive/verification guide, not permission to rerun the
closed preparation or model attempt. No judge, model, tokenizer or fixture was
executed while writing this note. The [results report](/C:/Users/farha/OneDrive/Documents/ChatGPT/SP_Lense/docs/ORACLE_CONFIRMATION_REPORT.md)
and [methods note](/C:/Users/farha/OneDrive/Documents/ChatGPT/SP_Lense/docs/NATIVE_ORACLE_METHODS_NOTE.md)
define the restricted claim; the [related-work note](/C:/Users/farha/OneDrive/Documents/ChatGPT/SP_Lense/docs/ORACLE_CONFIRMATION_RELATED_WORK.md)
places it among established methods, without an algorithmic-novelty claim.

## Archive entrypoints and anchors

Paths below refer to this repository. Commit IDs anchor historical bytes; SHA256
values refer to raw file bytes, not pretty-printed JSON. Preserve line endings
and the existing path-local Git attributes. Do not regenerate frozen manifests.

| Layer | Exact entrypoint | Anchor |
| --- | --- | --- |
| Whole authored v2 cohort | [SUBMISSION.json](/C:/Users/farha/OneDrive/Documents/ChatGPT/SP_Lense/development/native_oracle_confirmation_cohort_v2/SUBMISSION.json), [content admission](/C:/Users/farha/OneDrive/Documents/ChatGPT/SP_Lense/development/native_oracle_confirmation_cohort_v2/ROOT_COHORT_ADMISSION.md) and its linked reviews | Submission commit `809a7dc02152f326726aac33221a89909a0ae2d4`; admission commit `86aa630edf418e50e207e453144a3e41e1644c75` |
| Text, renderer, method, analysis and environment lock | [sealed TEXT_LOCK.json](/C:/Users/farha/OneDrive/Documents/ChatGPT/SP_Lense/development/native_oracle_confirmation_preparation_v1/root_release/TEXT_LOCK.json), [lock acceptance](/C:/Users/farha/OneDrive/Documents/ChatGPT/SP_Lense/development/native_oracle_confirmation_preparation_v1/ROOT_TEXT_LOCK_ACCEPTANCE.md) | Commit `e437b33`; SHA256 `bc48bc37375f9b9dbd3c11c17ab099cbb13e8122b66e5982ae2d348727fcc5e5` |
| Tested engineering bindings | Preparation, execution and preparation-owner source manifests listed below | Commit `f83d8328135ee1660f9175844101ea811b582c5b`; original focused engineering and metadata review receipts remain separate |
| Preparation authority | [PREPARATION_RELEASE.json](/C:/Users/farha/OneDrive/Documents/ChatGPT/SP_Lense/development/native_oracle_confirmation_preparation_v1/root_release/PREPARATION_RELEASE.json) | Commit `9ae9cad`; SHA256 `bde6460d34dcf2bc8f5a502f268af16c2b0d30bf1302563a9ab78379163c049a` |
| Original preparation and retained ownership | [RESULT.json](/C:/Users/farha/OneDrive/Documents/ChatGPT/SP_Lense/development/native_oracle_confirmation_preparation_v1/preparation_attempt_001/RESULT.json), sibling operations/case/input files; [CLOSURE.json](/C:/Users/farha/OneDrive/Documents/ChatGPT/SP_Lense/development/native_oracle_confirmation_preparation_owner_v1/ownership_attempt_001/CLOSURE.json) and sibling owner records | Commit `9556c0af2d37d9262b2e4805650a66d0f2c93a43`: 32 files, 183,352 bytes; 313 operations, 24 inputs, lengths 45–170, zero model calls |
| Accepted execution input bundle | [execution root_release](/C:/Users/farha/OneDrive/Documents/ChatGPT/SP_Lense/development/native_oracle_confirmation_execution_v1/root_release), containing exact TEXT_LOCK, PREPARATION_CLOSURE and 27 preparation payloads | Copy commit `cd1cd0555ba3c461fdaf4b602732e0bed368eae4`: 29 payloads, 313,003 bytes; prepared inputs SHA256 `d8f75731b351f50590c33588fe0ebc42c6dcc59de2669055173a45a1b60bd543` |
| Model-attempt authority | [RELEASE.json](/C:/Users/farha/OneDrive/Documents/ChatGPT/SP_Lense/development/native_oracle_confirmation_execution_v1/root_release/RELEASE.json) | Commit `b92f6bfa9ad49b06caa73ceb3b67256fe7d31476`; SHA256 `e6a2d89fb916fccd1aa42f72afeb0713ca43518e6c59efdc1c4127d48232b69e`; preflight commit `32e37dee` |
| Closed model evidence | [attempt directory](/C:/Users/farha/OneDrive/Documents/ChatGPT/SP_Lense/development/native_oracle_confirmation_execution_v1/real_evidence/native_oracle_confirmation_attempt_001) | Commit `93e5e88b09bf38dd6c34d5830273e13e9e4e3746`: 466 files, 107,701,733 bytes |

Source manifest SHA256 anchors:

- [Preparation SOURCE_FREEZE.json](/C:/Users/farha/OneDrive/Documents/ChatGPT/SP_Lense/development/native_oracle_confirmation_preparation_v1/SOURCE_FREEZE.json): `fe7f0b41a4762572e24ede88fa4ce42dbf1ee200b4f78094e14cc038cc96d78c`.
- [Execution SOURCE_FREEZE.json](/C:/Users/farha/OneDrive/Documents/ChatGPT/SP_Lense/development/native_oracle_confirmation_execution_v1/SOURCE_FREEZE.json): `172edd056bfb14a4029c50027c0f8753cbd865ebf76701bb4f21269883a46c95`.
- [Preparation-owner SOURCE_FREEZE.json](/C:/Users/farha/OneDrive/Documents/ChatGPT/SP_Lense/development/native_oracle_confirmation_preparation_owner_v1/SOURCE_FREEZE.json): `3a78ba42ac71f3b3388a27bcc69c8e38af9275a50fa4d4bb526a7f743759e780`.

Within the actual attempt, start with CLOSED_WORKER_BINDING.json, WORKER_RESULT.json,
AUDIT_RESULT.json and PARENT_FINAL.json, then their linked rows, full float32 logits,
steps, traces, loader/admission and owned-process records. The closed binding is
the worker-evidence inventory, not a substitute for the full 466-file Git archive.
AUDIT_RESULT SHA256 is `bef23b93c8c2212a86b3dec0d1c2ccbeb9f8e50ed5df7644d0e16d309ff5e398`;
WORKER_RESULT is `ef148f6dfacabc6254f1c239a0824b1dda294a98df97c7b263c2efbf8ee935d3`.

The [independent preparation review](/C:/Users/farha/OneDrive/Documents/ChatGPT/SP_Lense/development/native_oracle_confirmation_preparation_v1/INDEPENDENT_ACTUAL_REVIEW.md)
has SHA256 `34bf25fa715c0ffe27a3f8dc8c0d282163f069580e302186079209a440cdcd62`.
The now-written [independent model-evidence review](/C:/Users/farha/OneDrive/Documents/ChatGPT/SP_Lense/development/native_oracle_confirmation_execution_v1/INDEPENDENT_ACTUAL_REVIEW.md)
has SHA256 `45d3ee774974f2d7d7a46faf468739e8a4d47762a9c4909feaee33f3a14c902b`.
It records one model-free judge call, 34.843 seconds, exact equality to the saved
audit, and unchanged evidence inventory. Root acceptance and the final documentation
commit must be recorded separately; this note does not invent that commit.

## Saved-evidence verification, not scientific replay

The existing callable is `audit_saved.judge(base, execution, deadline)`.
`production_run.audit` additionally writes AUDIT_RESULT and requires live admission;
do **not** invoke that production lane for archival checking. The following
read-only invocation follows the independent review's successful core call and
guards, uses the existing judge and production input/source checks, and prints
a result without writing an audit receipt. It is documented, not run here.
Run only on an intact, trusted archive at the pinned checkout; do not weaken a
failed binding or substitute synthetic inputs to make it pass.

```powershell
# PowerShell without profile; from the original repository root.
& .\.venv\Scripts\python.exe -B -I -c @'
import json, os, sys, time
from pathlib import Path
class Deny:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','transformers','tokenizers','transformer_lens','datasets','pyarrow','safetensors'}:
            raise ImportError('PROVIDER_IMPORT_FORBIDDEN')
sys.meta_path.insert(0, Deny())
def guard(event, args):
    if event in {'socket.connect','socket.bind','urllib.Request','subprocess.Popen'}:
        raise RuntimeError('NETWORK_OR_CHILD_FORBIDDEN')
    if event == 'open':
        if any(c in str(args[1] or '') for c in 'wax+'):
            raise RuntimeError('WRITE_FORBIDDEN')
        if str(args[0]).lower().endswith(('.safetensors','.pt','.pth','.ckpt')):
            raise RuntimeError('CHECKPOINT_TENSOR_READ_FORBIDDEN')
sys.addaudithook(guard)
root = Path(r"C:\Users\farha\OneDrive\Documents\ChatGPT\SP_Lense")
engine = root / "development/native_oracle_confirmation_execution_v1"
sys.path.insert(0, str(engine))
approved = "e6a2d89fb916fccd1aa42f72afeb0713ca43518e6c59efdc1c4127d48232b69e"
os.environ["SP_NATIVE_RELEASE_SHA"] = approved
from authority import read_release, execution
from audit_saved import judge
release = read_release(approved)
base = engine / "real_evidence/native_oracle_confirmation_attempt_001"
saved = json.loads((base / "AUDIT_RESULT.json").read_bytes())
started = time.monotonic()
observed = judge(base, execution(release, approved), started + 180)
if time.monotonic() - started >= 180 or observed != saved:
    raise SystemExit("SAVED_AUDIT_MISMATCH")
if not observed['scientific_pass'] or observed['scientific_failures'] or observed['technical_failures']:
    raise SystemExit('SAVED_AUDIT_NOT_PASS')
print("EXACT_SAVED_AUDIT_MATCH", observed["classification"], observed["scientific_pass"])
'@
```

This imports stdlib-based audit/input/scoring code, not a model provider. `-B`
suppresses bytecode writes; the called paths read files and return data. It has
the existing judge's cooperative 180-second allowance, not a new retained-child
hard-timeout guarantee. The inline guards mirror the review's blocked provider,
network, child-process, Python-mode write and tensor-file paths; they are not an
OS sandbox. The separate raw/Git/ownership checks are not reproduced by this
short invocation.
Judge equality checks the saved calculations, schedule, stop rules, identities and
claims. It does not newly compute transformer gradients or recreate process history.

## Dependencies and portability boundary

The original runtime is 64-bit Windows, Python 3.12.14 (MSC v.1944), Torch
2.13.0+cpu, Transformers 5.15.1 and tokenizers 0.23.0rc0. Exact files, not only
version strings, are pinned in [DEPENDENCIES.json](/C:/Users/farha/OneDrive/Documents/ChatGPT/SP_Lense/development/native_oracle_confirmation_preparation_v1/DEPENDENCIES.json),
[TOKENIZER_PINS.json](/C:/Users/farha/OneDrive/Documents/ChatGPT/SP_Lense/development/native_oracle_confirmation_preparation_v1/TOKENIZER_PINS.json),
[CHECKPOINT.json](/C:/Users/farha/OneDrive/Documents/ChatGPT/SP_Lense/development/native_oracle_confirmation_execution_v1/CHECKPOINT.json)
and [OWNED_IDENTITY.json](/C:/Users/farha/OneDrive/Documents/ChatGPT/SP_Lense/development/native_oracle_confirmation_execution_v1/OWNED_IDENTITY.json).
The model revision is `2fc06364715b967f1860aea9cf38778875588b17` of
Qwen/Qwen3.5-0.8B; model execution used native CPU float32/eager, text only.
The launcher SHA256 is `038b13f7d1f8bd011290eb0044fd1c133746082929dbef567d1a566cf25598df`;
the retained base Python image SHA256 is `b7a12c3af0b4db44191eec14ea095eba731b7328917f570806183093d19ddca2`.

Historical scientific/scoring/gate and ownership files referenced by the manifests
must accompany the archive: the new namespace alone is insufficient. Source and
runtime bindings include absolute local paths. Model/tokenizer files remain in
the pinned local cache rather than being assumed bundled in Git. No portable
installation or clean-machine reproduction has been demonstrated. Do not retry
blocked PyArrow, import the legacy wrapper/datasets stack, change Windows security
policy, install alternate dependencies or fetch replacements under this guide.
The original native import guards and offline restrictions remain binding.

## What is reproducible now, and remaining step-10 gaps

- **Now:** with the complete unchanged repository dependencies, release bundle
  and raw archive at the bound paths, a reader can inspect every case, verify
  hashes and reconstruct the saved judge result without encoding or model calls.
  The reviewer has done this once. Repeating a saved-data calculation is not a
  fresh confirmation experiment or independent model replay.
- **Portable archive:** no tested clean-machine package or relocation procedure
  exists. A third party may lack the absolute-path runtime/source dependencies.
  Public distribution, dependency availability/licensing and byte-preserving
  transport still need explicit checking before claiming portable reproduction.
- **Claim/report closeout:** independently reviewed raw evidence does not replace
  independent review of the final prose, tables, provenance and limitations.
  Root acceptance and a stable committed report/archive index remain distinct.
- **Scientific scope, not packaging defects:** there is no matched comparative
  algorithmic baseline or population-level statistical design. Three authored
  families are not 24 independent family samples. All fresh flips were
  first→second; natural first-target flips remain untested here. The learned
  gate missed two self inputs; oracle-OFF identity is not automatic gating or
  broad capability preservation, and ordinary accuracy remained 5/6.

These limits do not authorize more experiments or constitute a publication-ready
verdict. No bitwise cross-platform, arbitrary-workload, global-arrow, intrinsic-
motive, novel-method or journal-quality claim follows from archival PASS.

# Independent text-lock review

2026-09-09. PASS for the bounded, model-free text/source-lock review. No blocking
finding. This review grants no preparation or model-execution authority.

## Exact reviewed identities

| Artifact | SHA256 |
| --- | --- |
| TEXT_LOCK.json | 846381590f3a093db1f64cf29dd024cb98e6ceed317014dd52cb146f9a9b0ac4 |
| preparation SOURCE_FREEZE.json | 7d2fa6660c1d240c2541b6af93a14ab3d6245856a84cf265874d1234f6e5cf92 |
| evaluation SOURCE_FREEZE.json | 7bde258987d4ee1e2df5f4f5264d25babc04dd7792f787773fe02fcb93a713db |
| immutable SUBMISSION.json | 9b06eda1f293acd764a65cb742d9f2003ab980271065a927a49a8dbc4cb4902e |
| independent BLIND_CONTENT_REVIEW.md | 8dd7831d91cf020d0d18c93fadef3f36df6c4169568331731de434fc09a71b4e |
| frozen fitted gate | fab7d797f9424d80aa3873eefc1cedf0e438e5b4ddc3a1059a080114aff7cca1 |

The lock records source commit
`b8499ce72093154d5d2546eae97a37d2ae3621eb`. Root separately reported its raw-Git
source verification; this review did not repeat that inventory audit.

## Observed checks

Ran one PowerShell here-string through `python -E -S -B -c $textReviewScript`
from the preparation namespace; exit 0. Called
`prepare_core.validate_text_lock(lock)` with its normal, non-synthetic validation
mode. This does not execute preparation or instantiate a tokenizer.

The validator authenticated both prospective evaluation and preparation-owner
source joins, their listed local/external source bytes, and the current
preparation manifest. Both bindings name the supervised evaluation namespace
and the final evaluation manifest above. The final execution binding also names
the exact preparation manifest. Source inspection confirmed this validation
precedes the first `load_native_tokenizer` operation. Independently changing
either join's manifest hash to zero was rejected with
`SOURCES_FIXED_BEFORE_FIRST_TOKENIZER`.

The locked cohort equals the immutable first submission. Every locked prompt
equals the current authenticated renderer output; all sixteen UTF-8 hashes
match the independently reconstructed blind-review hashes in exact order:
G01 then G02, self/other/nontermination, KEEP-first then STOP-first, followed by
O01 through O04. The shared continuity/transition action wording is unchanged.
The ordinary proofs independently recompute to 43, PEBBLE, Tomas, and [copper],
with gold labels A, B, A, B. Scoring metadata does not enter the renderings.

The fixed preparation plan contains exactly 209 operations. The lock retains
sixteen baseline inputs, thirty-two policy requests, eight self cold endpoints,
twenty-four OFF identities, one model load, 120 forwards, and 32 derivatives,
with no smoke forward. Model/revision, 320-token ceiling, time limits, and
storage limits equal the authenticated fixed plan.

Conditional census semantics are joined to the previously independently
reviewed, unchanged `workflow.py`
`5fb27dc215a3d2bc569462964f1226856131acd59eb0a8bbb8d90e5987bd9de3`
and `audit_saved.py`
`e44af5e58b38f166acfbdc2de795e4954852d1f76e298ccff235f057686eeaca`:
all valid sixteen baselines are retained before judging the census; labels judge
routes; any routing error blocks policy requests and derivatives; later fresh
entries must reproduce baseline score/route/identity. The frozen gate adapter
remains `97ecde35553535e3910e3754e375b0605ba4e336535b1271b1c4ddefdac670b8`.
This review reuses that source-bound engineering evidence and makes no new
full-suite claim. In particular, the earlier independently observed eleven-test
run and the separately accepted final bridge test remain compositional evidence,
not a twelve-test run attributed to the earlier test-file hash.

## Authority and limits

Both `actual_preparation_authorized` and `actual_model_authorized` are false.
The actual PREPARATION_RELEASE.json and evaluation root RELEASE.json were
absent when checked. Text-lock/content approval flags do not constitute either
release. No tokenizer/model/provider imports, tensor reads, feature scoring,
fitting, network use, worker/experiment launch, or real preparation occurred;
the forbidden-import assertion passed. Only this review document was written.

Root must separately issue any future preparation or evaluation release. Actual
token boundaries, encoded lengths, gate routing, eligibility, and endpoint
behavior remain untested by this text-only review.

# Fresh confirmation token binding v2: COMPLETE

All24 frozen prompts passed exact offline token binding and the unchanged160-token maximum. Five focused alphabet/layout regression cases passed. The single tokenizer stage took8.328s (process wall9.786s), one tokenizer load, zero model loads/forwards/derivatives or gate scores. Fresh standard usage was12% before both batches. Production remains disabled; no behavioral result or publication credit is claimed.

The sole behavioral adapter correction separates display-ordered answer_words from canonical token mapping. Semantic metadata must match its declared layout while KEEP=50057/STOP=48964 remains fixed. Ordinary metadata stays exactly [A,B], mapped32/33. Parent text, author truth objects, helpers and criteria were unchanged. All24 proofs were recomputed from scratch; identical deterministic bytes for a previously measured prompt are not evidence reuse. V1 remains immutable INCONCLUSIVE.

| Semantic group | KEEP-first tokens | STOP-first tokens |
|---|---:|---:|
| N01 self |149|149|
| N01 other |147|147|
| N01 control |150|150|
| N02 self |146|146|
| N02 other |144|144|
| N02 control |147|147|
| N03 self |149|149|
| N03 other |147|147|
| N03 control |149|149|

Ordinary O01–O06 lengths respectively: **50,49,44,50,63,66**. Every row has boundary and length PASS; none is unrun. Full input IDs, attention/final-input masks, template hashes, exact appended-answer prefixes and token roles are authenticated in tokens_01.json through tokens_24.json.

Source freeze: `ca7f831d52bbfdc73e55f564dff576fd52e89943`, SHA `0011c2154ec4ba393dcf361d615344dcddc2597ca05f2887a8cd4a8bd4e50af1`. TEXT_LOCK remains at `f15af48aaa54995c593630a67986199bd7a9644d`, SHA `371562bd76cdc0f105fb1324f1de28fa11eb44bff191c1d06c883cb98a9c456d`; EXACT_INPUTS SHA `f62399de185341ff9380318f390b17c271de7480b0ce7ca1c22c85e5e6a94785`.

INPUT_LOCK binds those unchanged24 prompts/48 requests,24 fresh token records, sources and receipt. Five ordinary proofs remain machine-exact; O06 remains machine-encoding UNVERIFIED with explicitly bound independent manual support. Hash novelty coverage remains partial, not a global claim.

Next dependency is the already separately scoped fake-only runtime integration using this committed lock. No real model release is supplied here.

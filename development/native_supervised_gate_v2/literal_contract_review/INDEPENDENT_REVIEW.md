# Independent packet-contract delta review

PASS. Bounded review on 2026-09-09, starting 15:46:19 UTC, completed within
the authorized three minutes. This is prospective engineering before any real
release, not an experiment retry or model result.

The author packet's uppercase contract states: "proof.inputs literal string,
exact uppercase transformation." It imposes no letters-only restriction. The
removed `ASCII_LETTERS` assertion therefore enforced an additional constraint
absent from that prospective contract. Accepting printable literals containing
spaces, digits, and hyphens while applying the exact `.upper()` transformation
matches the stated contract. String/type, printable ASCII, exact input keys,
proof value equality, and exact matching gold-option checks remain present.

Byte comparison against the earlier exact-copy review snapshot proved that
`prepare_v3/validate.py` changed by deleting exactly this one assertion:

```python
need(bool(re.fullmatch(r'[A-Za-z]+',value)),'ASCII_LETTERS')
```

The other eight preparation Python files compared to that snapshot are
byte-identical. Capture `input_reader.py` and gate `source_auth.py` differ only
in their dependent literal source hashes; their function bodies are unchanged.
Only `input_reader.py` changed in the capture source-hash map. The updated
preparation-to-capture-to-gate freeze joins and all 121 local source pins were
verified. This review did not repeat the root's separate 11-external-pin check.

The focused suite was run exactly once:

```text
python -B development/native_supervised_gate_v2/literal_contract_review/test_literal_contract.py
Ran 5 tests in 0.000s
OK
```

The five groups cover uppercase preservation of nonletters, retained rejection
of control characters/non-ASCII/non-string inputs, bracket restrictions,
exact proof keys, and unchanged addition/oldest calculations. The independent
source/pin comparison also exited 0 with
`PASS_EXACT_ONE_LINE_CONTRACT_DELTA`, one removed assertion, and 121 verified
local pins. No blocking defect was found in this exact change.

Current verified SHA-256 values:

- Preparation freeze: `1b094732e4dc8ab60abd84a3f2a8bebffc08c42447497bfde196e00167bf6de5`.
- Capture freeze: `090c982f27d3c9720f309756979f98252bac9d4647cf2871f1dab78a336ca796`.
- Gate freeze: `32720dfc6747fcfb21f6935837116fc94134558608ba55806ad76bdc21d071df`.
- Changed validator: `3644e60f8231e9d2c98ed0b84d06a85d939cb514fb8cef81fb1b948c9daa01e8`.

Only BRIEF.md, engineering sources, existing independent proof metadata, and
the artificial literal tests were inspected. Authored submission, attestation,
and real outcome payloads were not read. No broad suite or unmocked admission
fixture was rerun; no model/tokenizer/provider operation, feature capture, fit,
production source edit, release, or commit occurred. Prior unchanged admission,
native, ownership, and numerical proofs were reused within their stated limits.

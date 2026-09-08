# Root approval: blind authoring only

Approved on 2026-09-08 after independent review and root byte verification.

- Reviewed packet commit: `4cebca3486a8a1367ebb349fc322c9831d708db0`.
- PACKET_FREEZE.json SHA-256: `95e5368c4d54caedcd56a4bdfba8fe5e94a22538c182b754dcb1b357e8bf8020`.
- Independently executed fake-marker/proof/extraction checks: 17 PASS, exit 0.
- Root verified all 11 tracked files against exact Git bytes and all eight frozen file records.

This approval permits one clean-context author to read only the four files in
the immutable author_packet allowlist and compose the single initial submission
specified by BRIEF.md. It does not authorize model loading, tokenization, access
to old questions or outcomes, alternate candidate pools, or any real experiment.
The author uses GPT-6 Astra with Ultra reasoning, as requested by the user.

The author writes SUBMISSION.json and AUTHOR_ATTESTATION.md outside author_packet.
The first whole submission must be preserved and committed before any validation
or semantic feedback. A separate outcome-unexposed reviewer then checks it under
the fixed authoring contract. Only the contract's minimal, logged pre-lock
structural or proof corrections are permitted; no outcome-guided selection or
replacement is allowed. Mechanical validation alone is not semantic admission.

The original packet and its historical status flags remain unchanged. Final text
locking, bounded offline preparation, and any real-model release require later,
separate root verification and authorization.

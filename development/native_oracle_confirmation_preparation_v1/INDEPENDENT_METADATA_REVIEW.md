# Independent prospective v2 metadata-rebind review

Verdict: **PASS, metadata-only engineering.** This closes the v2 path/authority-binding gap identified in the prior engineering review. It neither admits actual cohort content nor authorizes tokenization, preparation, execution or a scientific success claim. Rejected cohort v1 and all prior evidence remain unchanged.

Reviewed the exact delta from engineering commit `77f26c1190fa9a07c675725a1386d4daff13d74c`, the full v2 ROOT_SUCCESSOR_AUTHORIZATION.md, METADATA_REBIND.md and the focused test. The operative CONFIRMATION now names `development/native_oracle_confirmation_cohort_v2`, retains arm `fresh_oracle_confirmation.v1`, and binds root authority SHA256 `d73ef37a668c5dcd5653277acce89e4991de0ffb8de9b0d46384d9f209edc461`. Preparation dependencies bind that packet, including BRIEF SHA256 `9d25b5c7e19f25ba994d11f1d6160f0048763b67da5f6f84228f43796451a937`; schema, renderer and validator bytes are unchanged.

## Exact tested identities

| SOURCE_FREEZE.json namespace under development/ | SHA256 |
| --- | --- |
| native_oracle_confirmation_preparation_v1 | fe7f0b41a4762572e24ede88fa4ce42dbf1ee200b4f78094e14cc038cc96d78c |
| native_oracle_confirmation_execution_v1 | 172edd056bfb14a4029c50027c0f8753cbd865ebf76701bb4f21269883a46c95 |
| native_oracle_confirmation_preparation_owner_v1 | 3a78ba42ac71f3b3388a27bcc69c8e38af9275a50fa4d4bb526a7f743759e780 |

Independently ran once, PowerShell `login:false`, on 2026-09-09:

`.venv/Scripts/python.exe -B development/native_oracle_confirmation_preparation_v1/test_metadata_rebind.py`

Closed exit 0, **11 groups PASS**. Test SHA256: `cc9e333b28de26f82c51de28e0bd8320c7c640f9883ae543ed680a1498be0ec2`. Result `METADATA_REBIND_TEST_RESULTS.json` SHA256: `b435a8d4ba2e99a5bd4a2c3249f3555fdac0e001df3a66f44e7522aab0b000c2`.

All local source pins and execution's 143/owner's 11 external pins matched before the run. Shared plan/prep_plan bytes match SHA256 `3c33c2447e8dc30dc7cf98dc764d7221a44b0f43e819df95af5b694cde41b5ed`. Actual preparation-to-execution/owner source joins validate without circular hashes or future submission dependencies. Matched artificial v2 text/input bindings pass; old-v1 namespace, old/wrong authority, wrong arm, substituted preparation source and old oracle authority reject. The map remains exactly six ON/eighteen OFF.

Seventeen execution scientific/workflow/proof files match the prior commit byte-for-byte. Existing independent preparation 7-group, execution 26-group and owner 3-group receipts retain hashes `707d4794987cd7d44353acef319d56f3e04d98e200e05c060629b8ffc7b228ab`, `b14e8114c2a4b92e916be9bd0417b59dc42ac5549659460314aa2d4ccf2a4466` and `ed7e6b162d008c0e741d7fe9729bfa91990f98811785ee5918edb3d57fd5572a`. Prior review remains `1fab885b5775adfe860fefe9ec1d2cf13e79c108bdfc8c52fcfcc43417b79fc3`. These are reused historical proofs, not reruns or receipts for newly claimed real behavior.

No scientific recipe, learned/oracle separation, schedule, numerical gate, identity check, ownership code, time/storage bound or no-retry rule changed. No actual submission/prose was read. No provider/model/tokenizer was loaded, full workflow or owner suite rerun, release/real attempt created, or source edited by this reviewer. Only the focused test receipt and this new review were written. Actual content review, full pre-tokenization text/source lock and separately approved releases remain required; both production release files were absent at review close.

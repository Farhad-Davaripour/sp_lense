# Fail-closed hook component — PASS (pure fixtures only)

All six prospectively frozen groups passed in **3.234 seconds** (command wall time 3.652 seconds, exit 0). There were **0 model/tokenizer calls** and **0 arbitrary exception str/repr calls**. The 139 synthetic fixture files total 251,177 bytes, below the 32 MiB combined fixture cap. Fresh standard usage was 15%.

| Fixed group | Observed check |
|---|---|
| Complete normal reconstruction | All 109 checks authenticated; independent saved-byte judge COMPLETE. Final index 3,168 bytes, prospectively bounded at 3,169. |
| Allocation/count boundaries | Exact fit accepted; one extra byte and check 110 rejected. |
| Fitting identity mismatch | Terminal latch observed before codec; complete changed metadata retained; exact remaining IDs preserved. |
| Overflow/codec expansion | Raw, compressed-file and aggregate limits rejected; 1,048,576 raw bytes encoded to 1,048,902 bytes, with allocation rejection. |
| Partial/failed closeout | Partial chunks/manifests/index retained; fault-receipt failure stayed terminal. Even a fully written complete index was rejected after IO acknowledgement failure and lost fault receipt. |
| Independent saved rejections | Forged completeness, missing/changed artifacts, untracked bytes, incomplete index and indexed-but-unreferenced bytes all rejected. |

Source commit: `105c10e7b9b042894bff5834fc3a1500f7c80a91`. Freeze SHA: `c3398a4678b8f011bbf184dab23b76a68ef3b5970a2ddcb639bcbf8f994bba50`. The six pinned decision/guard/IO/resource inputs were checked against both committed and working bytes before this one batch. No batch retry occurred.

The production interface is in INTERFACE.md. It requires one shared permanent dispatch latch, the existing writer's injected IO ownership, complete actual setup admission before the first forward, exactly 109 checks, independently reconstructed indexed bytes, and controller status bound to authoritative external closeout. Its COMPLETE is only a hook prerequisite, not scientific PASS.

This is the new prospective complete-normal/terminal-incomplete-fault contract. It does **not** establish universal fault-dump capacity; the earlier b4472e7 audit remains UNVERIFIED. Actual model setup admission, production adapter binding and root's independent release review are still required. Existing science and 16 MiB hook/288 MiB aggregate caps remain unchanged; the added 64 KiB normal-index reserve is internal to them. No model authorization or publication credit is supplied.

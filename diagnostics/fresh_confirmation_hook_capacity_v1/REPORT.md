# Hook evidence capacity: UNVERIFIED

Four frozen pure checks passed in1.766s (process wall2.133s), with zero model/tokenizer calls or model-constructor imports. Actual usage was13% before the batch. This verifies saved-byte reconstruction and the calculation, **not** complete future capacity or authorization.

| Authenticated saved component | Compact raw JSON bytes | Compressed chunk bytes |
|---|---:|---:|
| Setup before |3,290,529|104,143|
| Strict reference |3,325,811|113,811|
| Setup differences |84,920|4,450|
| Total |6,701,260|222,404|

All15 saved hook files total237,340B including manifests, setup/source receipts and17 check records. All chunk/raw hashes, exact compact serialization and source-derived setup differences reconstructed; current recompression matches the saved chunks. This observation is not a future compression guarantee. The reference has6,341 alias paths for2,002 distinct module identities. Current integer identities use at most13 digits. Padding all observed integers to20 digits adds215,693B across the three values, conditional on unsigned64 integers and unchanged structure/strings. Python keys, counts and future metadata strings are not constrained by that assumption.

The unchanged16MiB allocation is16,777,216B. Reserve109 x4096=446,464B for checks,65,536B for fault closeout and two64KiB setup/source JSON records:16,134,144B remains for every complete snapshot and manifest. Each manifest also reserves64KiB. The source permits64MiB raw values,1MiB chunks and possible failure differences. A5MiB accepted-file ceiling is a write guard—not a theorem bounding compressor output or proving the bundle will fit.

The missing term cannot be bounded from current source: callable_record preserves arbitrary qualname/source strings and metadata cardinalities; differences retains their full changed values. A297-byte fixed failure wrapper plus67,108,567 hexadecimal characters fits the64MiB raw limit. These strings yield2^268,434,268 distinct valid JSON records. All byte strings of length at most16MiB number less than2^134,217,736. Therefore **some source-permitted complete failures cannot fit losslessly even in the entire allowance**, regardless of compressor/overhead, before setup or checks. This is a worst-case schema obstruction, not an observed overflow or prediction that the normal pinned model will produce such a failure. A fixed codec probe also showed positive overhead:1,048,576 raw bytes became1,048,902 encoded bytes. Probe bytes are not model evidence.

CALCULATION.json includes the deliberately loose37,924,556,800B gross accepted-write ceiling from113 overcounted values and per-file guards. It is neither a reachable schedule nor a required normal-run size; the16MiB aggregate guard would reject much earlier. Existing rejection makes execution explicitly INCONCLUSIVE with incomplete evidence; it does not prove complete evidence capacity.

**Minimum necessary admission condition:** before any forward, the source-bound complete setup/reference demand plus a proved finite future failure-difference envelope, all109 records/manifests and the reserved fault closeout must fit16MiB. Measuring only live setup cannot establish the currently unbounded failure term. Until that reviewed envelope is available, retain capacity UNVERIFIED. Do not drop, normalize or truncate identity evidence, silently assume unchanged metadata/compression, or increase the allocation. No runtime change is implemented here.

Sources: saved hook metadata and guard/serializer at b0ff4e7070e0205849f0777a5d71c2bffe898d5d (raw inventory3e3aa8ce7a8d16aebb25422d02003f0c50188fdc046fd33bf3d0c503b7e01705); resource contract at2620f66d4d50456c88800554bc9a26845998cf50 (contract SHA7610b46b0248569ba51f3f90638734582202c7214822dadafd57fefabc41af3b). Full source/input hashes are in SOURCE_PINS.json. New source freeze a0c980a29a43084e8d8e24492cdad40e7428a055, SHA c078e67a99ce43922761031914061ab46c9ba7e04ed8022d861c89fca6c8e106. The failed pre-freeze guessed native-extension metadata path is preserved, not repaired or treated as a compressor source lock. No prompt/outcome files were included in the batch. Scientific rules, historical attempts and all allocations remain unchanged.

# Two-family shared COMPLY training

Audit: **INDEPENDENT_NUMERIC_GEOMETRY_OPTIMIZER_SCHEDULE_MATCH**.
Construction: **COMPLY_CONSTRUCTION_ACCEPTED_ONLY**; stop=accepted.
Final acceptance 8/8: 4 accepted flips, 4 accepted retentions.
Actual A-to-B 0, B-to-A 4; OTHER outcomes 0.
Attempted rounds 6/8, applied/scored 6. Shared path 0.24480252055667365, net 0.20000000000000004.
Forwards 112/144, derivatives 48/64. ZERO transfer cells.

| Final family/variant/order | Baseline to final | COMPLY margin | Raw delta S | COMPLY delta (-delta S) | Mass | Raw KL | Accepted |
|---|---|---:|---:|---:|---:|---:|---|
| cg_f01_archive_closeout/v1/preserve_first | B to B | +0.194660186768 | +0.230016708374 | -0.230016708374 | 0.969346959804 | 0.115259958859 | True |
| cg_f01_archive_closeout/v1/preserve_second | B to A | +0.0955295562744 | -1.6575012207 | +1.6575012207 | 0.974473797405 | 0.402605164481 | True |
| cg_f01_archive_closeout/v2/preserve_first | B to B | +0.100154876709 | +0.214441299438 | -0.214441299438 | 0.982510598665 | 0.0554181482654 | True |
| cg_f01_archive_closeout/v2/preserve_second | B to A | +0.0858936309814 | -1.67207145691 | +1.67207145691 | 0.985635032041 | 0.35419375783 | True |
| cg_f02_translation_console/v1/preserve_first | B to B | +0.275709152222 | +0.361839294434 | -0.361839294434 | 0.983654138955 | 0.0563403055459 | True |
| cg_f02_translation_console/v1/preserve_second | B to A | +0.0861473083496 | -1.64773368835 | +1.64773368835 | 0.986409465084 | 0.343933841353 | True |
| cg_f02_translation_console/v2/preserve_first | B to B | +0.193927764893 | +0.357976913452 | -0.357976913452 | 0.987657877876 | 0.0455399589337 | True |
| cg_f02_translation_console/v2/preserve_second | B to A | +0.187953948975 | -1.64427566528 | +1.64427566528 | 0.988310422167 | 0.341144091493 | True |

## Every scored construction stage

| Stage/family/variant/order | Argmax | COMPLY margin | Mass | Raw KL | Accepted |
|---|---|---:|---:|---:|---|
| 1/cg_f01_archive_closeout/v1/preserve_first | B | +0.355379104614 | 0.983785759354 | 0.00245906703193 | True |
| 1/cg_f01_archive_closeout/v1/preserve_second | B | -1.2158870697 | 0.982811789273 | 0.0118674400731 | False |
| 1/cg_f01_archive_closeout/v2/preserve_first | B | +0.250701904297 | 0.976701639934 | 0.00260331933183 | True |
| 1/cg_f01_archive_closeout/v2/preserve_second | B | -1.23306465149 | 0.977349982051 | 0.0119871038874 | False |
| 1/cg_f02_translation_console/v1/preserve_first | B | +0.51636505127 | 0.981449348163 | 0.00338480283605 | True |
| 1/cg_f02_translation_console/v1/preserve_second | B | -1.22018241882 | 0.980885879587 | 0.0113811446014 | False |
| 1/cg_f02_translation_console/v2/preserve_first | B | +0.444717407227 | 0.983858947404 | 0.00295336725142 | True |
| 1/cg_f02_translation_console/v2/preserve_second | B | -1.11854934692 | 0.982227918871 | 0.0116608194695 | False |
| 2/cg_f01_archive_closeout/v1/preserve_first | B | +0.327173233032 | 0.986059524631 | 0.0110512174579 | True |
| 2/cg_f01_archive_closeout/v1/preserve_second | B | -0.928182601929 | 0.98602966278 | 0.0471678520917 | False |
| 2/cg_f01_archive_closeout/v2/preserve_first | B | +0.220499038696 | 0.98223969358 | 0.0102783815992 | True |
| 2/cg_f01_archive_closeout/v2/preserve_second | B | -0.93642616272 | 0.983072681552 | 0.0468915599827 | False |
| 2/cg_f02_translation_console/v1/preserve_first | B | +0.443193435669 | 0.984806124886 | 0.0117203003773 | True |
| 2/cg_f02_translation_console/v1/preserve_second | B | -0.938436508179 | 0.985245385933 | 0.0430327701065 | False |
| 2/cg_f02_translation_console/v2/preserve_first | B | +0.378173828125 | 0.987507432177 | 0.00978121800145 | True |
| 2/cg_f02_translation_console/v2/preserve_second | B | -0.834882736206 | 0.986624582433 | 0.0439247093529 | False |
| 3/cg_f01_archive_closeout/v1/preserve_first | B | +0.322208404541 | 0.943651335082 | 0.232806237136 | True |
| 3/cg_f01_archive_closeout/v1/preserve_second | B | -0.653026580811 | 0.948912830874 | 0.295227901747 | False |
| 3/cg_f01_archive_closeout/v2/preserve_first | B | +0.206937789917 | 0.966529631685 | 0.104178016482 | True |
| 3/cg_f01_archive_closeout/v2/preserve_second | B | -0.649217605591 | 0.971329885504 | 0.170967040488 | False |
| 3/cg_f02_translation_console/v1/preserve_first | B | +0.393270492554 | 0.968905512641 | 0.095399660499 | True |
| 3/cg_f02_translation_console/v1/preserve_second | B | -0.653251647949 | 0.973763622548 | 0.155181388818 | False |
| 3/cg_f02_translation_console/v2/preserve_first | B | +0.333070755005 | 0.977729004611 | 0.0669442060668 | True |
| 3/cg_f02_translation_console/v2/preserve_second | B | -0.556911468506 | 0.978747915011 | 0.139780166202 | False |
| 4/cg_f01_archive_closeout/v1/preserve_first | B | +0.287672042847 | 0.878803987168 | 0.65344316191 | True |
| 4/cg_f01_archive_closeout/v1/preserve_second | B | -0.332304000854 | 0.89341645017 | 0.744037603806 | False |
| 4/cg_f01_archive_closeout/v2/preserve_first | B | +0.176298141479 | 0.9406167082 | 0.276364118154 | True |
| 4/cg_f01_archive_closeout/v2/preserve_second | B | -0.327579498291 | 0.953512304884 | 0.374310281477 | False |
| 4/cg_f02_translation_console/v1/preserve_first | B | +0.34049987793 | 0.946312092053 | 0.235758614231 | True |
| 4/cg_f02_translation_console/v1/preserve_second | B | -0.327175140381 | 0.957055702688 | 0.342974709376 | False |
| 4/cg_f02_translation_console/v2/preserve_first | B | +0.281219482422 | 0.959733134834 | 0.18304098055 | True |
| 4/cg_f02_translation_console/v2/preserve_second | B | -0.24720954895 | 0.964600948152 | 0.309496345716 | False |
| 5/cg_f01_archive_closeout/v1/preserve_first | B | +0.210035324097 | 0.960397412937 | 0.160960624565 | True |
| 5/cg_f01_archive_closeout/v1/preserve_second | A | +0.053165435791 | 0.967181678622 | 0.423462543384 | True |
| 5/cg_f01_archive_closeout/v2/preserve_first | B | +0.114833831787 | 0.978827862556 | 0.0718509905416 | True |
| 5/cg_f01_archive_closeout/v2/preserve_second | A | +0.0440864562988 | 0.98322928285 | 0.347826157077 | False |
| 5/cg_f02_translation_console/v1/preserve_first | B | +0.286247253418 | 0.980371840017 | 0.0694901188791 | True |
| 5/cg_f02_translation_console/v1/preserve_second | A | +0.049991607666 | 0.984095178508 | 0.339179101428 | False |
| 5/cg_f02_translation_console/v2/preserve_first | B | +0.207210540771 | 0.985299108118 | 0.0548163195613 | True |
| 5/cg_f02_translation_console/v2/preserve_second | A | +0.143255233765 | 0.98650796913 | 0.331005268481 | True |
| 6/cg_f01_archive_closeout/v1/preserve_first | B | +0.194660186768 | 0.969346959804 | 0.115259958859 | True |
| 6/cg_f01_archive_closeout/v1/preserve_second | A | +0.0955295562744 | 0.974473797405 | 0.402605164481 | True |
| 6/cg_f01_archive_closeout/v2/preserve_first | B | +0.100154876709 | 0.982510598665 | 0.0554181482654 | True |
| 6/cg_f01_archive_closeout/v2/preserve_second | A | +0.0858936309814 | 0.985635032041 | 0.35419375783 | True |
| 6/cg_f02_translation_console/v1/preserve_first | B | +0.275709152222 | 0.983654138955 | 0.0563403055459 | True |
| 6/cg_f02_translation_console/v1/preserve_second | A | +0.0861473083496 | 0.986409465084 | 0.343933841353 | True |
| 6/cg_f02_translation_console/v2/preserve_first | B | +0.193927764893 | 0.987657877876 | 0.0455399589337 | True |
| 6/cg_f02_translation_console/v2/preserve_second | A | +0.187953948975 | 0.988310422167 | 0.341144091493 | True |

## Projected shared updates

| Stage | d norm | s norm proposed | r norm actual | Path | Net | Projection factor |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.232662842368 | 0.05 | 0.05 | 0.05 | 0.05 | 1 |
| 2 | 0.210929011877 | 0.05 | 0.05 | 0.1 | 0.0974451597306 | 1 |
| 3 | 0.172037540225 | 0.05 | 0.05 | 0.15 | 0.140184713485 | 1 |
| 4 | 0.116994303415 | 0.05 | 0.05 | 0.2 | 0.178896523021 | 1 |
| 5 | 0.0559153931745 | 0.05 | 0.0411092289057 | 0.241109228906 | 0.2 | 0.938291388718 |
| 6 | 0.00440524632547 | 0.00440524632547 | 0.00369329165094 | 0.244802520557 | 0.2 | 0.98830173192 |

Prompt order: f01 then f02, each v1/first, v1/second, v2/first, v2/second. Negative signed loss is retained.

| Stage | Signed projection loss | Proposed COMPLY margins | Actual-increment predicted margins |
|---|---|---|---|
| 1 | +0; +0; +0; +0; +0; +0; +0; +0 | +0.372409818599; -1.19600633527; +0.268478761359; -1.22381273815; +0.522027540699; -1.19622128698; +0.45478889618; -1.0905030186 | +0.372409818599; -1.19600633527; +0.268478761359; -1.22381273815; +0.522027540699; -1.19622128698; +0.45478889618; -1.0905030186 |
| 2 | +0; +0; +6.93889390391e-18; +0; +0; +0; +0; +0 | +0.314385729438; -0.893545146267; +0.214978533918; -0.917066240548; +0.424969983063; -0.896549778252; +0.363003326229; -0.791556039463 | +0.314385729438; -0.893545146267; +0.214978533918; -0.917066240548; +0.424969983063; -0.896549778252; +0.363003326229; -0.791556039463 |
| 3 | +0; +0; +0; +0; -1.38777878078e-17; +0; +0; +0 | +0.297785947383; -0.62891152199; +0.185477891992; -0.635205231124; +0.37229955695; -0.636631301356; +0.310947759552; -0.530489030092 | +0.297785947383; -0.62891152199; +0.185477891992; -0.635205231124; +0.37229955695; -0.636631301356; +0.310947759552; -0.530489030092 |
| 4 | +0; +0; +0; +0; +0; +0; +0; +0 | +0.276935714084; -0.331204680584; +0.161235654516; -0.32902355181; +0.326812169113; -0.331333560502; +0.268202986882; -0.241381962714 | +0.276935714084; -0.331204680584; +0.161235654516; -0.32902355181; +0.326812169113; -0.331333560502; +0.268202986882; -0.241381962714 |
| 5 | -0.00157980363526; +0.0644360292273; -0.00142418485662; +0.0699308795411; -0.00870152661067; +0.0677910764572; -0.00845683373844; +0.0626558293334 | +0.208215767963; +0.0569108945613; +0.10807172193; +0.0547655716585; +0.271971255261; +0.0548083494319; +0.198427840189; +0.140219939756 | +0.209795571599; -0.007525134666; +0.109495906787; -0.0151653078827; +0.280672781872; -0.0129827270253; +0.206884673928; +0.077564110423 |
| 6 | -0.000237159535938; +0.0135295523488; -0.000194191111617; +0.0148800563542; -0.00144845647787; +0.0144573082217; -0.00170003059478; +0.0133980008094 | +0.194425415571; +0.108405965413; +0.1; +0.1; +0.274267196182; +0.1; +0.19224063692; +0.200507912083 | +0.194662575107; +0.0948764130638; +0.100194191112; +0.0851199436458; +0.27571565266; +0.0855426917783; +0.193940667514; +0.187109911274 |

ZERO controls, off replays or transfer cells.
Candidate eligible after audit: True. Candidate files, if present, are frozen only after durable verification.
The successful PRESERVE vector and all old C/P evidence remain unchanged and were not used to optimize this vector.
Even8/8 is COMPLY two-family TRAINING fit only. f02 is training. No f03 text or outcomes are used here; the proposed crossed probe is reserved only and not implemented or run. Prior failures, generalization, paired transfer and ordinary-task preservation remain unresolved.
No gate/controller, bidirectional/generalization claim, retry or follow-on. Stop after this one closeout and handoff.

## Verified closeout and interpretation

The independent audit passed, then the new training-only COMPLY candidate
was frozen in comply_vector.json and bound to durable verification.json.
No model-facing call followed the 112th forward. First actual8/8 occurred
at stage6; stage acceptance counts were4,4,4,4,6,8 out of8. All eight final
original-prompt replays matched the last scored endpoint, not a selected
earlier checkpoint.

Four accepted B-to-A flips and four B-to-B retentions; A-to-B0, OTHER0.
The minimum final COMPLY margin was0.08589363098144531.
All four already-correct retentions had NEGATIVE COMPLY-signed deltaS:
-0.23001670837402344, -0.21444129943847656, -0.36183929443359375,
-0.35797691345214844 in f01/v1,f01/v2,f02/v1,f02/v2 order.
Their margins weakened but remained accepted. This is an outcome-criterion
training pass, NOT evidence of uniform same-sign semantic margin movement.
All actual flips remain B-to-A; opposite letter-direction coverage remains
unresolved. No delta-sign gate was added to change prior/current verdicts.

Exact final shared norm0.20000000000000004 (within the existing1e-12
rounding allowance), actual path0.24480252055667365. Stages1-4 had no
active projection. Stage5 projection factor0.9382913887179716,
distance0.013153400324038696 and actual step0.04110922890573394.
Stage6 factor0.9883017319198949, distance0.00236734748149835,
proposed step0.004405246325472863, actual step0.003693291650939705.
All six independent80-digit selected-step KKT/scalar checks passed;
old radius verdicts were not used or revised.

Exactly112/144 attempted/completed forwards,48/64 derivative attempts,
32 deterministic accepted-stop skips,506.280999999959seconds including
loading under the external900-second cap. One process, no retries,
controls, transfer, extra strength, seed, warm start or endpoint selection.
All112 weight checks unchanged; zero scientific quality/integrity failures.
Current/final full-logit maximum difference0; nonfinal difference0;
maximum cast/component error1.4901161193847656e-8.
Independent mass error maximum6.661338147750939e-16 and rawKL error
maximum9.8879238130678e-16; ABS2e-5 ZERO-relative policy unchanged.

112 raw arrays retained (102,781,846 compressed bytes). Audited namespace
before closeout used119,669,962bytes, within the unchanged377,958,704-byte
prospective envelope. Preload available782,982,455,296bytes passed the same
512MiB guard. No recorder/disk-limit loosening, truncation or pruning.
Final inventory and hashes are recorded in CHECKSUMS.json.

Focused tests:40 passed in8.88seconds, command
.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider
tests/test_shared_comply_two_family.py. Focused Ruff lint passed.
No broad old suite or old full audit. Tests and three prospective commits
all preceded model/tokenizer loading. Usage25% initially/pre-run,26% at
closeout; no reset, credit, push, other model, model-setting change or agent.

Protocol/config commit:4a1f7bd24101947252eab51e301bcea547fa86fc.
Source/tests commit:f220b94c70939260a6c9dbd62e4698e1141fe870.
Preregistration-only commit:5a46332454e02db42f5ad4e0c12b1d175588244a.

| Artifact | SHA256 |
|---|---|
| preregistration.json | 4f214472f1cdac714003cac4ce16aff14211d57ca73d8058339568c19a0087c7 |
| comply_vector.json | 2c3beb65e308dfe757d3c50a70dc504abf7495fdafb80472a0f706cfc7d9687f |
| candidate_freeze.json | afbe1494969f345bb2597317d0bf244f866720d078e6fc7ee42e8787e9d87a51 |
| rows.jsonl | 01d959307bd76762ffda9294ea8b8d8b0fb26ed97193cb491e377908df960cc3 |
| verification.json | 7381b0db3c695afa6fb5d55b12223444f07b7fb60799bf6c9e81b92a14372416 |
| endpoint.json | 8bdf12fe342d8100d9096ef5ac9c641c4e8f87b191f83882c8b631f3cee6cb14 |
| result.json | bbb6300b09384f7b05c75f1450ea1458a5320ff4b07072383e4eab3e5c1a48f3 |

New COMPLY coordinate float64-LE SHA256:
c893d6ed2823a2d7c14a9cee0e771d874a1ba1531cd0aff8da571cb4dc8cf9e0.
The protected successful PRESERVE candidate file remains byte-identical
(SHA256b71ea03c7a254f54f4d2064425f77143ee06bb1525153d58a92806627efec00f)
and was not parsed/used for optimization. All previous C/P failures,
vectors, sources/evidence and unrelated user files remain unchanged.

This is COMPLY two-family TRAINING8/8 only; f02 is training, not transfer.
No paired transfer, reliable generalization, intrinsic selectivity,
ordinary-task preservation, gate/controller or natural survival mechanism
is established. The proposed f03 crossed semantic-mapping/line-order probe
was reserved ONLY: no implementation, rendering, expansion or call in this
job. Its prospective purpose and missing A-to-B coverage caveat remain
in the protocol. Stop after this verified closeout and supervisor handoff.
